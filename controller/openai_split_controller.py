#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import queue
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen35_model_spec import ModelSpec, SplitConfig, load_bound_embed_head_metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from controller.orchestrator import (
    OrchestratorCancelled,
    OrchestratorError,
    SplitChatCompletionRunner,
)
from controller.remote_middle import RemoteMiddleClient, RemoteMiddleError
from controller.schemas import ChatCompletionRequest, ErrorBody, ErrorEnvelope, ModelCard, ModelListResponse
from controller.engine.om_engine import OmEngineError

RUNNER: SplitChatCompletionRunner | None = None


def parse_split(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("split must be 'prefix_end,suffix_start', e.g. '4,20'")
    return int(parts[0]), int(parts[1])


def _make_model_name(model_spec: ModelSpec, split: tuple[int, int], engine: str) -> str:
    size_map = {1024: "0.8B", 2048: "2B", 2560: "4B", 4096: "9B", 5120: "27B"}
    hs = model_spec.hidden_size
    size_str = size_map.get(hs, f"{hs}")
    prefix_ct = split[0]
    suffix_ct = model_spec.num_hidden_layers - split[1]
    middle_ct = split[1] - split[0]
    return f"qwen3.5-{size_str}-split-{prefix_ct}-{middle_ct}-{suffix_ct}-{engine}"


def build_app(args) -> FastAPI:
    global RUNNER

    bound_config = None
    if args.engine == "om" and args.om_mode == "bound_embed_head":
        if not args.bound_asset_dir:
            raise ValueError("bound_embed_head mode requires --bound-asset-dir")
        model_spec, split_config, bound_config = load_bound_embed_head_metadata(args.bound_asset_dir)
    else:
        # Load model spec — try from_pretrained, fall back to metadata.json for OM
        try:
            model_spec = ModelSpec.from_pretrained(args.model_path)
        except Exception:
            if args.engine == "om" and args.prefix_om:
                from scripts.qwen35_model_spec import load_metadata
                model_spec, _, _, _ = load_metadata(
                    args.prefix_om.replace(".om", ".metadata.json"))
            else:
                raise
        split_config = SplitConfig(args.split[0], args.split[1], model_spec.num_hidden_layers)
    effective_split = (split_config.prefix_end, split_config.suffix_start)
    model_name = args.model_name or _make_model_name(model_spec, effective_split, args.engine)

    if args.engine == "onnx":
        from controller.engine.onnx_engine import OnnxSplitEngine
        if not args.prefix_onnx or not args.suffix_onnx:
            raise ValueError("onnx engine requires --prefix-onnx and --suffix-onnx")
        engine = OnnxSplitEngine(
            model_id=model_name,
            max_len=args.max_len,
            model_spec=model_spec,
            split_config=split_config,
            prefix_onnx=args.prefix_onnx,
            suffix_onnx=args.suffix_onnx,
        )
    else:
        from controller.engine.om_engine import OmSplitEngine
        if args.om_mode == "om_split" and (not args.prefix_om or not args.suffix_om):
            raise ValueError("om_split mode requires --prefix-om and --suffix-om")
        engine = OmSplitEngine(
            model_id=model_name,
            max_len=args.max_len,
            model_spec=model_spec,
            split_config=split_config,
            prefix_om=args.prefix_om,
            suffix_om=args.suffix_om,
            mode=args.om_mode,
            bound_asset_dir=args.bound_asset_dir,
            bound_config=bound_config,
        )

    remote = RemoteMiddleClient(
        server_url=args.server_url,
        model_name=args.remote_model_name or model_name,
        hidden_size=model_spec.hidden_size,
        max_len=args.max_len,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        checksum=args.checksum,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global RUNNER
        app.state.engine_loaded = False
        app.state.engine_load_error = None
        RUNNER = SplitChatCompletionRunner(
            engine=engine,
            remote_middle=remote,
            tokenizer_dir=args.tokenizer_dir,
            max_len=args.max_len,
            model_name=model_name,
        )
        yield
        engine.close()
        app.state.engine_loaded = False

    app = FastAPI(lifespan=lifespan)

    def _error_response(status_code: int, message: str, code: str) -> JSONResponse:
        payload = ErrorEnvelope(error=ErrorBody(message=message, code=code))
        return JSONResponse(status_code=status_code, content=payload.model_dump())

    def _ensure_engine_loaded(app: FastAPI) -> None:
        if getattr(app.state, "engine_loaded", False):
            return
        try:
            engine.load()
        except Exception as exc:  # noqa: BLE001
            app.state.engine_load_error = str(exc)
            raise
        app.state.engine_loaded = True
        app.state.engine_load_error = None

    @app.get("/healthz")
    async def healthz():
        try:
            remote_health = remote.health()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "ok": True,
            "model": model_name,
            "engine": args.engine,
            "hidden_size": model_spec.hidden_size,
            "engine_loaded": bool(getattr(app.state, "engine_loaded", False)),
            "engine_load_error": getattr(app.state, "engine_load_error", None),
            "remote": remote_health,
        }

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(data=[ModelCard(id=model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(http_request: FastAPIRequest, request: ChatCompletionRequest):
        if request.model != model_name:
            return _error_response(400, f"unsupported model: {request.model}", "BAD_MODEL")
        assert RUNNER is not None
        try:
            _ensure_engine_loaded(app)
            if request.stream:
                async def stream_with_disconnect_watch():
                    cancel_event = threading.Event()
                    out_queue: queue.Queue[tuple[str, object]] = queue.Queue()

                    def worker() -> None:
                        try:
                            for chunk in RUNNER.run_stream(request, cancel_event=cancel_event):
                                out_queue.put(("chunk", chunk))
                        except Exception as exc:  # noqa: BLE001
                            out_queue.put(("error", exc))
                        finally:
                            out_queue.put(("done", None))

                    thread = threading.Thread(target=worker, daemon=True)
                    thread.start()
                    try:
                        while True:
                            if await http_request.is_disconnected():
                                cancel_event.set()
                            try:
                                kind, payload = out_queue.get_nowait()
                            except queue.Empty:
                                if not thread.is_alive():
                                    break
                                await asyncio.sleep(0.05)
                                continue

                            if kind == "chunk":
                                yield payload
                                continue
                            if kind == "error":
                                if isinstance(payload, OrchestratorCancelled):
                                    break
                                raise payload
                            break
                    finally:
                        cancel_event.set()
                        thread.join(timeout=1.0)

                return StreamingResponse(stream_with_disconnect_watch(),
                                         media_type="text/event-stream")

            cancel_event = threading.Event()

            async def monitor_disconnect() -> None:
                while not cancel_event.is_set():
                    if await http_request.is_disconnected():
                        cancel_event.set()
                        return
                    await asyncio.sleep(0.05)

            response_task = asyncio.create_task(
                asyncio.to_thread(RUNNER.run_non_stream, request, cancel_event)
            )
            disconnect_task = asyncio.create_task(monitor_disconnect())
            done, _ = await asyncio.wait(
                {response_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and cancel_event.is_set():
                try:
                    await response_task
                except OrchestratorCancelled:
                    pass
                raise HTTPException(status_code=499, detail="client disconnected")
            response = await response_task
            cancel_event.set()
            await disconnect_task
            return JSONResponse(content=response.model_dump())
        except OmEngineError as exc:
            app.state.engine_load_error = str(exc)
            return _error_response(503, str(exc), "ENGINE_LOAD_ERROR")
        except OrchestratorCancelled:
            raise HTTPException(status_code=499, detail="client disconnected")
        except (RemoteMiddleError, OrchestratorError) as exc:
            return _error_response(502, str(exc), "GENERATION_ERROR")
        except Exception as exc:  # noqa: BLE001
            if not getattr(app.state, "engine_loaded", False):
                app.state.engine_load_error = str(exc)
            return _error_response(500, str(exc), "INTERNAL_ERROR")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=["onnx", "om"], default="onnx")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--remote-model-name", default="")
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer-dir", default="model/Qwen3.5-0.8B")
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--split", type=parse_split, default=(4, 20),
                        help="prefix_end,suffix_start  (e.g. 4,20 for 4/16/4)")
    parser.add_argument("--prefix-onnx")
    parser.add_argument("--suffix-onnx")
    parser.add_argument("--prefix-om")
    parser.add_argument("--suffix-om")
    parser.add_argument("--om-mode", choices=["om_split", "bound_embed_head"], default="om_split")
    parser.add_argument("--bound-asset-dir", default="")
    parser.add_argument("--connect-timeout", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
