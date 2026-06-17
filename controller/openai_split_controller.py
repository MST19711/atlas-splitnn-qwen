#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen35_model_spec import ModelSpec, SplitConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from controller.orchestrator import OrchestratorError, SplitChatCompletionRunner
from controller.remote_middle import RemoteMiddleClient, RemoteMiddleError
from controller.schemas import ChatCompletionRequest, ErrorBody, ErrorEnvelope, ModelCard, ModelListResponse

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

    model_spec = ModelSpec.from_pretrained(args.model_path)
    split_config = SplitConfig(args.split[0], args.split[1], model_spec.num_hidden_layers)
    model_name = args.model_name or _make_model_name(model_spec, args.split, args.engine)

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
        if not args.prefix_om or not args.suffix_om:
            raise ValueError("om engine requires --prefix-om and --suffix-om")
        engine = OmSplitEngine(
            model_id=model_name,
            max_len=args.max_len,
            model_spec=model_spec,
            split_config=split_config,
            prefix_om=args.prefix_om,
            suffix_om=args.suffix_om,
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
        engine.load()
        RUNNER = SplitChatCompletionRunner(
            engine=engine,
            remote_middle=remote,
            tokenizer_dir=args.tokenizer_dir,
            max_len=args.max_len,
            model_name=model_name,
        )
        yield
        engine.close()

    app = FastAPI(lifespan=lifespan)

    def _error_response(status_code: int, message: str, code: str) -> JSONResponse:
        payload = ErrorEnvelope(error=ErrorBody(message=message, code=code))
        return JSONResponse(status_code=status_code, content=payload.model_dump())

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
            "remote": remote_health,
        }

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(data=[ModelCard(id=model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        if request.model != model_name:
            return _error_response(400, f"unsupported model: {request.model}", "BAD_MODEL")
        assert RUNNER is not None
        try:
            if request.stream:
                return StreamingResponse(RUNNER.run_stream(request),
                                         media_type="text/event-stream")
            response = RUNNER.run_non_stream(request)
            return JSONResponse(content=response.model_dump())
        except (RemoteMiddleError, OrchestratorError) as exc:
            return _error_response(502, str(exc), "GENERATION_ERROR")
        except Exception as exc:  # noqa: BLE001
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
    parser.add_argument("--connect-timeout", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
