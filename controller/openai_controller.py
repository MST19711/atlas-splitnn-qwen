#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import queue
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from controller.engine.om_engine import OmEngineError
from controller.generation.config import SamplingParams
from controller.generation.runner import GenerationCancelled, GenerationError, TokenGenerationRunner, normalize_stop
from controller.modeling.factory import BackendConfig, create_model, parse_split
from controller.remote_middle import RemoteMiddleError
from controller.schemas import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    ErrorBody,
    ErrorEnvelope,
    ModelCard,
    ModelListResponse,
    StreamChoice,
    StreamDelta,
)
from controller.tokenization.qwen35 import Qwen35TokenizerAdapter

MODEL = None
RUNNER = None
TOKENIZER = None


class OpenAIChatAdapter:
    def __init__(self, model, tokenizer: Qwen35TokenizerAdapter):
        self.model = model
        self.tokenizer = tokenizer
        self.runner = TokenGenerationRunner(model=model, tokenizer=tokenizer)

    @staticmethod
    def resolve_sampling_params(request: ChatCompletionRequest) -> SamplingParams:
        if request.enable_thinking:
            return SamplingParams(
                max_new_tokens=request.max_tokens,
                temperature=1.0 if request.temperature is None else request.temperature,
                top_k=20 if request.top_k is None else request.top_k,
                top_p=0.95 if request.top_p is None else request.top_p,
                presence_penalty=1.5 if request.presence_penalty is None else request.presence_penalty,
                repetition_penalty=1.0 if request.repetition_penalty is None else request.repetition_penalty,
                stop=normalize_stop(request.stop),
                enable_thinking=True,
            )
        return SamplingParams(
            max_new_tokens=request.max_tokens,
            temperature=0.7 if request.temperature is None else request.temperature,
            top_k=40 if request.top_k is None else request.top_k,
            top_p=1.0 if request.top_p is None else request.top_p,
            presence_penalty=0.0 if request.presence_penalty is None else request.presence_penalty,
            repetition_penalty=1.0 if request.repetition_penalty is None else request.repetition_penalty,
            stop=normalize_stop(request.stop),
            enable_thinking=False,
        )

    def build_prompt_ids(self, request: ChatCompletionRequest) -> list[int]:
        prompt = self.tokenizer.format_messages(request.messages, enable_thinking=request.enable_thinking)
        return self.tokenizer.encode_prompt(prompt)

    def run_non_stream(self, request: ChatCompletionRequest, cancel_event=None) -> ChatCompletionResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        params = self.resolve_sampling_params(request)
        prompt_ids = self.build_prompt_ids(request)
        chunks = []
        finish_reason = "stop"
        for step in self.runner.generate(prompt_ids, params, cancel_event=cancel_event):
            if step.delta_text:
                chunks.append(step.delta_text)
            if step.finish_reason is not None:
                finish_reason = step.finish_reason
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created,
            model=self.model.info.model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessageResponse(role="assistant", content="".join(chunks)),
                    finish_reason=finish_reason,
                )
            ],
        )

    def run_stream(self, request: ChatCompletionRequest, cancel_event=None):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        first = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created,
            model=self.model.info.model_name,
            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=""), finish_reason=None)],
        )
        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
        params = self.resolve_sampling_params(request)
        prompt_ids = self.build_prompt_ids(request)
        for step in self.runner.generate(prompt_ids, params, cancel_event=cancel_event):
            if step.delta_text:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=self.model.info.model_name,
                    choices=[StreamChoice(index=0, delta=StreamDelta(content=step.delta_text), finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if step.finish_reason is not None:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=self.model.info.model_name,
                    choices=[StreamChoice(index=0, delta=StreamDelta(), finish_reason=step.finish_reason)],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"


def build_app(args) -> FastAPI:
    global MODEL, RUNNER, TOKENIZER
    config = BackendConfig(
        backend=args.backend,
        model_name=args.model_name,
        remote_model_name=args.remote_model_name,
        tokenizer_dir=args.tokenizer_dir,
        max_len=args.max_len,
        server_url=args.server_url,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        checksum=args.checksum,
        model_path=args.model_path,
        split=args.split,
        prefix_onnx=args.prefix_onnx,
        suffix_onnx=args.suffix_onnx,
        prefix_om=args.prefix_om,
        suffix_om=args.suffix_om,
        bound_asset_dir=args.bound_asset_dir,
        model_om=args.model_om,
    )
    model = create_model(config)
    tokenizer = Qwen35TokenizerAdapter(args.tokenizer_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global MODEL, RUNNER, TOKENIZER
        MODEL = model
        TOKENIZER = tokenizer
        RUNNER = OpenAIChatAdapter(model=model, tokenizer=tokenizer)
        app.state.model_loaded = False
        app.state.model_load_error = None
        if model.info.backend_kind == "qwen35_kvcache_om":
            try:
                model.load()
            except Exception as exc:  # noqa: BLE001
                app.state.model_load_error = str(exc)
            else:
                app.state.model_loaded = True
                app.state.model_load_error = None
        yield
        model.close()
        app.state.model_loaded = False

    app = FastAPI(lifespan=lifespan)

    def error_response(status_code: int, message: str, code: str) -> JSONResponse:
        payload = ErrorEnvelope(error=ErrorBody(message=message, code=code))
        return JSONResponse(status_code=status_code, content=payload.model_dump())

    def is_model_loaded() -> bool:
        if bool(getattr(app.state, "model_loaded", False)):
            return True
        runtime = getattr(model, "runtime", None)
        if runtime is not None and getattr(runtime, "acl", None) is not None:
            return True
        engine = getattr(model, "engine", None)
        if engine is not None and bool(getattr(engine, "_loaded", False)):
            return True
        return False

    def ensure_model_loaded() -> None:
        if is_model_loaded():
            app.state.model_loaded = True
            return
        try:
            model.load()
        except Exception as exc:  # noqa: BLE001
            app.state.model_load_error = str(exc)
            raise
        app.state.model_loaded = True
        app.state.model_load_error = None

    @app.get("/healthz")
    async def healthz():
        payload = {
            "ok": True,
            "model": model.info.model_name,
            "backend": model.info.backend_kind,
            "max_len": model.info.max_len,
            "model_loaded": is_model_loaded(),
            "model_load_error": getattr(app.state, "model_load_error", None),
        }
        remote = getattr(model, "remote_middle", None)
        if remote is not None:
            payload["remote"] = remote.health()
        return payload

    @app.get("/v1/models")
    async def list_models():
        return ModelListResponse(data=[ModelCard(id=model.info.model_name)])

    @app.post("/v1/chat/completions")
    async def chat_completions(http_request: FastAPIRequest, request: ChatCompletionRequest):
        if request.model != model.info.model_name:
            return error_response(400, f"unsupported model: {request.model}", "BAD_MODEL")
        try:
            ensure_model_loaded()
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
                                if isinstance(payload, GenerationCancelled):
                                    break
                                error = ErrorEnvelope(
                                    error=ErrorBody(
                                        message=str(payload),
                                        code="GENERATION_ERROR",
                                    )
                                )
                                yield f"data: {error.model_dump_json(exclude_none=True)}\n\n"
                                yield "data: [DONE]\n\n"
                                break
                            break
                    finally:
                        cancel_event.set()
                        thread.join(timeout=1.0)

                return StreamingResponse(stream_with_disconnect_watch(), media_type="text/event-stream")

            cancel_event = threading.Event()
            response = RUNNER.run_non_stream(request, cancel_event)
            return JSONResponse(content=response.model_dump())
        except OmEngineError as exc:
            app.state.model_load_error = str(exc)
            return error_response(503, str(exc), "MODEL_LOAD_ERROR")
        except GenerationCancelled:
            raise HTTPException(status_code=499, detail="client disconnected")
        except (RemoteMiddleError, GenerationError) as exc:
            return error_response(502, str(exc), "GENERATION_ERROR")
        except Exception as exc:  # noqa: BLE001
            if not getattr(app.state, "model_loaded", False):
                app.state.model_load_error = str(exc)
            return error_response(500, str(exc), "INTERNAL_ERROR")

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend",
        choices=["splitnn_om", "splitnn_bound_embed_head", "splitnn_onnx", "qwen35_kvcache_om"],
        default="splitnn_om",
    )
    parser.add_argument("--model-name", default="")
    parser.add_argument("--remote-model-name", default="")
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--model-om", default="")
    parser.add_argument("--tokenizer-dir", default="model/Qwen3.5-0.8B")
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--split", type=parse_split, default=(4, 20))
    parser.add_argument("--prefix-onnx")
    parser.add_argument("--suffix-onnx")
    parser.add_argument("--prefix-om")
    parser.add_argument("--suffix-om")
    parser.add_argument("--bound-asset-dir", default="")
    parser.add_argument("--connect-timeout", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
