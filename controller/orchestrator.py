from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Generator

from controller.generation.config import SamplingParams
from controller.generation.runner import GenerationCancelled, GenerationError, GenerationStep, TokenGenerationRunner, normalize_stop
from controller.modeling.splitnn_qwen35 import SplitNNQwen35Model
from controller.schemas import (
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    StreamChoice,
    StreamDelta,
)
from controller.tokenization.qwen35 import Qwen35TokenizerAdapter


class OrchestratorError(GenerationError):
    pass


class OrchestratorCancelled(GenerationCancelled):
    pass


@dataclass
class GenerationResult:
    text: str
    finish_reason: str


@dataclass
class SamplingConfig:
    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float
    repetition_penalty: float


class OpenAIChatAdapter:
    def __init__(self, model, tokenizer: Qwen35TokenizerAdapter):
        self.model = model
        self.tokenizer = tokenizer
        self.runner = TokenGenerationRunner(model=model, tokenizer=tokenizer)

    @staticmethod
    def _resolve_sampling_config(request: ChatCompletionRequest) -> SamplingParams:
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

    def _build_prompt_ids(self, request: ChatCompletionRequest) -> list[int]:
        prompt = self.tokenizer.format_messages(request.messages, enable_thinking=request.enable_thinking)
        return self.tokenizer.encode_prompt(prompt)

    def generate_steps(
        self,
        request: ChatCompletionRequest,
        cancel_event: threading.Event | None = None,
    ) -> Generator[GenerationStep, None, None]:
        params = self._resolve_sampling_config(request)
        prompt_ids = self._build_prompt_ids(request)
        yield from self.runner.generate(prompt_ids, params, cancel_event=cancel_event)

    def run_non_stream(
        self,
        request: ChatCompletionRequest,
        cancel_event: threading.Event | None = None,
    ) -> ChatCompletionResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        chunks = []
        finish_reason = "stop"
        for step in self.generate_steps(request, cancel_event=cancel_event):
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

    def run_stream(
        self,
        request: ChatCompletionRequest,
        cancel_event: threading.Event | None = None,
    ) -> Generator[str, None, None]:
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
        for step in self.generate_steps(request, cancel_event=cancel_event):
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


class SplitChatCompletionRunner(OpenAIChatAdapter):
    def __init__(self, engine, remote_middle, tokenizer_dir: str, max_len: int, model_name: str):
        del max_len
        model = SplitNNQwen35Model(
            model_name=model_name,
            max_len=engine.max_len,
            vocab_size=engine.model_spec.vocab_size,
            backend_kind="splitnn_compat",
            engine=engine,
            remote_middle=remote_middle,
        )
        tokenizer = Qwen35TokenizerAdapter(tokenizer_dir)
        super().__init__(model=model, tokenizer=tokenizer)
