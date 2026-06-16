from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
from transformers import AutoTokenizer

from controller.engine.base import SplitEngine
from controller.remote_middle import RemoteMiddleClient
from controller.schemas import (
    ChatCompletionRequest, ChatCompletionResponse, ChatChoice,
    ChatCompletionChunk, ChatMessageResponse, StreamChoice, StreamDelta,
)


class OrchestratorError(RuntimeError):
    pass


@dataclass
class GenerationResult:
    text: str
    finish_reason: str


def _normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    pieces = []
    for item in content:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if text:
            pieces.append(text)
    return "".join(pieces)


def _sample(logits: np.ndarray, temperature: float, top_k: int) -> int:
    logits = logits.astype(np.float64)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits /= temperature
    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones(len(logits), dtype=bool)
        mask[idx] = False
        logits[mask] = -np.inf
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


class SplitChatCompletionRunner:
    def __init__(
        self,
        engine: SplitEngine,
        remote_middle: RemoteMiddleClient,
        tokenizer_dir: str,
        max_len: int,
        model_name: str,
    ):
        self.engine = engine
        self.remote_middle = remote_middle
        self.max_len = max_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)

    def _format_messages(self, request: ChatCompletionRequest) -> str:
        msgs = [{"role": m.role, "content": _normalize_message_content(m.content)}
                for m in request.messages]
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.enable_thinking,
        )

    @staticmethod
    def _normalize_stop(stop: str | list[str] | None) -> list[str]:
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop]
        return [x for x in stop if x]

    @staticmethod
    def _apply_stop(text: str, stops: Iterable[str]) -> tuple[str, bool]:
        best = None
        for stop in stops:
            pos = text.find(stop)
            if pos >= 0 and (best is None or pos < best):
                best = pos
        if best is None:
            return text, False
        return text[:best], True

    def _generate(self, request: ChatCompletionRequest) -> Generator[tuple[str, str | None], None, None]:
        prompt = self._format_messages(request)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise OrchestratorError("empty prompt ids")
        prompt_ids = prompt_ids[-self.max_len :]
        stops = self._normalize_stop(request.stop)

        session_id = uuid.uuid4().hex
        self.engine.start_session()
        self.remote_middle.open(session_id)
        finish_reason = "stop"
        output_text = ""
        try:
            logits = None
            for pos, token_id in enumerate(prompt_ids):
                hidden_l4 = self.engine.run_prefix(int(token_id), pos)
                hidden_l20, _ = self.remote_middle.step(session_id, hidden_l4, pos)
                logits = self.engine.run_suffix(hidden_l20, pos)

            assert logits is not None
            current_logits = logits
            pos = len(prompt_ids)
            for _ in range(request.max_tokens):
                if pos >= self.max_len:
                    finish_reason = "length"
                    break
                token_id = _sample(current_logits[0, 0, :], request.temperature, request.top_k)
                if token_id == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
                next_text = output_text + piece
                trimmed, hit_stop = self._apply_stop(next_text, stops)
                delta = trimmed[len(output_text) :]
                if delta:
                    output_text = trimmed
                    yield delta, None
                if hit_stop:
                    finish_reason = "stop"
                    break
                hidden_l4 = self.engine.run_prefix(int(token_id), pos)
                hidden_l20, _ = self.remote_middle.step(session_id, hidden_l4, pos)
                current_logits = self.engine.run_suffix(hidden_l20, pos)
                pos += 1
            yield "", finish_reason
        finally:
            try:
                self.remote_middle.close(session_id)
            finally:
                self.engine.end_session()

    def run_non_stream(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        chunks = []
        finish_reason = "stop"
        for delta, maybe_finish in self._generate(request):
            if delta:
                chunks.append(delta)
            if maybe_finish is not None:
                finish_reason = maybe_finish
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=created,
            model=self.model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessageResponse(role="assistant", content="".join(chunks)),
                    finish_reason=finish_reason,
                )
            ],
        )

    def run_stream(self, request: ChatCompletionRequest) -> Generator[str, None, None]:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        first = ChatCompletionChunk(
            id=request_id,
            object="chat.completion.chunk",
            created=created,
            model=self.model_name,
            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=""),
                                  finish_reason=None)],
        )
        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
        for delta, maybe_finish in self._generate(request):
            if delta:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=self.model_name,
                    choices=[StreamChoice(index=0, delta=StreamDelta(content=delta),
                                          finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
            if maybe_finish is not None:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    object="chat.completion.chunk",
                    created=created,
                    model=self.model_name,
                    choices=[StreamChoice(index=0, delta=StreamDelta(),
                                          finish_reason=maybe_finish)],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
