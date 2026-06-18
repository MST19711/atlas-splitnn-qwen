from __future__ import annotations

import time
import threading
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


class OrchestratorCancelled(OrchestratorError):
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


def _normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    pieces = []
    for item in content:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        if text:
            pieces.append(text)
    return "".join(pieces)


def _sample(
    logits: np.ndarray,
    config: SamplingConfig,
    token_counts: dict[int, int],
) -> int:
    logits = logits.astype(np.float64).copy()
    if config.presence_penalty > 0 or config.repetition_penalty not in (0.0, 1.0):
        for token_id, count in token_counts.items():
            if count <= 0:
                continue
            if config.presence_penalty > 0:
                logits[token_id] -= config.presence_penalty
            if config.repetition_penalty not in (0.0, 1.0):
                if logits[token_id] > 0:
                    logits[token_id] /= config.repetition_penalty
                else:
                    logits[token_id] *= config.repetition_penalty
    if config.temperature <= 0:
        return int(np.argmax(logits))
    logits /= config.temperature
    if 0 < config.top_k < len(logits):
        idx = np.argpartition(logits, -config.top_k)[-config.top_k:]
        mask = np.ones(len(logits), dtype=bool)
        mask[idx] = False
        logits[mask] = -np.inf
    if 0.0 < config.top_p < 1.0:
        order = np.argsort(logits)[::-1]
        sorted_logits = logits[order]
        probs = np.exp(sorted_logits - np.max(sorted_logits))
        probs /= probs.sum()
        cumsum = np.cumsum(probs)
        remove = cumsum > config.top_p
        if np.any(remove):
            # Keep at least one token.
            remove[0] = False
            logits[order[remove]] = -np.inf
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise OrchestratorCancelled("request cancelled")

    @staticmethod
    def _resolve_sampling_config(request: ChatCompletionRequest) -> SamplingConfig:
        if request.enable_thinking:
            return SamplingConfig(
                temperature=1.0 if request.temperature is None else request.temperature,
                top_k=20 if request.top_k is None else request.top_k,
                top_p=0.95 if request.top_p is None else request.top_p,
                presence_penalty=1.5 if request.presence_penalty is None else request.presence_penalty,
                repetition_penalty=1.0 if request.repetition_penalty is None else request.repetition_penalty,
            )
        return SamplingConfig(
            temperature=0.7 if request.temperature is None else request.temperature,
            top_k=40 if request.top_k is None else request.top_k,
            top_p=1.0 if request.top_p is None else request.top_p,
            presence_penalty=0.0 if request.presence_penalty is None else request.presence_penalty,
            repetition_penalty=1.0 if request.repetition_penalty is None else request.repetition_penalty,
        )

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

    def _generate(
        self,
        request: ChatCompletionRequest,
        cancel_event: threading.Event | None = None,
    ) -> Generator[tuple[str, str | None], None, None]:
        if request.enable_thinking and self.engine.model_spec.hidden_size == 2048:
            raise OrchestratorError(
                "Qwen3.5-2B thinking mode is unstable in this deployment and has been disabled. "
                "Please use enable_thinking=false, or switch to a larger Qwen3.5 model for thinking mode."
            )
        prompt = self._format_messages(request)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            raise OrchestratorError("empty prompt ids")
        prompt_ids = prompt_ids[-self.max_len :]
        stops = self._normalize_stop(request.stop)
        sampling = self._resolve_sampling_config(request)
        thinking_phase = request.enable_thinking
        think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        think_end_token_id = think_end_ids[0] if len(think_end_ids) == 1 else None
        thinking_token_budget = 256

        session_id = uuid.uuid4().hex
        self.engine.start_session()
        self.remote_middle.open(session_id)
        finish_reason = "stop"
        output_text = ""
        visible_output_ids: list[int] = []
        token_counts: dict[int, int] = {}
        thinking_tokens = 0
        try:
            logits = None
            for pos, token_id in enumerate(prompt_ids):
                self._raise_if_cancelled(cancel_event)
                hidden_l4 = self.engine.run_prefix(int(token_id), pos)
                self._raise_if_cancelled(cancel_event)
                hidden_l20, _ = self.remote_middle.step(session_id, hidden_l4, pos)
                self._raise_if_cancelled(cancel_event)
                logits = self.engine.run_suffix(hidden_l20, pos)

            assert logits is not None
            current_logits = logits
            pos = len(prompt_ids)
            for _ in range(request.max_tokens):
                self._raise_if_cancelled(cancel_event)
                if pos >= self.max_len:
                    finish_reason = "length"
                    break
                token_id = _sample(current_logits[0, 0, :], sampling, token_counts)
                next_token_id = token_id
                if thinking_phase:
                    if think_end_token_id is not None and token_id == think_end_token_id:
                        thinking_phase = False
                    elif token_id == self.tokenizer.eos_token_id and think_end_token_id is not None:
                        # Qwen3.5-2B may terminate inside thinking mode. Force a close-think token
                        # and continue generation so the model has a chance to produce the final answer.
                        next_token_id = think_end_token_id
                        thinking_phase = False
                    elif think_end_token_id is not None and thinking_tokens >= thinking_token_budget:
                        next_token_id = think_end_token_id
                        thinking_phase = False
                    else:
                        thinking_tokens += 1
                else:
                    if token_id == self.tokenizer.eos_token_id:
                        finish_reason = "stop"
                        break
                    visible_output_ids.append(token_id)
                    decoded_text = self.tokenizer.decode(visible_output_ids, skip_special_tokens=True)
                    trimmed, hit_stop = self._apply_stop(decoded_text, stops)
                    delta = trimmed[len(output_text) :]
                    if delta:
                        output_text = trimmed
                        yield delta, None
                    if hit_stop:
                        finish_reason = "stop"
                        break

                token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
                self._raise_if_cancelled(cancel_event)
                hidden_l4 = self.engine.run_prefix(int(next_token_id), pos)
                self._raise_if_cancelled(cancel_event)
                hidden_l20, _ = self.remote_middle.step(session_id, hidden_l4, pos)
                self._raise_if_cancelled(cancel_event)
                current_logits = self.engine.run_suffix(hidden_l20, pos)
                pos += 1
            yield "", finish_reason
        finally:
            try:
                self.remote_middle.close(session_id)
            finally:
                self.engine.end_session()

    def run_non_stream(
        self,
        request: ChatCompletionRequest,
        cancel_event: threading.Event | None = None,
    ) -> ChatCompletionResponse:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        chunks = []
        finish_reason = "stop"
        for delta, maybe_finish in self._generate(request, cancel_event=cancel_event):
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
            model=self.model_name,
            choices=[StreamChoice(index=0, delta=StreamDelta(role="assistant", content=""),
                                  finish_reason=None)],
        )
        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"
        for delta, maybe_finish in self._generate(request, cancel_event=cancel_event):
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
