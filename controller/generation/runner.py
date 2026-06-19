from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from controller.generation.config import SamplingParams
from controller.generation.logits_processors import apply_presence_penalty, apply_repetition_penalty
from controller.generation.strategies import greedy_select, sample_with_top_k_top_p
from controller.modeling.base import Qwen35Model


class GenerationCancelled(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


@dataclass
class GenerationStep:
    delta_text: str
    finish_reason: str | None


def normalize_stop(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return [item for item in stop if item]


def apply_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    best = None
    for stop in stops:
        pos = text.find(stop)
        if pos >= 0 and (best is None or pos < best):
            best = pos
    if best is None:
        return text, False
    return text[:best], True


class TokenGenerationRunner:
    def __init__(self, model: Qwen35Model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def raise_if_cancelled(cancel_event) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("request cancelled")

    def _flush_pending_output(
        self,
        pending_output_ids: list[int],
        output_text: str,
        stops: list[str],
    ) -> tuple[str, str, bool]:
        if not pending_output_ids:
            return output_text, "", False
        decoded = self.tokenizer.decode_tokens(pending_output_ids, skip_special_tokens=True)
        if not decoded or "\ufffd" in decoded:
            return output_text, "", False
        candidate = output_text + decoded
        trimmed, hit_stop = apply_stop(candidate, stops)
        delta = trimmed[len(output_text) :]
        pending_output_ids.clear()
        return trimmed, delta, hit_stop

    def generate(
        self,
        prompt_ids: list[int],
        params: SamplingParams,
        cancel_event=None,
    ) -> Generator[GenerationStep, None, None]:
        if not prompt_ids:
            raise GenerationError("empty prompt ids")
        prompt_ids = prompt_ids[-self.model.info.max_len :]
        think_end_ids = self.tokenizer.encode_prompt("</think>")
        think_end_token_id = think_end_ids[0] if len(think_end_ids) == 1 else None
        thinking_phase = params.enable_thinking
        thinking_token_budget = 256
        thinking_tokens = 0
        pending_output_ids: list[int] = []
        token_counts: dict[int, int] = {}
        output_text = ""
        finish_reason = "stop"
        session = self.model.create_session()
        try:
            self.raise_if_cancelled(cancel_event)
            current_logits = session.prefill(prompt_ids)
            for _ in range(params.max_new_tokens):
                self.raise_if_cancelled(cancel_event)
                if session.position >= self.model.info.max_len:
                    finish_reason = "length"
                    break
                token_id = self._select_token(current_logits[0, 0, :], params, token_counts)
                next_token_id = token_id
                if thinking_phase:
                    if think_end_token_id is not None and token_id == think_end_token_id:
                        thinking_phase = False
                    elif token_id == self.tokenizer.eos_token_id and think_end_token_id is not None:
                        next_token_id = think_end_token_id
                        thinking_phase = False
                    elif think_end_token_id is not None and thinking_tokens >= thinking_token_budget:
                        next_token_id = think_end_token_id
                        thinking_phase = False
                    else:
                        thinking_tokens += 1
                else:
                    if token_id == self.tokenizer.eos_token_id:
                        output_text, delta, hit_stop = self._flush_pending_output(
                            pending_output_ids,
                            output_text,
                            params.stop,
                        )
                        if delta:
                            yield GenerationStep(delta_text=delta, finish_reason=None)
                        if hit_stop:
                            finish_reason = "stop"
                        finish_reason = "stop"
                        break
                    pending_output_ids.append(token_id)
                    output_text, delta, hit_stop = self._flush_pending_output(
                        pending_output_ids,
                        output_text,
                        params.stop,
                    )
                    if delta:
                        yield GenerationStep(delta_text=delta, finish_reason=None)
                    if hit_stop:
                        finish_reason = "stop"
                        break
                token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
                self.raise_if_cancelled(cancel_event)
                current_logits = session.decode_next(int(next_token_id))
            if pending_output_ids:
                output_text, delta, _ = self._flush_pending_output(
                    pending_output_ids,
                    output_text,
                    params.stop,
                )
                if delta:
                    yield GenerationStep(delta_text=delta, finish_reason=None)
            yield GenerationStep(delta_text="", finish_reason=finish_reason)
        finally:
            session.close()

    def _select_token(
        self,
        logits: np.ndarray,
        params: SamplingParams,
        token_counts: dict[int, int],
    ) -> int:
        work = logits.astype(np.float64).copy()
        work = apply_presence_penalty(work, token_counts, params.presence_penalty)
        work = apply_repetition_penalty(work, token_counts, params.repetition_penalty)
        if params.temperature <= 0:
            return greedy_select(work)
        return sample_with_top_k_top_p(work, params.temperature, params.top_k, params.top_p)
