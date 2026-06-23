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
        cache_info: dict | None = None,
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

        if cache_info is not None:
            cache_info.setdefault("cache_status", "disabled")
            cache_info.setdefault("cache_len", 0)

        registry = self.model.cache_registry
        entry = None

        if registry is not None:
            token_seq = tuple(prompt_ids)
            entry, matched_len, _walked = registry.lookup(token_seq)
            if entry is not None and matched_len > 0:
                entry = registry.acquire(entry)
                session = self.model.create_session(cache_entry=entry)
                try:
                    session.restore(entry.snapshot, entry.position)
                except Exception:
                    session.close()
                    entry = None
            if entry is not None:
                start_pos = entry.position
                delta_ids = prompt_ids[matched_len:]
                if cache_info is not None:
                    cache_info["cache_status"] = "hit"
                    cache_info["cache_len"] = matched_len
            else:
                start_pos = 0
                delta_ids = prompt_ids
                if cache_info is not None:
                    cache_info["cache_status"] = "miss"
                    cache_info["cache_len"] = 0
                session = self.model.create_session()
        else:
            start_pos = 0
            delta_ids = prompt_ids
            session = self.model.create_session()

        generated_ids: list[int] = []
        prompt_snap = None
        supports_prompt_snapshot = True
        try:
            self.raise_if_cancelled(cancel_event)
            if delta_ids:
                current_logits = session.prefill(delta_ids, position=start_pos)
            else:
                # Prompt fully cached – run last token to get first decode logits
                session.position = start_pos - 1
                current_logits = session.prefill([prompt_ids[-1]], position=start_pos - 1)
            # SplitNN's remote middle session is mutable runtime state rather than an
            # immutable snapshot handle, so prompt-only snapshots would pair local
            # prefix/suffix cache with a mismatched remote middle state.
            supports_prompt_snapshot = not hasattr(session, "session_id")
            if registry is not None and supports_prompt_snapshot and len(prompt_ids) > 0 and delta_ids and start_pos == 0:
                try:
                    prompt_snap = session.snapshot()
                except Exception:
                    pass
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
                    pending_output_ids.append(token_id)
                    output_text, delta, _ = self._flush_pending_output(
                        pending_output_ids, output_text, params.stop,
                    )
                    if delta:
                        yield GenerationStep(delta_text=delta, finish_reason=None)
                else:
                    if token_id == self.tokenizer.eos_token_id:
                        output_text, delta, hit_stop = self._flush_pending_output(
                            pending_output_ids, output_text, params.stop,
                        )
                        if delta:
                            yield GenerationStep(delta_text=delta, finish_reason=None)
                        if hit_stop:
                            finish_reason = "stop"
                        finish_reason = "stop"
                        break
                    pending_output_ids.append(token_id)
                    output_text, delta, hit_stop = self._flush_pending_output(
                        pending_output_ids, output_text, params.stop,
                    )
                    if delta:
                        yield GenerationStep(delta_text=delta, finish_reason=None)
                    if hit_stop:
                        finish_reason = "stop"
                        break
                token_counts[next_token_id] = token_counts.get(next_token_id, 0) + 1
                generated_ids.append(next_token_id)
                self.raise_if_cancelled(cancel_event)
                current_logits = session.decode_next(int(next_token_id))
            if pending_output_ids:
                output_text, delta, _ = self._flush_pending_output(
                    pending_output_ids, output_text, params.stop,
                )
                if delta:
                    yield GenerationStep(delta_text=delta, finish_reason=None)
            yield GenerationStep(delta_text="", finish_reason=finish_reason)
        finally:
            if registry is not None and not self._cancelled_reason(cancel_event, finish_reason):
                try:
                    snap = session.snapshot()
                    full_seq = prompt_ids + generated_ids
                    mid_sid = getattr(session, 'session_id', None)
                    registry.save(
                        tuple(full_seq), snap,
                        position=session.position,
                        middle_session_id=mid_sid,
                    )
                except Exception:
                    pass
                if prompt_snap is not None and supports_prompt_snapshot:
                    try:
                        mid_sid = getattr(session, 'session_id', None)
                        registry.save(
                            tuple(prompt_ids), prompt_snap,
                            position=len(prompt_ids),
                            middle_session_id=mid_sid,
                        )
                    except Exception:
                        pass
            if entry is not None:
                registry.release(entry)
            session.close()

    @staticmethod
    def _cancelled_reason(cancel_event, finish_reason: str) -> bool:
        if finish_reason == "length":
            return True
        if cancel_event is not None and cancel_event.is_set():
            return True
        return False

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
