from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float
    presence_penalty: float
    repetition_penalty: float
    stop: list[str]
    enable_thinking: bool
