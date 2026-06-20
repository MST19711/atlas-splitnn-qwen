from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    max_len: int
    vocab_size: int
    backend_kind: str


class Qwen35Session(ABC):
    @abstractmethod
    def prefill(self, input_ids: list[int]) -> np.ndarray: ...

    @abstractmethod
    def decode_next(self, token_id: int) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...


class Qwen35Model(ABC):
    def __init__(self, info: ModelInfo):
        self._info = info

    @property
    def info(self) -> ModelInfo:
        return self._info

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def create_session(self) -> Qwen35Session: ...
