from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from controller.cache.snapshot import CacheSnapshot


@dataclass(frozen=True)
class ModelInfo:
    model_name: str
    max_len: int
    vocab_size: int
    backend_kind: str


class Qwen35Session(ABC):
    @abstractmethod
    def prefill(self, input_ids: list[int], position: int = 0) -> np.ndarray: ...

    @abstractmethod
    def decode_next(self, token_id: int) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    def snapshot(self) -> "CacheSnapshot":
        raise NotImplementedError

    def restore(self, snap: "CacheSnapshot", position: int) -> None:
        raise NotImplementedError


class Qwen35Model(ABC):
    def __init__(self, info: ModelInfo):
        self._info = info
        self.cache_registry = None

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
    def create_session(self, cache_entry: "CacheEntry | None" = None) -> Qwen35Session: ...

    def supports_prefix_cache(self) -> bool:
        return self.cache_registry is not None
