from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EngineInfo:
    engine_type: str
    model_id: str
    max_len: int


class SplitEngine(ABC):
    def __init__(self, model_id: str, max_len: int):
        self.model_id = model_id
        self.max_len = max_len

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            engine_type=self.__class__.__name__,
            model_id=self.model_id,
            max_len=self.max_len,
        )

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_session(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        raise NotImplementedError
