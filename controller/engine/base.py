from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from scripts.qwen35_model_spec import ModelSpec


class SplitEngine(ABC):
    def __init__(self, model_id: str, max_len: int, model_spec: ModelSpec | None = None):
        self.model_id = model_id
        self.max_len = max_len
        self.model_spec = model_spec

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def start_session(self) -> None: ...

    @abstractmethod
    def end_session(self) -> None: ...

    @abstractmethod
    def run_prefix(self, token_id: int, position: int) -> np.ndarray: ...

    @abstractmethod
    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray: ...


@dataclass
class EngineInfo:
    engine_type: str
    model_id: str
    max_len: int
