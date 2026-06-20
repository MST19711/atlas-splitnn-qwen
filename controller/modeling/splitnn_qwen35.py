from __future__ import annotations

import uuid

import numpy as np

from controller.engine.base import SplitEngine
from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session
from controller.remote_middle import RemoteMiddleClient


class SplitNNQwen35Session(Qwen35Session):
    def __init__(self, engine: SplitEngine, remote_middle: RemoteMiddleClient):
        self.engine = engine
        self.remote_middle = remote_middle
        self.session_id = uuid.uuid4().hex
        self.position = 0
        self._closed = False
        self.engine.start_session()
        self.remote_middle.open(self.session_id)

    def prefill(self, input_ids: list[int]) -> np.ndarray:
        if not input_ids:
            raise ValueError("empty input_ids")
        logits = None
        for token_id in input_ids:
            logits = self.decode_next(int(token_id))
        assert logits is not None
        return logits

    def decode_next(self, token_id: int) -> np.ndarray:
        if self._closed:
            raise RuntimeError("session already closed")
        hidden_prefix = self.engine.run_prefix(int(token_id), self.position)
        hidden_middle, _ = self.remote_middle.step(self.session_id, hidden_prefix, self.position)
        logits = self.engine.run_suffix(hidden_middle, self.position)
        self.position += 1
        return np.asarray(logits, dtype=np.float16).reshape(1, 1, -1)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.remote_middle.close(self.session_id)
        finally:
            self.engine.end_session()
            self._closed = True


class SplitNNQwen35Model(Qwen35Model):
    def __init__(
        self,
        model_name: str,
        max_len: int,
        vocab_size: int,
        backend_kind: str,
        engine: SplitEngine,
        remote_middle: RemoteMiddleClient,
    ):
        super().__init__(
            ModelInfo(
                model_name=model_name,
                max_len=max_len,
                vocab_size=vocab_size,
                backend_kind=backend_kind,
            )
        )
        self.engine = engine
        self.remote_middle = remote_middle

    def load(self) -> None:
        self.engine.load()

    def close(self) -> None:
        self.engine.close()

    def is_loaded(self) -> bool:
        return self.engine.is_loaded()

    def create_session(self) -> Qwen35Session:
        return SplitNNQwen35Session(self.engine, self.remote_middle)
