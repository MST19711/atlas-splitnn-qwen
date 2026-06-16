from __future__ import annotations

import numpy as np
import onnxruntime as ort

from controller.engine.base import SplitEngine
from scripts.qwen35_split_common import CONV_D, CONV_KS, HDIM, HIDDEN_SIZE, KV_H, K_DIM, K_H, V_DIM

PREFIX_NL_DN = 3
PREFIX_NL_GA = 1
SUFFIX_NL_DN = 3
SUFFIX_NL_GA = 1


def _zero_cache_np(nl_dn: int, nl_ga: int, max_len: int) -> dict[str, np.ndarray]:
    feed = {}
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = np.zeros((1, K_H, K_DIM, V_DIM), dtype=np.float16)
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = np.zeros((1, CONV_D, CONV_KS - 1), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = np.zeros((1, KV_H, max_len, HDIM), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"v_past_{i}"] = np.zeros((1, KV_H, max_len, HDIM), dtype=np.float16)
    return feed


def _update_feed(feed: dict[str, np.ndarray], outputs: list[np.ndarray], nl_dn: int, nl_ga: int) -> np.ndarray:
    main = outputs[0]
    idx = 1
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_ga):
        feed[f"v_past_{i}"] = outputs[idx]
        idx += 1
    return main


class OnnxSplitEngine(SplitEngine):
    def __init__(self, model_id: str, max_len: int, prefix_onnx: str, suffix_onnx: str):
        super().__init__(model_id=model_id, max_len=max_len)
        self.prefix_onnx = prefix_onnx
        self.suffix_onnx = suffix_onnx
        self.prefix_sess: ort.InferenceSession | None = None
        self.suffix_sess: ort.InferenceSession | None = None
        self.prefix_feed: dict[str, np.ndarray] | None = None
        self.suffix_feed: dict[str, np.ndarray] | None = None

    def load(self) -> None:
        self.prefix_sess = ort.InferenceSession(self.prefix_onnx, providers=["CPUExecutionProvider"])
        self.suffix_sess = ort.InferenceSession(self.suffix_onnx, providers=["CPUExecutionProvider"])

    def close(self) -> None:
        self.prefix_sess = None
        self.suffix_sess = None
        self.prefix_feed = None
        self.suffix_feed = None

    def start_session(self) -> None:
        self.prefix_feed = _zero_cache_np(PREFIX_NL_DN, PREFIX_NL_GA, self.max_len)
        self.suffix_feed = _zero_cache_np(SUFFIX_NL_DN, SUFFIX_NL_GA, self.max_len)

    def end_session(self) -> None:
        self.prefix_feed = None
        self.suffix_feed = None

    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        assert self.prefix_sess is not None and self.prefix_feed is not None
        self.prefix_feed["input_ids"] = np.array([[token_id]], dtype=np.int64)
        self.prefix_feed["position"] = np.array([position], dtype=np.int64)
        outputs = self.prefix_sess.run(None, self.prefix_feed)
        hidden = _update_feed(self.prefix_feed, outputs, PREFIX_NL_DN, PREFIX_NL_GA)
        return hidden.reshape(1, 1, HIDDEN_SIZE).astype(np.float16, copy=False)

    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        assert self.suffix_sess is not None and self.suffix_feed is not None
        self.suffix_feed["hidden_states"] = hidden_state.astype(np.float16, copy=False).reshape(1, 1, HIDDEN_SIZE)
        self.suffix_feed["position"] = np.array([position], dtype=np.int64)
        outputs = self.suffix_sess.run(None, self.suffix_feed)
        logits = _update_feed(self.suffix_feed, outputs, SUFFIX_NL_DN, SUFFIX_NL_GA)
        return logits.astype(np.float16, copy=False)
