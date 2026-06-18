from __future__ import annotations

import threading

import numpy as np
import onnxruntime as ort

from controller.engine.base import SplitEngine
from scripts.qwen35_model_spec import ModelSpec, SplitConfig


def _zero_cache_np(model_spec: ModelSpec, nl_dn: int, nl_ga: int, max_len: int) -> dict[str, np.ndarray]:
    conv_ks = model_spec.linear_conv_kernel_dim
    feed = {}
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = np.zeros((1, model_spec.linear_num_value_heads,
                                          model_spec.linear_key_head_dim,
                                          model_spec.linear_value_head_dim), dtype=np.float16)
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = np.zeros((1, model_spec.conv_dim, conv_ks - 1), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = np.zeros((1, model_spec.num_key_value_heads, max_len,
                                          model_spec.head_dim), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"v_past_{i}"] = np.zeros((1, model_spec.num_key_value_heads, max_len,
                                          model_spec.head_dim), dtype=np.float16)
    return feed


def _update_feed(feed: dict[str, np.ndarray], outputs: list[np.ndarray],
                 nl_dn: int, nl_ga: int) -> np.ndarray:
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
    def __init__(self, model_id: str, max_len: int,
                 model_spec: ModelSpec, split_config: SplitConfig,
                 prefix_onnx: str, suffix_onnx: str):
        super().__init__(model_id=model_id, max_len=max_len, model_spec=model_spec)
        self.split_config = split_config
        self.prefix_onnx = prefix_onnx
        self.suffix_onnx = suffix_onnx
        self.prefix_nl_dn, self.prefix_nl_ga = model_spec.compute_segment(*split_config.prefix_range)
        self.suffix_nl_dn, self.suffix_nl_ga = model_spec.compute_segment(*split_config.suffix_range)
        self.prefix_sess: ort.InferenceSession | None = None
        self.suffix_sess: ort.InferenceSession | None = None
        self.prefix_feed: dict[str, np.ndarray] | None = None
        self.suffix_feed: dict[str, np.ndarray] | None = None
        self._prefix_has_pos: bool = True
        self._suffix_has_pos: bool = True
        self._load_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self.prefix_sess = ort.InferenceSession(self.prefix_onnx, providers=["CPUExecutionProvider"])
            self.suffix_sess = ort.InferenceSession(self.suffix_onnx, providers=["CPUExecutionProvider"])
            p_inputs = {i.name for i in self.prefix_sess.get_inputs()}
            s_inputs = {i.name for i in self.suffix_sess.get_inputs()}
            self._prefix_has_pos = "position" in p_inputs
            self._suffix_has_pos = "position" in s_inputs
            self._loaded = True

    def close(self) -> None:
        self.prefix_sess = None
        self.suffix_sess = None
        self.prefix_feed = None
        self.suffix_feed = None
        self._loaded = False

    def start_session(self) -> None:
        self.load()
        self.prefix_feed = _zero_cache_np(self.model_spec, self.prefix_nl_dn, self.prefix_nl_ga,
                                           self.max_len)
        self.suffix_feed = _zero_cache_np(self.model_spec, self.suffix_nl_dn, self.suffix_nl_ga,
                                           self.max_len)

    def end_session(self) -> None:
        self.prefix_feed = None
        self.suffix_feed = None

    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        assert self.prefix_sess is not None and self.prefix_feed is not None
        self.prefix_feed["input_ids"] = np.array([[token_id]], dtype=np.int64)
        if self._prefix_has_pos:
            self.prefix_feed["position"] = np.array([position], dtype=np.int64)
        outputs = self.prefix_sess.run(None, self.prefix_feed)
        hidden = _update_feed(self.prefix_feed, outputs, self.prefix_nl_dn, self.prefix_nl_ga)
        return hidden.reshape(1, 1, self.model_spec.hidden_size).astype(np.float16, copy=False)

    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        assert self.suffix_sess is not None and self.suffix_feed is not None
        self.suffix_feed["hidden_states"] = (hidden_state.astype(np.float16, copy=False)
                                             .reshape(1, 1, self.model_spec.hidden_size))
        if self._suffix_has_pos:
            self.suffix_feed["position"] = np.array([position], dtype=np.int64)
        outputs = self.suffix_sess.run(None, self.suffix_feed)
        logits = _update_feed(self.suffix_feed, outputs, self.suffix_nl_dn, self.suffix_nl_ga)
        return logits.astype(np.float16, copy=False)
