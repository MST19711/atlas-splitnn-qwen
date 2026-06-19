from __future__ import annotations

import unittest

import numpy as np

from controller.modeling.kvcache_qwen35 import Qwen35KvCacheSession
from controller.modeling.splitnn_qwen35 import SplitNNQwen35Session


class FakeEngine:
    def __init__(self):
        self.calls = []

    def start_session(self):
        self.calls.append("start_session")

    def end_session(self):
        self.calls.append("end_session")

    def run_prefix(self, token_id, position):
        self.calls.append(("run_prefix", token_id, position))
        return np.array([[[token_id]]], dtype=np.float16)

    def run_suffix(self, hidden_state, position):
        self.calls.append(("run_suffix", float(hidden_state[0, 0, 0]), position))
        return np.array([[[position]]], dtype=np.float16)


class FakeRemote:
    def __init__(self):
        self.calls = []

    def open(self, session_id):
        self.calls.append(("open", session_id))

    def close(self, session_id):
        self.calls.append(("close", session_id))

    def step(self, session_id, hidden_state, position):
        self.calls.append(("step", session_id, float(hidden_state[0, 0, 0]), position))
        return hidden_state + 10, 1.0


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append("reset")

    def execute(self, token_id, position):
        self.calls.append(("execute", token_id, position))
        return np.array([token_id + position], dtype=np.float16)


class ModelingTests(unittest.TestCase):
    def test_splitnn_session_call_order_and_close(self):
        engine = FakeEngine()
        remote = FakeRemote()
        session = SplitNNQwen35Session(engine, remote)
        logits = session.prefill([11, 12])
        self.assertEqual(logits.shape, (1, 1, 1))
        self.assertEqual(engine.calls[0], "start_session")
        self.assertEqual(engine.calls[1], ("run_prefix", 11, 0))
        self.assertEqual(remote.calls[1][0], "step")
        session.close()
        self.assertEqual(engine.calls[-1], "end_session")
        self.assertEqual(remote.calls[-1][0], "close")

    def test_kvcache_session_position_continues_after_prefill(self):
        runtime = FakeRuntime()
        session = Qwen35KvCacheSession(runtime)
        first = session.prefill([5, 6])
        second = session.decode_next(7)
        self.assertEqual(first.shape, (1, 1, 1))
        self.assertEqual(second.shape, (1, 1, 1))
        self.assertEqual(runtime.calls, ["reset", ("execute", 5, 0), ("execute", 6, 1), ("execute", 7, 2)])
