from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from controller.modeling.kvcache_qwen35 import Qwen35KvCacheSession, _ACLSessionRuntime
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

    def open(self, session_id, prefix_hash=None, resume_token_pos=None):
        self.calls.append(("open", session_id))

    def close(self, session_id, evict=False):
        self.calls.append(("close", session_id))

    def step(self, session_id, hidden_state, position):
        self.calls.append(("step", session_id, float(hidden_state[0, 0, 0]), position))
        return hidden_state + 10, 1.0


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def prepare_fresh(self):
        self.calls.append("prepare_fresh")

    def execute(self, token_id, position):
        self.calls.append(("execute", token_id, position))
        return np.array([token_id + position], dtype=np.float16)


class FakeAclRt:
    def __init__(self):
        self.calls = []

    def set_context(self, context):
        self.calls.append(("set_context", context))
        return 0

    def memset(self, dev_ptr, size, value, count):
        self.calls.append(("memset", dev_ptr, size, value, count))
        return 0


class FakeAcl:
    def __init__(self):
        self.rt = FakeAclRt()


class ModelingTests(unittest.TestCase):
    def test_acl_runtime_prepare_fresh_zeros_both_cache_sets(self):
        runtime = _ACLSessionRuntime.__new__(_ACLSessionRuntime)
        runtime.acl = FakeAcl()
        runtime._context = object()
        runtime._lock = threading.Lock()
        runtime._zero_cache_host = {}
        runtime.layout = SimpleNamespace(s_bytes=2, c_bytes=3, kv_bytes=5)
        runtime._s_a = [11, 12]
        runtime._c_a = [21]
        runtime._k_a = [31]
        runtime._v_a = [41]
        runtime._s_b = [13]
        runtime._c_b = [22]
        runtime._k_b = [32]
        runtime._v_b = [42]
        runtime._step = 99

        runtime.prepare_fresh()

        memset_calls = [call for call in runtime.acl.rt.calls if call[0] == "memset"]
        self.assertEqual(
            memset_calls,
            [
                ("memset", 11, 2, 0, 2),
                ("memset", 12, 2, 0, 2),
                ("memset", 21, 3, 0, 3),
                ("memset", 31, 5, 0, 5),
                ("memset", 41, 5, 0, 5),
                ("memset", 13, 2, 0, 2),
                ("memset", 22, 3, 0, 3),
                ("memset", 32, 5, 0, 5),
                ("memset", 42, 5, 0, 5),
            ],
        )
        self.assertEqual(runtime._step, 0)

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
        self.assertEqual(runtime.calls, ["prepare_fresh", ("execute", 5, 0), ("execute", 6, 1), ("execute", 7, 2)])

    def test_kvcache_session_prefill_honors_position_argument(self):
        runtime = FakeRuntime()
        session = Qwen35KvCacheSession(runtime)
        session.prefill([8, 9], position=3)
        session.decode_next(10)
        self.assertEqual(
            runtime.calls,
            ["prepare_fresh", ("execute", 8, 3), ("execute", 9, 4), ("execute", 10, 5)],
        )
