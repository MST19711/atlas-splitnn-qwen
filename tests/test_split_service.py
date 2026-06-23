from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import torch

from server.qwen35_split_service import SessionState, SplitService


def _make_service() -> SplitService:
    service = SplitService.__new__(SplitService)
    service.max_len = 16
    service.hidden_size = 8
    service.hidden_shape = "1,1,8"
    service.hidden_bytes = 16
    service.device = torch.device("cpu")
    service.max_sessions = 8
    service.session_timeout_sec = 300
    service.sessions = {}
    service._alias_index = {}
    service.sessions_lock = threading.Lock()
    service.model_spec = SimpleNamespace(
        linear_conv_kernel_dim=4,
        linear_num_value_heads=2,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
        conv_dim=7,
        num_key_value_heads=2,
        head_dim=4,
    )
    service.mid_nl_dn = 1
    service.mid_nl_ga = 1
    service.model_name = "test-split-model"
    service.split_config = SimpleNamespace(prefix_end=4, suffix_start=20)
    return service


class SplitServiceTests(unittest.TestCase):
    def test_open_session_clones_resume_source_state(self):
        service = _make_service()
        source = SessionState.create(
            "source",
            service.max_len,
            service.device,
            service.model_spec,
            service.mid_nl_dn,
            service.mid_nl_ga,
        )
        source.position_next = 6
        source.s_cache[0].fill_(1)
        source.c_cache[0].fill_(2)
        source.k_cache[0].fill_(3)
        source.v_cache[0].fill_(4)
        source.ref_count = 1
        service.sessions[source.session_id] = source

        payload = {
            "session_id": "forked",
            "model": service.model_name,
            "max_len": service.max_len,
            "hidden_size": service.hidden_size,
            "dtype": "fp16",
            "protocol_version": 2,
            "resume_from_session_id": "source",
        }
        result = service.open_session(payload)

        self.assertTrue(result["session_resumed"])
        self.assertEqual(result["session_id"], "forked")
        forked = service.sessions["forked"]
        self.assertEqual(forked.position_next, 6)
        self.assertIsNot(forked.s_cache[0], source.s_cache[0])
        self.assertEqual(float(forked.s_cache[0][0, 0, 0, 0]), 1.0)
        forked.s_cache[0][0, 0, 0, 0] = 9
        self.assertEqual(float(source.s_cache[0][0, 0, 0, 0]), 1.0)

    def test_close_session_evict_releases_after_unlock(self):
        service = _make_service()
        session = SessionState.create(
            "victim",
            service.max_len,
            service.device,
            service.model_spec,
            service.mid_nl_dn,
            service.mid_nl_ga,
        )
        session.ref_count = 1
        service.sessions[session.session_id] = session

        result = service.close_session({"session_id": "victim", "evict": True})

        self.assertTrue(result["released"])
        self.assertNotIn("victim", service.sessions)


if __name__ == "__main__":
    unittest.main()
