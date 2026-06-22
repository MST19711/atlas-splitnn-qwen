from __future__ import annotations

import time
import unittest

import numpy as np

from controller.cache.registry import PrefixCacheRegistry
from controller.cache.snapshot import CacheSnapshot


def _make_snapshot(nl_dn: int = 2, nl_ga: int = 1) -> CacheSnapshot:
    return CacheSnapshot(
        s_states=[np.zeros((1, 16, 128, 128), dtype=np.float16) for _ in range(nl_dn)],
        c_states=[np.zeros((1, 6144, 3), dtype=np.float16) for _ in range(nl_dn)],
        k_states=[np.zeros((1, 2, 32, 256), dtype=np.float16) for _ in range(nl_ga)],
        v_states=[np.zeros((1, 2, 32, 256), dtype=np.float16) for _ in range(nl_ga)],
    )


class RegistryLookupTests(unittest.TestCase):
    def setUp(self):
        self.reg = PrefixCacheRegistry(max_entries=4, ttl_sec=300, min_prefix_len=2)

    def tearDown(self):
        self.reg.stop()

    def test_lookup_exact_match(self):
        s = _make_snapshot()
        self.reg.save((1, 2, 3), s, position=3)
        entry, match_len, walked = self.reg.lookup((1, 2, 3))
        self.assertIsNotNone(entry)
        self.assertEqual(match_len, 3)
        self.assertEqual(entry.position, 3)

    def test_lookup_prefix_match(self):
        s = _make_snapshot()
        self.reg.save((1, 2, 3, 4), s, position=4)
        entry, match_len, walked = self.reg.lookup((1, 2, 3, 4, 5))
        self.assertIsNotNone(entry)
        self.assertEqual(match_len, 4)

    def test_lookup_divergent_no_entry_returns_none(self):
        s = _make_snapshot()
        self.reg.save((1, 2, 3, 4), s, position=4)
        entry, match_len, walked = self.reg.lookup((1, 2, 99))
        self.assertIsNone(entry)
        self.assertEqual(match_len, 0)
        self.assertEqual(walked, 2)

    def test_lookup_below_min_prefix_len(self):
        s = _make_snapshot()
        self.reg.save((1,), s, position=1)
        entry, match_len, _walked = self.reg.lookup((1, 2))
        self.assertIsNone(entry)

    def test_lookup_longest_prefix_wins(self):
        s1 = _make_snapshot()
        s2 = _make_snapshot()
        self.reg.save((1, 2, 3), s1, position=3)
        self.reg.save((1, 2, 5, 6), s2, position=4)
        entry, match_len, _walked = self.reg.lookup((1, 2, 3, 7))
        self.assertIsNotNone(entry)
        self.assertEqual(match_len, 3)
        self.assertEqual(entry.position, 3)


class RegistryLRUTests(unittest.TestCase):
    def setUp(self):
        self.reg = PrefixCacheRegistry(max_entries=3, ttl_sec=300, min_prefix_len=1)

    def tearDown(self):
        self.reg.stop()

    def test_evicts_when_full(self):
        for i in range(4):
            s = _make_snapshot()
            s.s_states[0][0, 0, 0, 0] = float(i + 1)  # mark
            self.reg.save((i,), s, position=1)
            time.sleep(0.001)  # ensure distinct timestamps
        self.assertLessEqual(len(self.reg._by_hash), 3)

    def test_eviction_skips_in_use(self):
        s1 = _make_snapshot()
        entry = self.reg.save((1,), s1, position=1)
        self.reg.acquire(entry)
        for i in range(2, 5):
            s = _make_snapshot()
            self.reg.save((i,), s, position=1)
            time.sleep(0.001)
        found, _, _ = self.reg.lookup((1,))
        self.assertIsNotNone(found)
        self.reg.release(entry)


class RegistryForkTests(unittest.TestCase):
    def setUp(self):
        self.reg = PrefixCacheRegistry(max_entries=4, ttl_sec=300, min_prefix_len=2)

    def tearDown(self):
        self.reg.stop()

    def test_acquire_forks_when_in_use(self):
        s = _make_snapshot()
        entry = self.reg.save((1, 2), s, position=2)
        e1 = self.reg.acquire(entry)
        self.assertEqual(e1.ref_count, 1)
        e2 = self.reg.acquire(entry)
        self.assertIsNot(e1, e2)
        self.assertEqual(e1.ref_count, 1)
        self.assertEqual(e2.ref_count, 1)

    def test_acquire_returns_same_when_free(self):
        s = _make_snapshot()
        entry = self.reg.save((1, 2), s, position=2)
        e1 = self.reg.acquire(entry)
        self.reg.release(e1)
        e2 = self.reg.acquire(entry)
        self.assertIs(e1, e2)
        self.assertEqual(e2.ref_count, 1)

    def test_fork_does_not_affect_original(self):
        s = _make_snapshot()
        s.s_states[0][0, 0, 0, 0] = np.float16(42.0)
        entry = self.reg.save((1, 2), s, position=2)
        e1 = self.reg.acquire(entry)
        e2 = self.reg.acquire(entry)
        e1.snapshot.s_states[0][0, 0, 0, 0] = np.float16(99.0)
        self.assertEqual(e2.snapshot.s_states[0][0, 0, 0, 0], np.float16(42.0))


class SnapshotTests(unittest.TestCase):
    def test_byte_size(self):
        s = _make_snapshot(nl_dn=2, nl_ga=1)
        self.assertGreater(s.byte_size(), 0)

    def test_is_empty(self):
        s = CacheSnapshot()
        self.assertTrue(s.is_empty())
        s2 = _make_snapshot()
        self.assertFalse(s2.is_empty())

    def test_copy_deep(self):
        s = _make_snapshot(nl_dn=1, nl_ga=1)
        s.s_states[0][0, 0, 0, 0] = np.float16(7.0)
        c = s.copy()
        c.s_states[0][0, 0, 0, 0] = np.float16(99.0)
        self.assertEqual(s.s_states[0][0, 0, 0, 0], np.float16(7.0))
        self.assertEqual(c.s_states[0][0, 0, 0, 0], np.float16(99.0))
