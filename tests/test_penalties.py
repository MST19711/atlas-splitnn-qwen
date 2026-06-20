from __future__ import annotations

import unittest

import numpy as np

from controller.generation.logits_processors import apply_presence_penalty, apply_repetition_penalty


class PresencePenaltyTests(unittest.TestCase):
    def test_no_penalty_when_zero(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_presence_penalty(logits, token_counts={}, penalty=0.0)
        np.testing.assert_array_equal(result, logits)

    def test_penalty_reduces_seen_tokens(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_presence_penalty(logits, token_counts={0: 1}, penalty=2.0)
        self.assertLess(result[0], 1.0)
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], 3.0)

    def test_penalty_with_empty_counts(self):
        logits = np.array([1.0, 2.0], dtype=np.float32)
        result = apply_presence_penalty(logits, token_counts={}, penalty=1.5)
        np.testing.assert_array_equal(result, logits)


class RepetitionPenaltyTests(unittest.TestCase):
    def test_no_penalty_when_one(self):
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = apply_repetition_penalty(logits, token_counts={}, penalty=1.0)
        np.testing.assert_array_equal(result, logits)

    def test_penalty_scales_by_count(self):
        logits = np.array([10.0, 5.0], dtype=np.float32)
        result = apply_repetition_penalty(logits, token_counts={0: 2}, penalty=1.1)
        self.assertLess(result[0], 10.0)
        self.assertEqual(result[1], 5.0)


if __name__ == "__main__":
    unittest.main()
