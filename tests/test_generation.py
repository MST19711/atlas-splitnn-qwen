from __future__ import annotations

import unittest

import numpy as np

from controller.generation.logits_processors import apply_presence_penalty, apply_repetition_penalty
from controller.generation.runner import TokenGenerationRunner, apply_stop
from controller.generation.config import SamplingParams
from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session


class DummyTokenizer:
    eos_token_id = 99

    def __init__(self):
        self.mapping = {1: "A", 2: "B", 3: "C"}

    def encode_prompt(self, text: str):
        if text == "</think>":
            return [77]
        return []

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        return "".join(self.mapping.get(t, f"<{t}>") for t in token_ids)


class ByteFallbackTokenizer(DummyTokenizer):
    def __init__(self):
        super().__init__()
        self.mapping = {126: "\ufffd", 94: "¡", 1: "A", 99: ""}

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        if token_ids == [126]:
            return "\ufffd"
        if token_ids == [126, 94]:
            return "¡"
        return "".join(self.mapping.get(t, f"<{t}>") for t in token_ids)


class DummySession(Qwen35Session):
    def __init__(self, logits_list):
        self.logits_list = list(logits_list)
        self.position = 0
        self.closed = False
        self.calls = []

    def prefill(self, input_ids, position: int = 0):
        self.calls.append(("prefill", list(input_ids)))
        self.position = position + len(input_ids)
        return self.logits_list.pop(0)

    def decode_next(self, token_id: int):
        self.calls.append(("decode_next", token_id))
        self.position += 1
        return self.logits_list.pop(0)

    def close(self):
        self.closed = True

    def snapshot(self): ...

    def restore(self, snap, position):
        self.position = position


class DummyModel(Qwen35Model):
    def __init__(self, session):
        super().__init__(ModelInfo(model_name="dummy", max_len=8, vocab_size=100, backend_kind="dummy"))
        self.session = session

    def load(self):
        return

    def close(self):
        return

    def is_loaded(self) -> bool:
        return True

    def create_session(self, cache_entry=None):
        return self.session


def _logits(preferred_token: int, vocab_size: int = 100):
    arr = np.full((1, 1, vocab_size), -1000.0, dtype=np.float16)
    arr[0, 0, preferred_token] = 10.0
    return arr


class GenerationTests(unittest.TestCase):
    def test_presence_penalty(self):
        logits = np.array([1.0, 2.0, 3.0])
        updated = apply_presence_penalty(logits, {1: 2}, 0.5)
        self.assertEqual(updated.tolist(), [1.0, 1.5, 3.0])

    def test_repetition_penalty(self):
        logits = np.array([1.0, -2.0, 3.0])
        updated = apply_repetition_penalty(logits, {0: 1, 1: 1}, 2.0)
        self.assertEqual(updated[0], 0.5)
        self.assertEqual(updated[1], -4.0)

    def test_apply_stop(self):
        trimmed, hit = apply_stop("hello world", [" world"])
        self.assertEqual(trimmed, "hello")
        self.assertTrue(hit)

    def test_generation_runner_stops_on_eos(self):
        session = DummySession([_logits(1), _logits(99)])
        model = DummyModel(session)
        runner = TokenGenerationRunner(model=model, tokenizer=DummyTokenizer())
        params = SamplingParams(
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            stop=[],
            enable_thinking=False,
        )
        steps = list(runner.generate([42], params))
        self.assertEqual([step.delta_text for step in steps[:-1]], ["A"])
        self.assertEqual(steps[-1].finish_reason, "stop")
        self.assertTrue(session.closed)

    def test_generation_runner_applies_stop_string(self):
        session = DummySession([_logits(1), _logits(2), _logits(99)])
        model = DummyModel(session)
        runner = TokenGenerationRunner(model=model, tokenizer=DummyTokenizer())
        params = SamplingParams(
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            stop=["AB"],
            enable_thinking=False,
        )
        steps = list(runner.generate([42], params))
        self.assertEqual([step.delta_text for step in steps[:-1]], ["A"])
        self.assertEqual(steps[-1].finish_reason, "stop")

    def test_generation_runner_buffers_incomplete_byte_fallback(self):
        session = DummySession([_logits(126, vocab_size=256), _logits(94, vocab_size=256), _logits(99, vocab_size=256)])
        model = DummyModel(session)
        runner = TokenGenerationRunner(model=model, tokenizer=ByteFallbackTokenizer())
        params = SamplingParams(
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            stop=[],
            enable_thinking=False,
        )
        steps = list(runner.generate([42], params))
        self.assertEqual([step.delta_text for step in steps[:-1]], ["¡"])
        self.assertEqual(steps[-1].finish_reason, "stop")
