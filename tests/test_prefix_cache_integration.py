from __future__ import annotations

import unittest

import numpy as np

from controller.cache.registry import PrefixCacheRegistry
from controller.cache.snapshot import CacheSnapshot
from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session


_PREFILL_TOKENS: int = 0


class _CountingSession(Qwen35Session):
    def __init__(self):
        self.position = 0
        self.closed = False
        self.decode_step = 0

    def prefill(self, input_ids, position: int = 0):
        global _PREFILL_TOKENS
        count = len(input_ids)
        _PREFILL_TOKENS += count
        self.position = position + count
        self.decode_step = 0
        arr = np.full((1, 1, 100), -1000.0, dtype=np.float16)
        arr[0, 0, 1] = 10.0
        return arr

    def decode_next(self, token_id):
        self.position += 1
        self.decode_step += 1
        arr = np.full((1, 1, 100), -1000.0, dtype=np.float16)
        if self.decode_step >= 2:
            arr[0, 0, 99] = 10.0  # EOS on 2nd decode
        else:
            arr[0, 0, 2] = 10.0  # non-EOS on 1st decode
        return arr

    def close(self):
        self.closed = True

    def snapshot(self):
        return CacheSnapshot()

    def restore(self, snap, position):
        self.position = position
        self.decode_step = 0


class _CountingModel(Qwen35Model):
    def __init__(self, registry=None):
        super().__init__(ModelInfo(model_name="test-model", max_len=256, vocab_size=100, backend_kind="qwen35_kvcache_om"))
        self.loaded = False
        self.cache_registry = registry

    def load(self):
        self.loaded = True

    def close(self):
        pass

    def is_loaded(self) -> bool:
        return self.loaded

    def create_session(self, cache_entry=None):
        return _CountingSession()


class CountingTokenizer:
    eos_token_id = 99

    def format_messages(self, messages, enable_thinking: bool = False):
        del messages, enable_thinking
        return "prompt"

    def encode_prompt(self, text: str):
        if text == "</think>":
            return [77]
        return [42]

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        return "X"


class PrefixCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        global _PREFILL_TOKENS
        _PREFILL_TOKENS = 0

    def test_cache_hit_reduces_prefill(self):
        global _PREFILL_TOKENS
        from controller.generation.runner import TokenGenerationRunner
        from controller.generation.config import SamplingParams

        reg = PrefixCacheRegistry(max_entries=4, ttl_sec=300, min_prefix_len=2, tag="test")
        model = _CountingModel(registry=reg)
        tokenizer = CountingTokenizer()
        runner = TokenGenerationRunner(model=model, tokenizer=tokenizer)
        params = SamplingParams(
            max_new_tokens=2, temperature=0.0, top_k=40, top_p=1.0,
            presence_penalty=0.0, repetition_penalty=1.0, stop=[], enable_thinking=False,
        )

        prompt_ids = [100, 200, 300, 400]
        round1_tokens = _PREFILL_TOKENS
        info = {}
        list(runner.generate(prompt_ids, params, cache_info=info))
        round1_count = _PREFILL_TOKENS - round1_tokens
        self.assertEqual(info.get("cache_status"), "miss")
        self.assertEqual(round1_count, 4)

        # Round 2: full sequence = round1_prompt + round1_generated + new tokens
        # Round 1 generates [1, 2] then EOS, so saved = [100,200,300,400,1,2]
        prompt_ids_2 = [100, 200, 300, 400, 1, 2, 500, 600]
        before2 = _PREFILL_TOKENS
        info2 = {}
        list(runner.generate(prompt_ids_2, params, cache_info=info2))
        round2_count = _PREFILL_TOKENS - before2
        self.assertEqual(info2.get("cache_status"), "hit")
        self.assertGreater(info2.get("cache_len", 0), 0)
        self.assertEqual(round2_count, 2, f"Expected 2 delta tokens, got {round2_count}")

        reg.stop()

    def test_cache_disabled_full_prefill(self):
        from controller.generation.runner import TokenGenerationRunner
        from controller.generation.config import SamplingParams

        model = _CountingModel(registry=None)
        tokenizer = CountingTokenizer()
        runner = TokenGenerationRunner(model=model, tokenizer=tokenizer)
        params = SamplingParams(
            max_new_tokens=2, temperature=0.0, top_k=40, top_p=1.0,
            presence_penalty=0.0, repetition_penalty=1.0, stop=[], enable_thinking=False,
        )

        prompt_ids = [100, 200, 300]
        info = {}
        list(runner.generate(prompt_ids, params, cache_info=info))
        self.assertEqual(info.get("cache_status"), "disabled")

    def test_divergent_prefix_triggers_rebuild(self):
        global _PREFILL_TOKENS
        from controller.generation.runner import TokenGenerationRunner
        from controller.generation.config import SamplingParams

        reg = PrefixCacheRegistry(max_entries=4, ttl_sec=300, min_prefix_len=2, tag="test")
        model = _CountingModel(registry=reg)
        tokenizer = CountingTokenizer()
        runner = TokenGenerationRunner(model=model, tokenizer=tokenizer)
        params = SamplingParams(
            max_new_tokens=2, temperature=0.0, top_k=40, top_p=1.0,
            presence_penalty=0.0, repetition_penalty=1.0, stop=[], enable_thinking=False,
        )

        prompt_a = [100, 200, 300, 400]
        list(runner.generate(prompt_a, params))
        _PREFILL_TOKENS = 0

        prompt_b = [100, 200, 999, 888]
        info = {}
        list(runner.generate(prompt_b, params, cache_info=info))
        self.assertEqual(info.get("cache_status"), "miss")

        reg.stop()
