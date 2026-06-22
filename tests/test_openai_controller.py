from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import controller.openai_controller as openai_controller
from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session
from controller.schemas import ChatCompletionRequest


class FakeSession(Qwen35Session):
    def __init__(self):
        self.position = 0
        self.closed = False

    def prefill(self, input_ids, position: int = 0):
        self.position = position + len(input_ids)
        return _logits(1)

    def decode_next(self, token_id):
        self.position += 1
        return _logits(99)

    def close(self):
        self.closed = True


class FakeModel(Qwen35Model):
    def __init__(self, backend_kind="splitnn_om"):
        super().__init__(ModelInfo(model_name="fake-model", max_len=16, vocab_size=100, backend_kind=backend_kind))
        self.loaded = False
        self.closed = False
        self.remote_middle = SimpleNamespace(health=lambda: {"ok": True})

    def load(self):
        self.loaded = True

    def close(self):
        self.closed = True

    def is_loaded(self) -> bool:
        return self.loaded

    def create_session(self, cache_entry=None):
        return FakeSession()


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, *_args, **_kwargs):
        pass

    def format_messages(self, messages, enable_thinking: bool):
        del messages, enable_thinking
        return "prompt"

    def encode_prompt(self, text: str):
        if text == "</think>":
            return [77]
        return [42]

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        mapping = {1: "hello"}
        return "".join(mapping.get(t, "") for t in token_ids)


def _logits(token_id: int, vocab_size: int = 100):
    import numpy as np

    arr = np.full((1, 1, vocab_size), -1000.0, dtype=np.float16)
    arr[0, 0, token_id] = 10.0
    return arr


def _make_args():
    return SimpleNamespace(
        backend="splitnn_om",
        model_name="fake-model",
        remote_model_name="",
        tokenizer_dir="unused",
        max_len=16,
        server_url="http://unused",
        connect_timeout=1.0,
        read_timeout=1.0,
        checksum=False,
        model_path="model/Qwen3.5-0.8B",
        split=(4, 20),
        prefix_onnx=None,
        suffix_onnx=None,
        prefix_om=None,
        suffix_om=None,
        bound_asset_dir="",
        model_om="",
    )


class OpenAIControllerTests(unittest.TestCase):
    @staticmethod
    def _route(app, path: str):
        for route in app.router.routes:
            if getattr(route, "path", None) == path:
                return route.endpoint
        raise AssertionError(f"route not found: {path}")

    def test_non_stream_chat_completion(self):
        with patch.object(openai_controller, "Qwen35TokenizerAdapter", FakeTokenizer), patch.object(
            openai_controller, "create_model", lambda _config: FakeModel()
        ):
            app = openai_controller.build_app(_make_args())
            route = self._route(app, "/v1/chat/completions")
            model = FakeModel()
            tokenizer = FakeTokenizer()
            openai_controller.MODEL = model
            openai_controller.TOKENIZER = tokenizer
            app.state.runner = openai_controller.OpenAIChatAdapter(model, tokenizer)
            app.state.model_loaded = False
            app.state.model_load_error = None

            class DummyRequest:
                async def is_disconnected(self):
                    return False

            async def run_test():
                response = await route(
                    DummyRequest(),
                    ChatCompletionRequest(
                        model="fake-model",
                        messages=[{"role": "user", "content": "hi"}],
                        stream=False,
                    ),
                )
                return json.loads(response.body.decode("utf-8"))

            data = asyncio.run(run_test())
        self.assertEqual(data["choices"][0]["message"]["content"], "hello")

    def test_stream_chat_completion(self):
        with patch.object(openai_controller, "Qwen35TokenizerAdapter", FakeTokenizer), patch.object(
            openai_controller, "create_model", lambda _config: FakeModel()
        ):
            app = openai_controller.build_app(_make_args())
            route = self._route(app, "/v1/chat/completions")
            model = FakeModel()
            tokenizer = FakeTokenizer()
            openai_controller.MODEL = model
            openai_controller.TOKENIZER = tokenizer
            app.state.runner = openai_controller.OpenAIChatAdapter(model, tokenizer)
            app.state.model_loaded = False
            app.state.model_load_error = None

            class DummyRequest:
                async def is_disconnected(self):
                    return False

            async def run_test():
                response = await route(
                    DummyRequest(),
                    ChatCompletionRequest(
                        model="fake-model",
                        messages=[{"role": "user", "content": "hi"}],
                        stream=True,
                    ),
                )
                chunks = []
                async for chunk in response.body_iterator:
                    chunks.append(chunk if isinstance(chunk, str) else chunk.decode("utf-8"))
                return "".join(chunks)

            body = asyncio.run(run_test())
        self.assertIn("data: [DONE]", body)
        self.assertIn("\"content\":\"hello\"", body)

    def test_models_route(self):
        with patch.object(openai_controller, "Qwen35TokenizerAdapter", FakeTokenizer), patch.object(
            openai_controller, "create_model", lambda _config: FakeModel()
        ):
            app = openai_controller.build_app(_make_args())
            route = self._route(app, "/v1/models")

            async def run_test():
                response = await route()
                return response.model_dump()

            data = asyncio.run(run_test())
        self.assertEqual(data["data"][0]["id"], "fake-model")
