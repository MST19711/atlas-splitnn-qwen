#!/usr/bin/env python3
"""Local SplitNN simulation client using ONNX Runtime for prefix and suffix."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
import zlib

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from qwen35_split_common import CONV_D, CONV_KS, HDIM, HIDDEN_SIZE, KV_H, K_DIM, K_H, V_DIM

MODEL_NAME = "Qwen3.5-0.8B-split-4-16-4"
PROTOCOL_VERSION = 1
HIDDEN_BYTES = HIDDEN_SIZE * 2
PREFIX_NL_DN = 3
PREFIX_NL_GA = 1
SUFFIX_NL_DN = 3
SUFFIX_NL_GA = 1


def zero_cache_np(nl_dn: int, nl_ga: int, max_len: int) -> dict[str, np.ndarray]:
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


def update_feed(feed: dict[str, np.ndarray], outputs: list[np.ndarray], nl_dn: int, nl_ga: int) -> np.ndarray:
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


class RemoteMiddleClient:
    def __init__(self, server_url: str, max_len: int, checksum: bool):
        self.server_url = server_url.rstrip("/")
        self.max_len = max_len
        self.checksum = checksum
        self.session_id = uuid.uuid4().hex

    def _json_request(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.server_url}/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def open(self) -> dict:
        return self._json_request(
            "/v1/session/open",
            {
                "session_id": self.session_id,
                "model": MODEL_NAME,
                "max_len": self.max_len,
                "hidden_size": HIDDEN_SIZE,
                "dtype": "fp16",
                "protocol_version": PROTOCOL_VERSION,
            },
        )

    def close(self) -> dict:
        return self._json_request("/v1/session/close", {"session_id": self.session_id})

    def step(self, hidden: np.ndarray, position: int) -> tuple[np.ndarray, float]:
        body = hidden.astype(np.float16, copy=False).tobytes()
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Session-Id": self.session_id,
            "X-Protocol-Version": str(PROTOCOL_VERSION),
            "X-Position": str(position),
            "X-Hidden-Shape": "1,1,1024",
            "X-DType": "fp16",
            "X-Byte-Length": str(HIDDEN_BYTES),
        }
        if self.checksum:
            headers["X-Checksum"] = f"{zlib.crc32(body) & 0xffffffff:08x}"
        req = urllib.request.Request(
            f"{self.server_url}/v1/session/step",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = resp.read()
                if len(payload) != HIDDEN_BYTES:
                    raise RuntimeError(f"bad response length {len(payload)}")
                if self.checksum:
                    want = resp.headers.get("X-Checksum")
                    got = f"{zlib.crc32(payload) & 0xffffffff:08x}"
                    if want and want.lower() != got:
                        raise RuntimeError("response checksum mismatch")
                latency_ms = float(resp.headers.get("X-Server-Latency-Ms", "0"))
                return np.frombuffer(payload, dtype=np.float16).reshape(1, 1, HIDDEN_SIZE).copy(), latency_ms
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body_text}") from exc


def sample_argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits[0, 0, :]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--prefix-onnx", default="om_out/qwen3.5_split_prefix_max256.onnx")
    parser.add_argument("--suffix-onnx", default="om_out/qwen3.5_split_suffix_max256.onnx")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--prompt-ids", default="100,200,300")
    parser.add_argument("--tokenizer-dir", default="model/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    prefix = ort.InferenceSession(args.prefix_onnx, providers=["CPUExecutionProvider"])
    suffix = ort.InferenceSession(args.suffix_onnx, providers=["CPUExecutionProvider"])
    prefix_feed = zero_cache_np(PREFIX_NL_DN, PREFIX_NL_GA, args.max_len)
    suffix_feed = zero_cache_np(SUFFIX_NL_DN, SUFFIX_NL_GA, args.max_len)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    if args.prompt:
        msgs = [{"role": "user", "content": args.prompt}]
        formatted = tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tok.encode(formatted, add_special_tokens=False)
    else:
        prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        raise ValueError("prompt_ids is empty")
    prompt_ids = prompt_ids[-args.max_len :]
    print(f"[Prompt tokens: {len(prompt_ids)}]")

    remote = RemoteMiddleClient(args.server_url, args.max_len, args.checksum)
    print(json.dumps(remote.health(), ensure_ascii=False))
    print(json.dumps(remote.open(), ensure_ascii=False))

    total_prefix_ms = 0.0
    total_middle_ms = 0.0
    total_suffix_ms = 0.0
    last_logits = None
    generated_ids = []

    try:
        steps = prompt_ids + [None] * args.decode_steps
        current_id = None
        for pos, tid in enumerate(steps):
            if tid is None:
                tid = current_id
            prefix_feed["input_ids"] = np.array([[tid]], dtype=np.int64)
            prefix_feed["position"] = np.array([pos], dtype=np.int64)

            t0 = time.perf_counter()
            prefix_out = prefix.run(None, prefix_feed)
            hidden = update_feed(prefix_feed, prefix_out, PREFIX_NL_DN, PREFIX_NL_GA)
            total_prefix_ms += (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            hidden_mid, server_ms = remote.step(hidden, pos)
            total_middle_ms += (time.perf_counter() - t1) * 1000.0

            suffix_feed["hidden_states"] = hidden_mid.astype(np.float16, copy=False)
            suffix_feed["position"] = np.array([pos], dtype=np.int64)
            t2 = time.perf_counter()
            suffix_out = suffix.run(None, suffix_feed)
            last_logits = update_feed(suffix_feed, suffix_out, SUFFIX_NL_DN, SUFFIX_NL_GA)
            total_suffix_ms += (time.perf_counter() - t2) * 1000.0

            current_id = sample_argmax(last_logits)
            if tid not in prompt_ids[len(generated_ids) : len(generated_ids) + 1]:
                pass
            if pos >= len(prompt_ids):
                generated_ids.append(current_id)
                piece = tok.decode([current_id], skip_special_tokens=True)
                if piece:
                    print(f"text_piece={piece!r}")
            print(
                f"step={pos} token_in={tid} token_out={current_id} "
                f"server_ms={server_ms:.3f} logits_range=[{last_logits.min():.4f},{last_logits.max():.4f}]"
            )
    finally:
        print(json.dumps(remote.close(), ensure_ascii=False))

    if generated_ids:
        print("generated_text:")
        print(tok.decode(generated_ids, skip_special_tokens=True))

    print(
        f"PASS prefix_ms={total_prefix_ms:.1f} middle_rtt_ms={total_middle_ms:.1f} suffix_ms={total_suffix_ms:.1f}"
    )


if __name__ == "__main__":
    main()
