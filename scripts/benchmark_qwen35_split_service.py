#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
import uuid
import zlib

import numpy as np

HIDDEN_SIZE = 1024
HIDDEN_SHAPE = "1,1,1024"
HIDDEN_BYTES = HIDDEN_SIZE * 2
MODEL_NAME = "Qwen3.5-0.8B-split-4-16-4"
PROTOCOL_VERSION = 1


def _open(server_url: str, session_id: str, max_len: int) -> None:
    payload = {
        "session_id": session_id,
        "model": MODEL_NAME,
        "max_len": max_len,
        "hidden_size": HIDDEN_SIZE,
        "dtype": "fp16",
        "protocol_version": PROTOCOL_VERSION,
    }
    req = urllib.request.Request(
        f"{server_url}/v1/session/open",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"open failed: {body}")


def _close(server_url: str, session_id: str) -> None:
    payload = {"session_id": session_id}
    req = urllib.request.Request(
        f"{server_url}/v1/session/close",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30):
        return


def _step(server_url: str, session_id: str, position: int, body: bytes, checksum: bool) -> tuple[float, float]:
    headers = {
        "Content-Type": "application/octet-stream",
        "X-Session-Id": session_id,
        "X-Protocol-Version": str(PROTOCOL_VERSION),
        "X-Position": str(position),
        "X-Hidden-Shape": HIDDEN_SHAPE,
        "X-DType": "fp16",
        "X-Byte-Length": str(HIDDEN_BYTES),
    }
    if checksum:
        headers["X-Checksum"] = f"{zlib.crc32(body) & 0xffffffff:08x}"
    req = urllib.request.Request(
        f"{server_url}/v1/session/step",
        data=body,
        headers=headers,
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()
        if len(payload) != HIDDEN_BYTES:
            raise RuntimeError(f"unexpected body length: {len(payload)}")
        server_latency_ms = float(resp.headers.get("X-Server-Latency-Ms", "nan"))
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    return rtt_ms, server_latency_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:18080")
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hidden = rng.standard_normal((1, 1, HIDDEN_SIZE), dtype=np.float32).astype(np.float16)
    body = hidden.tobytes()

    session_id = f"bench-{uuid.uuid4().hex}"
    _open(args.server_url, session_id, args.max_len)

    rtts: list[float] = []
    server_latencies: list[float] = []
    try:
        total = args.warmup_steps + args.bench_steps
        for pos in range(total):
            rtt_ms, server_latency_ms = _step(args.server_url, session_id, pos, body, args.checksum)
            if pos >= args.warmup_steps:
                rtts.append(rtt_ms)
                server_latencies.append(server_latency_ms)
    finally:
        _close(args.server_url, session_id)

    total_s = sum(rtts) / 1000.0
    server_total_s = sum(server_latencies) / 1000.0
    print(f"max_len={args.max_len}")
    print(f"bench_steps={args.bench_steps}")
    print(f"avg_rtt_ms={statistics.mean(rtts):.3f}")
    print(f"p50_rtt_ms={statistics.median(rtts):.3f}")
    print(f"p95_rtt_ms={np.percentile(np.array(rtts), 95):.3f}")
    print(f"avg_server_ms={statistics.mean(server_latencies):.3f}")
    print(f"p50_server_ms={statistics.median(server_latencies):.3f}")
    print(f"p95_server_ms={np.percentile(np.array(server_latencies), 95):.3f}")
    print(f"tok_per_s_rtt={args.bench_steps / total_s:.3f}")
    print(f"tok_per_s_server={args.bench_steps / server_total_s:.3f}")


if __name__ == "__main__":
    main()
