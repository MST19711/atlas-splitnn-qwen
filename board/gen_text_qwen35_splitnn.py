#!/usr/bin/env python3
"""Qwen3.5 SplitNN inference for Atlas board: prefix OM + remote middle + suffix OM."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
import warnings
import zlib

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
import acl  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

M, H2D, D2H = 0, 1, 2
HIDDEN_SIZE = 1024
HIDDEN_BYTES = HIDDEN_SIZE * 2
PROTOCOL_VERSION = 1
MODEL_NAME = "Qwen3.5-0.8B-split-4-16-4"

NL_DN = 3
NL_GA = 1
K_H = 16
K_DIM = 128
V_DIM = 128
CONV_D = 6144
CONV_KS = 4
KV_H = 2
HDIM = 256
S_BYTES = 1 * K_H * K_DIM * V_DIM * 2
C_BYTES = 1 * CONV_D * (CONV_KS - 1) * 2


def check(ret, msg):
    if ret != 0:
        raise RuntimeError(f"{msg} failed, ret={ret}")


def _cache_bytes(max_len: int) -> int:
    return 1 * KV_H * max_len * HDIM * 2


def _alloc_ptr(allocs, size):
    ptr, ret = acl.rt.malloc(size, M)
    check(ret, "malloc")
    allocs.append(ptr)
    return ptr


def _add_buffer(dataset, ptr, size, tag):
    buf = acl.create_data_buffer(ptr, size)
    _, ret = acl.mdl.add_dataset_buffer(dataset, buf)
    check(ret, tag)
    return buf


class ACLRuntime:
    def __init__(self):
        check(acl.init(), "init")
        check(acl.rt.set_device(0), "set_device")

    def close(self):
        check(acl.rt.reset_device(0), "reset_device")
        check(acl.finalize(), "finalize")


class ACLSplitSegment:
    def __init__(self, model_path: str, max_len: int, input0_size: int, output0_size: int | None, input0_name: str):
        self.max_len = max_len
        self.kv_bytes = _cache_bytes(max_len)
        self.input0_size = input0_size
        self.input0_name = input0_name

        self.mid, ret = acl.mdl.load_from_file(model_path)
        check(ret, "load")
        self.desc = acl.mdl.create_desc()
        check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.num_inputs = acl.mdl.get_num_inputs(self.desc)
        self.num_outputs = acl.mdl.get_num_outputs(self.desc)
        self.output0_size = output0_size
        if self.output0_size is None:
            self.output0_size = acl.mdl.get_output_size_by_index(self.desc, 0)

        self._alloc = []
        self._d0 = _alloc_ptr(self._alloc, input0_size)
        self._dp = _alloc_ptr(self._alloc, 8)
        self._do = _alloc_ptr(self._alloc, self.output0_size)

        def alloc_set():
            return (
                [_alloc_ptr(self._alloc, S_BYTES) for _ in range(NL_DN)],
                [_alloc_ptr(self._alloc, C_BYTES) for _ in range(NL_DN)],
                [_alloc_ptr(self._alloc, self.kv_bytes) for _ in range(NL_GA)],
                [_alloc_ptr(self._alloc, self.kv_bytes) for _ in range(NL_GA)],
            )

        self._sA, self._cA, self._kA, self._vA = alloc_set()
        self._sB, self._cB, self._kB, self._vB = alloc_set()

        self._ds_in_A, self._ds_out_B, self._bufs_A = self._make_ds(
            self._d0, self._dp, self._sA, self._cA, self._kA, self._vA, self._do, self._sB, self._cB, self._kB, self._vB
        )
        self._ds_in_B, self._ds_out_A, self._bufs_B = self._make_ds(
            self._d0, self._dp, self._sB, self._cB, self._kB, self._vB, self._do, self._sA, self._cA, self._kA, self._vA
        )

        self._h0 = np.empty(input0_size, np.uint8)
        self._hp = np.empty(8, np.uint8)
        self._ho = np.empty(self.output0_size, np.uint8)
        self._step = 0

    def _make_ds(self, d0, dp, s_src, c_src, k_src, v_src, dout, s_dst, c_dst, k_dst, v_dst):
        ds_in = acl.mdl.create_dataset()
        ds_out = acl.mdl.create_dataset()
        bufs = []
        bufs.append(_add_buffer(ds_in, d0, self.input0_size, f"{self.input0_name} add_in"))
        bufs.append(_add_buffer(ds_in, dp, 8, "position add_in"))
        for ptr in s_src:
            bufs.append(_add_buffer(ds_in, ptr, S_BYTES, "s add_in"))
        for ptr in c_src:
            bufs.append(_add_buffer(ds_in, ptr, C_BYTES, "c add_in"))
        for ptr in k_src:
            bufs.append(_add_buffer(ds_in, ptr, self.kv_bytes, "k add_in"))
        for ptr in v_src:
            bufs.append(_add_buffer(ds_in, ptr, self.kv_bytes, "v add_in"))

        bufs.append(_add_buffer(ds_out, dout, self.output0_size, "main add_out"))
        for ptr in s_dst:
            bufs.append(_add_buffer(ds_out, ptr, S_BYTES, "s add_out"))
        for ptr in c_dst:
            bufs.append(_add_buffer(ds_out, ptr, C_BYTES, "c add_out"))
        for ptr in k_dst:
            bufs.append(_add_buffer(ds_out, ptr, self.kv_bytes, "k add_out"))
        for ptr in v_dst:
            bufs.append(_add_buffer(ds_out, ptr, self.kv_bytes, "v add_out"))
        return ds_in, ds_out, bufs

    def execute(self, input_bytes: bytes, position: int) -> np.ndarray:
        self._h0[: self.input0_size] = np.frombuffer(input_bytes, dtype=np.uint8, count=self.input0_size)
        self._hp[:8] = np.array([position], np.int64).view(np.uint8)
        acl.rt.memcpy(self._d0, self.input0_size, self._h0.ctypes.data, self.input0_size, H2D)
        acl.rt.memcpy(self._dp, 8, self._hp.ctypes.data, 8, H2D)

        if self._step % 2 == 0:
            ds_in, ds_out = self._ds_in_A, self._ds_out_B
        else:
            ds_in, ds_out = self._ds_in_B, self._ds_out_A
        self._step += 1

        check(acl.mdl.execute(self.mid, ds_in, ds_out), "execute")
        acl.rt.memcpy(self._ho.ctypes.data, self.output0_size, self._do, self.output0_size, D2H)
        return self._ho.view(np.float16)

    def close(self):
        for ptr in self._alloc:
            acl.rt.free(ptr)
        check(acl.mdl.unload(self.mid), "unload")


class PrefixModel(ACLSplitSegment):
    def __init__(self, model_path: str, max_len: int):
        super().__init__(model_path, max_len, input0_size=8, output0_size=HIDDEN_BYTES, input0_name="input_ids")

    def execute_token(self, token_id: int, position: int) -> np.ndarray:
        input_bytes = np.array([token_id], np.int64).view(np.uint8).tobytes()
        return self.execute(input_bytes, position).reshape(1, 1, HIDDEN_SIZE)


class SuffixModel(ACLSplitSegment):
    def __init__(self, model_path: str, max_len: int):
        super().__init__(model_path, max_len, input0_size=HIDDEN_BYTES, output0_size=None, input0_name="hidden_states")

    def execute_hidden(self, hidden: np.ndarray, position: int) -> np.ndarray:
        return self.execute(hidden.astype(np.float16, copy=False).tobytes(), position)


class RemoteMiddleClient:
    def __init__(self, server_url: str, max_len: int, connect_timeout: float, read_timeout: float, checksum: bool):
        self.server_url = server_url.rstrip("/")
        self.max_len = max_len
        self.timeout = max(connect_timeout, read_timeout)
        self.checksum = checksum
        self.session_id = uuid.uuid4().hex

    def _request_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    def health(self):
        req = urllib.request.Request(f"{self.server_url}/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def open(self):
        return self._request_json(
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

    def close(self):
        return self._request_json("/v1/session/close", {"session_id": self.session_id})

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
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                if len(payload) != HIDDEN_BYTES:
                    raise RuntimeError(f"bad response length: {len(payload)}")
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


def sample(logits, temp=0.7, top_k=40):
    logits = logits.astype(np.float64)
    if temp > 0:
        logits /= temp
    if 0 < top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.ones(len(logits), dtype=bool)
        mask[idx] = False
        logits[mask] = -np.inf
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(prefix_model, suffix_model, remote, tok, prompt, max_new=30, temp=0.7, top_k=40, max_len=256):
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    n_prompt = min(len(prompt_ids), max_len)
    prompt_ids = prompt_ids[-n_prompt:]
    print(f"[Prompt: {n_prompt} tokens]\n", flush=True)

    prefix_ms = 0.0
    middle_ms = 0.0
    middle_server_ms = 0.0
    suffix_ms = 0.0

    t_prefill = time.time()
    for pos, tid in enumerate(prompt_ids):
        t0 = time.perf_counter()
        hidden = prefix_model.execute_token(int(tid), pos)
        prefix_ms += (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        hidden_mid, server_ms = remote.step(hidden, pos)
        middle_ms += (time.perf_counter() - t1) * 1000.0
        middle_server_ms += server_ms

        t2 = time.perf_counter()
        suffix_model.execute_hidden(hidden_mid, pos)
        suffix_ms += (time.perf_counter() - t2) * 1000.0
    t_prefill = time.time() - t_prefill

    current_id = int(prompt_ids[-1])
    n_gen = 0
    t_decode = time.time()
    for step in range(max_new):
        pos = n_prompt + step
        if pos >= max_len:
            break

        t0 = time.perf_counter()
        hidden = prefix_model.execute_token(current_id, pos)
        prefix_ms += (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        hidden_mid, server_ms = remote.step(hidden, pos)
        middle_ms += (time.perf_counter() - t1) * 1000.0
        middle_server_ms += server_ms

        t2 = time.perf_counter()
        logits = suffix_model.execute_hidden(hidden_mid, pos)
        suffix_ms += (time.perf_counter() - t2) * 1000.0

        tid = sample(logits, temp, top_k)
        if tid == tok.eos_token_id:
            break
        current_id = tid
        n_gen += 1
        txt = tok.decode([tid], skip_special_tokens=True)
        sys.stdout.write(txt)
        sys.stdout.flush()
    t_decode = time.time() - t_decode

    tok_s = n_gen / t_decode if t_decode > 0 and n_gen > 0 else 0
    ms = t_decode / n_gen * 1000 if n_gen else 0
    print(
        f"\n\n[prefill {t_prefill:.1f}s, decode {n_gen} tok in {t_decode:.1f}s, {tok_s:.1f} tok/s, {ms:.0f} ms/tok]",
        flush=True,
    )
    print(
        f"[split timing] prefix={prefix_ms:.1f} ms, middle_rtt={middle_ms:.1f} ms, middle_cuda={middle_server_ms:.1f} ms, suffix={suffix_ms:.1f} ms",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix-model", default="/root/slm_deploy/qwen3.5_split_prefix_max256.om")
    parser.add_argument("--suffix-model", default="/root/slm_deploy/qwen3.5_split_suffix_max256.om")
    parser.add_argument("--tokenizer-dir", default="/root/slm_deploy")
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--prompt", default="你好，请介绍一下你自己")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--connect-timeout", type=float, default=1.0)
    parser.add_argument("--read-timeout", type=float, default=10.0)
    parser.add_argument("--checksum", action="store_true")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    msgs = [{"role": "user", "content": args.prompt}]
    formatted = tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    runtime = ACLRuntime()
    prefix_model = None
    suffix_model = None
    remote = RemoteMiddleClient(
        args.server_url,
        args.max_len,
        args.connect_timeout,
        args.read_timeout,
        args.checksum,
    )
    try:
        print(json.dumps(remote.health(), ensure_ascii=False), flush=True)
        print(json.dumps(remote.open(), ensure_ascii=False), flush=True)
        prefix_model = PrefixModel(args.prefix_model, args.max_len)
        suffix_model = SuffixModel(args.suffix_model, args.max_len)
        generate(
            prefix_model,
            suffix_model,
            remote,
            tok,
            formatted,
            args.max_tokens,
            args.temperature,
            args.top_k,
            args.max_len,
        )
    finally:
        try:
            print(json.dumps(remote.close(), ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"close failed: {exc}", flush=True)
        if prefix_model is not None:
            prefix_model.close()
        if suffix_model is not None:
            suffix_model.close()
        runtime.close()
        print("Done.", flush=True)


if __name__ == "__main__":
    main()
