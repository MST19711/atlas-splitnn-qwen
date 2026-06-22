#!/usr/bin/env python3
# Qwen3.5 SplitNN middle-segment HTTP service with CUDA execution.

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen35_split_common import (  # noqa: E402
    MiddleWrapper,
    ModelSpec,
    SplitConfig,
    apply_qwen35_patches,
    configure_eager_attention,
)

PROTOCOL_VERSION = 2


def parse_split(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("split must be 'prefix_end,suffix_start', e.g. '4,20'")
    return int(parts[0]), int(parts[1])


class ApiError(Exception):
    def __init__(self, status: int, error_code: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_code = error_code
        self.message = message


@dataclass
class SessionState:
    session_id: str
    max_len: int
    device: torch.device
    model_spec: ModelSpec = field(repr=False)
    nl_dn: int = 0
    nl_ga: int = 0
    created_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    position_next: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    s_cache: list[torch.Tensor] = field(default_factory=list)
    c_cache: list[torch.Tensor] = field(default_factory=list)
    k_cache: list[torch.Tensor] = field(default_factory=list)
    v_cache: list[torch.Tensor] = field(default_factory=list)
    aliases: set[str] = field(default_factory=set)
    ref_count: int = 0

    @classmethod
    def create(cls, session_id: str, max_len: int, device: torch.device,
               model_spec: ModelSpec, nl_dn: int, nl_ga: int,
               aliases: set[str] | None = None) -> "SessionState":
        conv_ks = model_spec.linear_conv_kernel_dim
        return cls(
            session_id=session_id,
            max_len=max_len,
            device=device,
            model_spec=model_spec,
            nl_dn=nl_dn,
            nl_ga=nl_ga,
            aliases=aliases or set(),
            s_cache=[
                torch.zeros((1, model_spec.linear_num_value_heads,
                               model_spec.linear_key_head_dim,
                               model_spec.linear_value_head_dim),
                            dtype=torch.float16, device=device)
                for _ in range(nl_dn)
            ],
            c_cache=[
                torch.zeros((1, model_spec.conv_dim, conv_ks - 1),
                            dtype=torch.float16, device=device)
                for _ in range(nl_dn)
            ],
            k_cache=[
                torch.zeros((1, model_spec.num_key_value_heads, max_len, model_spec.head_dim),
                            dtype=torch.float16, device=device)
                for _ in range(nl_ga)
            ],
            v_cache=[
                torch.zeros((1, model_spec.num_key_value_heads, max_len, model_spec.head_dim),
                            dtype=torch.float16, device=device)
                for _ in range(nl_ga)
            ],
        )

    def flat_cache(self) -> list[torch.Tensor]:
        return [*self.s_cache, *self.c_cache, *self.k_cache, *self.v_cache]

    def cache_bytes(self) -> int:
        total = 0
        for tensor in self.flat_cache():
            total += tensor.numel() * tensor.element_size()
        return total

    def release(self) -> None:
        self.s_cache.clear()
        self.c_cache.clear()
        self.k_cache.clear()
        self.v_cache.clear()

    def update_from_outputs(self, outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        hidden = outputs[0]
        idx = 1
        self.s_cache = list(outputs[idx : idx + self.nl_dn])
        idx += self.nl_dn
        self.c_cache = list(outputs[idx : idx + self.nl_dn])
        idx += self.nl_dn
        self.k_cache = list(outputs[idx : idx + self.nl_ga])
        idx += self.nl_ga
        self.v_cache = list(outputs[idx : idx + self.nl_ga])
        return hidden


class SplitService:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_len: int,
        split: tuple[int, int],
        session_timeout_sec: int,
        max_sessions: int,
        allow_cpu_fallback: bool = False,
    ):
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError(
                    "CUDA unavailable in the current PyTorch environment. "
                    f"torch={torch.__version__}, torch.version.cuda={torch.version.cuda}, "
                    f"torch.cuda.is_available()={torch.cuda.is_available()}. "
                    "This usually means the machine has an NVIDIA GPU but the active Python "
                    "environment only has a CPU-only PyTorch build installed."
                )
            print("[warn] CUDA unavailable, falling back to CPU for local simulation.", flush=True)
            requested_device = torch.device("cpu")
        if requested_device.type == "mps" and not torch.backends.mps.is_available():
            if not allow_cpu_fallback:
                raise RuntimeError(
                    "MPS unavailable in the current PyTorch environment. "
                    f"torch={torch.__version__}, "
                    f"torch.backends.mps.is_available()={torch.backends.mps.is_available()}."
                )
            print("[warn] MPS unavailable, falling back to CPU.", flush=True)
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.max_len = max_len
        self.session_timeout_sec = session_timeout_sec
        self.max_sessions = max_sessions
        self.allow_cpu_fallback = allow_cpu_fallback
        self.sessions: dict[str, SessionState] = {}
        self._alias_index: dict[str, str] = {}
        self.sessions_lock = threading.Lock()

        self.model_spec = ModelSpec.from_pretrained(model_path)
        self.split_config = SplitConfig(split[0], split[1], self.model_spec.num_hidden_layers)
        self.mid_nl_dn, self.mid_nl_ga = self.model_spec.compute_segment(
            *self.split_config.middle_range
        )
        self.hidden_size = self.model_spec.hidden_size
        self.hidden_shape = f"1,1,{self.hidden_size}"
        self.hidden_bytes = self.hidden_size * 2
        # Extract model size from path basename (e.g. "Qwen3.5-4B" → "4B")
        import os
        model_size = os.path.basename(os.path.normpath(model_path)).split("-")[-1]
        prefix_ct, suffix_ct = split
        self.model_name = (
            f"Qwen3.5-{model_size}"
            f"-split-{prefix_ct}-{self.split_config.suffix_start - self.split_config.prefix_end}"
            f"-{self.model_spec.num_hidden_layers - self.split_config.suffix_start}"
        )

        self._load_model(model_path)
        self._stop = threading.Event()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()

    def _release_sessions(self, session_ids: list[str]) -> int:
        released = 0
        with self.sessions_lock:
            for session_id in session_ids:
                session = self.sessions.pop(session_id, None)
                if session is None:
                    continue
                for alias in session.aliases:
                    self._alias_index.pop(alias, None)
                session.release()
                released += 1
        if released:
            try:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                elif self.device.type == "mps":
                    torch.mps.empty_cache()
            except Exception:
                pass
        return released

    def _load_model(self, model_path: str) -> None:
        apply_qwen35_patches()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        configure_eager_attention(model)
        model = model.eval().to(torch.float16).to(self.device)
        self.wrapper = MiddleWrapper(
            model.model, self.model_spec, self.split_config, self.max_len
        ).eval().to(self.device)

    def stop(self) -> None:
        self._stop.set()
        self._gc_thread.join(timeout=1)

    def _gc_loop(self) -> None:
        while not self._stop.wait(30):
            now = time.time()
            expired = []
            with self.sessions_lock:
                for session_id, session in self.sessions.items():
                    if session.ref_count == 0 and now - session.last_access_at > self.session_timeout_sec:
                        expired.append(session_id)
            self._release_sessions(expired)

    def health(self) -> dict:
        with self.sessions_lock:
            session_count = len(self.sessions)
            session_cache_bytes = sum(session.cache_bytes() for session in self.sessions.values())
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "model": self.model_name,
            "device": str(self.device),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "simulation_mode": self.device.type != "cuda",
            "sessions": session_count,
            "max_sessions": self.max_sessions,
            "session_cache_mb_estimate": round(session_cache_bytes / (1024 * 1024), 3),
            "hidden_size": self.hidden_size,
        }

    def open_session(self, payload: dict) -> dict:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing session_id")
        if payload.get("model") != self.model_name:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_MODEL", "unsupported model")
        if int(payload.get("max_len", -1)) != self.max_len:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_MAX_LEN", f"max_len must be {self.max_len}")
        if int(payload.get("hidden_size", -1)) != self.hidden_size:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SHAPE",
                           f"hidden_size must be {self.hidden_size}")
        if str(payload.get("dtype")) != "fp16":
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_DTYPE", "dtype must be fp16")
        pv = int(payload.get("protocol_version", PROTOCOL_VERSION))
        if pv not in (1, 2):
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_PROTOCOL_VERSION",
                           f"protocol_version must be 1 or 2")

        prefix_hash = str(payload.get("prefix_hash", "")).strip()
        resume_pos = payload.get("resume_token_pos")

        with self.sessions_lock:
            if prefix_hash:
                existing_id = self._alias_index.get(prefix_hash)
                if existing_id is not None:
                    session = self.sessions.get(existing_id)
                    if session is not None:
                        session.ref_count += 1
                        session.last_access_at = time.time()
                        if resume_pos is not None:
                            session.position_next = int(resume_pos)
                        return {
                            "ok": True,
                            "session_id": existing_id,
                            "max_len": self.max_len,
                            "hidden_size": self.hidden_size,
                            "dtype": "fp16",
                            "server_device": str(self.device),
                            "session_resumed": True,
                        }

            if session_id in self.sessions:
                raise ApiError(HTTPStatus.CONFLICT, "SESSION_EXISTS", "session already exists")
            if len(self.sessions) >= self.max_sessions:
                # Try to evict LRU with ref_count == 0
                expired = []
                for sid, s in self.sessions.items():
                    if s.ref_count == 0:
                        expired.append(sid)
                if not expired:
                    raise ApiError(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "TOO_MANY_SESSIONS",
                        f"active sessions exceed limit {self.max_sessions}",
                    )
                for sid in expired[:1]:
                    self._release_sessions([sid])

            aliases = {prefix_hash} if prefix_hash else set()
            session = SessionState.create(
                session_id, self.max_len, self.device,
                self.model_spec, self.mid_nl_dn, self.mid_nl_ga,
                aliases=aliases,
            )
            if resume_pos is not None:
                session.position_next = int(resume_pos)
            session.ref_count = 1
            self.sessions[session_id] = session
            for alias in aliases:
                self._alias_index[alias] = session_id

        return {
            "ok": True,
            "session_id": session_id,
            "max_len": self.max_len,
            "hidden_size": self.hidden_size,
            "dtype": "fp16",
            "server_device": str(self.device),
            "server_model_segment": (
                f"layers[{self.split_config.prefix_end}:{self.split_config.suffix_start}]"
            ),
            "session_resumed": False,
        }

    def close_session(self, payload: dict) -> dict:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing session_id")
        evict = payload.get("evict", False)
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                return {"ok": True, "session_id": session_id, "released": False}
            if session.ref_count > 0:
                session.ref_count -= 1
            session.last_access_at = time.time()
            if evict:
                released = self._release_sessions([session_id]) > 0
                return {"ok": True, "session_id": session_id, "released": released}
        return {"ok": True, "session_id": session_id, "released": False}

    def step_session(self, headers, body: bytes) -> tuple[bytes, dict]:
        session_id = headers.get("X-Session-Id", "").strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing X-Session-Id")
        try:
            position = int(headers.get("X-Position", "-1"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_POSITION", "invalid X-Position") from exc

        pv = int(headers.get("X-Protocol-Version", str(PROTOCOL_VERSION)))
        if pv not in (1, 2):
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_PROTOCOL_VERSION",
                           "unsupported protocol version")
        if headers.get("X-Hidden-Shape") != self.hidden_shape:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SHAPE",
                           f"shape must be {self.hidden_shape}")
        if headers.get("X-DType") != "fp16":
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_DTYPE", "dtype must be fp16")
        try:
            byte_length = int(headers.get("X-Byte-Length", "-1"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_LENGTH", "invalid X-Byte-Length") from exc
        if byte_length != self.hidden_bytes or len(body) != self.hidden_bytes:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_LENGTH",
                           f"payload must be {self.hidden_bytes} bytes")

        checksum = headers.get("X-Checksum")
        if checksum:
            actual = f"{zlib.crc32(body) & 0xffffffff:08x}"
            if checksum.lower() != actual:
                raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_CHECKSUM", "checksum mismatch")

        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "SESSION_NOT_FOUND", "session not found")

        with session.lock:
            if pv >= 2:
                session.position_next = position
            elif position != session.position_next:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "POSITION_MISMATCH",
                    f"expected position {session.position_next}, got {position}",
                )
            if position >= session.max_len:
                raise ApiError(HTTPStatus.BAD_REQUEST, "MAX_LEN_EXCEEDED",
                               "position exceeds max_len")

            start = time.perf_counter()
            try:
                hidden_np = np.frombuffer(body, dtype=np.float16).reshape(1, 1, self.hidden_size)
                hidden = torch.from_numpy(hidden_np.copy()).to(self.device, non_blocking=True)
                pos = torch.tensor([position], dtype=torch.int64, device=self.device)
                with torch.no_grad():
                    outputs = self.wrapper(hidden, pos, *session.flat_cache())
                hidden_out = session.update_from_outputs(outputs)
                session.position_next += 1
                session.last_access_at = time.time()
                hidden_bytes = (hidden_out.detach().to("cpu").contiguous()
                                .numpy().astype(np.float16).tobytes())
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, RuntimeError) and "MPS" not in str(exc) and "out of memory" not in str(exc).lower():
                    raise
                self._release_sessions([session_id])
                raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "OOM", str(exc)) from exc

            latency_ms = (time.perf_counter() - start) * 1000.0
            response_headers = {
                "X-Session-Id": session_id,
                "X-Position": str(position),
                "X-Hidden-Shape": self.hidden_shape,
                "X-DType": "fp16",
                "X-Byte-Length": str(self.hidden_bytes),
                "X-Server-Latency-Ms": f"{latency_ms:.3f}",
                "X-Protocol-Version": str(PROTOCOL_VERSION),
            }
            if checksum:
                response_headers["X-Checksum"] = f"{zlib.crc32(hidden_bytes) & 0xffffffff:08x}"
            return hidden_bytes, response_headers


SERVICE: Optional[SplitService] = None


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "Qwen35SplitService/2.0"

    def log_message(self, fmt, *args):
        print(
            f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}",
            flush=True,
        )

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_JSON", "invalid JSON body") from exc

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, body: bytes, headers: dict) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self) -> None:
        assert SERVICE is not None
        if self.command == "GET" and self.path == "/v1/health":
            self._send_json(HTTPStatus.OK, SERVICE.health())
            return
        if self.command == "POST" and self.path == "/v1/session/open":
            self._send_json(HTTPStatus.OK, SERVICE.open_session(self._read_json()))
            return
        if self.command == "POST" and self.path == "/v1/session/close":
            self._send_json(HTTPStatus.OK, SERVICE.close_session(self._read_json()))
            return
        if self.command == "POST" and self.path == "/v1/session/step":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload, headers = SERVICE.step_session(self.headers, body)
            self._send_binary(payload, headers)
            return
        raise ApiError(HTTPStatus.NOT_FOUND, "NOT_FOUND", "endpoint not found")

    def do_GET(self):
        try:
            self._dispatch()
        except ApiError as exc:
            self._send_json(exc.status, {"ok": False, "error_code": exc.error_code,
                                          "message": exc.message})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error_code": "INTERNAL_ERROR", "message": str(exc)},
            )

    def do_POST(self):
        try:
            self._dispatch()
        except ApiError as exc:
            self._send_json(exc.status, {"ok": False, "error_code": exc.error_code,
                                          "message": exc.message})
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error_code": "INTERNAL_ERROR", "message": str(exc)},
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--session-timeout-sec", type=int, default=300)
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--split", type=parse_split, default=(4, 20),
                        help="prefix_end,suffix_start  (e.g. 4,20 for 4/16/4)")
    parser.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="仅用于本地模拟；CUDA 不可用时退回 CPU。",
    )
    args = parser.parse_args()

    global SERVICE
    SERVICE = SplitService(
        model_path=args.model_path,
        device=args.device,
        max_len=args.max_len,
        split=args.split,
        session_timeout_sec=args.session_timeout_sec,
        max_sessions=args.max_sessions,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )
    server = ThreadedHTTPServer((args.host, args.port), Handler)
    server.socket.settimeout(10.0)
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        SERVICE.stop()


if __name__ == "__main__":
    main()
