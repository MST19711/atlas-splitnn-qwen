#!/usr/bin/env python3
"""Qwen3.5 SplitNN middle-segment HTTP service with CUDA execution."""

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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from qwen35_split_common import (  # noqa: E402
    CONV_D,
    CONV_KS,
    HDIM,
    HIDDEN_SIZE,
    KV_H,
    K_DIM,
    K_H,
    MIDDLE_NL_DN,
    MIDDLE_NL_GA,
    MiddleWrapper,
    V_DIM,
    apply_qwen35_patches,
    configure_eager_attention,
)

MODEL_NAME = "Qwen3.5-0.8B-split-4-16-4"
PROTOCOL_VERSION = 1
HIDDEN_SHAPE = "1,1,1024"
HIDDEN_BYTES = HIDDEN_SIZE * 2


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
    created_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    position_next: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    s_cache: list[torch.Tensor] = field(default_factory=list)
    c_cache: list[torch.Tensor] = field(default_factory=list)
    k_cache: list[torch.Tensor] = field(default_factory=list)
    v_cache: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def create(cls, session_id: str, max_len: int, device: torch.device) -> "SessionState":
        return cls(
            session_id=session_id,
            max_len=max_len,
            device=device,
            s_cache=[
                torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16, device=device)
                for _ in range(MIDDLE_NL_DN)
            ],
            c_cache=[
                torch.zeros((1, CONV_D, CONV_KS - 1), dtype=torch.float16, device=device)
                for _ in range(MIDDLE_NL_DN)
            ],
            k_cache=[
                torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16, device=device)
                for _ in range(MIDDLE_NL_GA)
            ],
            v_cache=[
                torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16, device=device)
                for _ in range(MIDDLE_NL_GA)
            ],
        )

    def flat_cache(self) -> list[torch.Tensor]:
        return [*self.s_cache, *self.c_cache, *self.k_cache, *self.v_cache]

    def update_from_outputs(self, outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        hidden = outputs[0]
        idx = 1
        self.s_cache = list(outputs[idx : idx + MIDDLE_NL_DN])
        idx += MIDDLE_NL_DN
        self.c_cache = list(outputs[idx : idx + MIDDLE_NL_DN])
        idx += MIDDLE_NL_DN
        self.k_cache = list(outputs[idx : idx + MIDDLE_NL_GA])
        idx += MIDDLE_NL_GA
        self.v_cache = list(outputs[idx : idx + MIDDLE_NL_GA])
        return hidden


class SplitService:
    def __init__(
        self,
        model_path: str,
        device: str,
        max_len: int,
        session_timeout_sec: int,
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
        self.device = requested_device
        self.max_len = max_len
        self.session_timeout_sec = session_timeout_sec
        self.allow_cpu_fallback = allow_cpu_fallback
        self.sessions: dict[str, SessionState] = {}
        self.sessions_lock = threading.Lock()
        self._load_model(model_path)
        self._stop = threading.Event()
        self._gc_thread = threading.Thread(target=self._gc_loop, daemon=True)
        self._gc_thread.start()

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
        self.wrapper = MiddleWrapper(model.model, self.max_len).eval().to(self.device)

    def stop(self) -> None:
        self._stop.set()
        self._gc_thread.join(timeout=1)

    def _gc_loop(self) -> None:
        while not self._stop.wait(30):
            now = time.time()
            expired = []
            with self.sessions_lock:
                for session_id, session in self.sessions.items():
                    if now - session.last_access_at > self.session_timeout_sec:
                        expired.append(session_id)
                for session_id in expired:
                    del self.sessions[session_id]

    def health(self) -> dict:
        with self.sessions_lock:
            session_count = len(self.sessions)
        return {
            "ok": True,
            "protocol_version": PROTOCOL_VERSION,
            "model": MODEL_NAME,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "simulation_mode": self.device.type != "cuda",
            "sessions": session_count,
        }

    def open_session(self, payload: dict) -> dict:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing session_id")
        if payload.get("model") != MODEL_NAME:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_MODEL", "unsupported model")
        if int(payload.get("max_len", -1)) != self.max_len:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_MAX_LEN", f"max_len must be {self.max_len}")
        if int(payload.get("hidden_size", -1)) != HIDDEN_SIZE:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SHAPE", f"hidden_size must be {HIDDEN_SIZE}")
        if str(payload.get("dtype")) != "fp16":
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_DTYPE", "dtype must be fp16")
        if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "BAD_PROTOCOL_VERSION",
                f"protocol_version must be {PROTOCOL_VERSION}",
            )

        with self.sessions_lock:
            if session_id in self.sessions:
                raise ApiError(HTTPStatus.CONFLICT, "SESSION_EXISTS", "session already exists")
            self.sessions[session_id] = SessionState.create(session_id, self.max_len, self.device)

        return {
            "ok": True,
            "session_id": session_id,
            "max_len": self.max_len,
            "hidden_size": HIDDEN_SIZE,
            "dtype": "fp16",
            "server_device": str(self.device),
            "server_model_segment": "layers[4:20]",
        }

    def close_session(self, payload: dict) -> dict:
        session_id = str(payload.get("session_id", "")).strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing session_id")
        with self.sessions_lock:
            released = self.sessions.pop(session_id, None) is not None
        return {"ok": True, "session_id": session_id, "released": released}

    def step_session(self, headers, body: bytes) -> tuple[bytes, dict]:
        session_id = headers.get("X-Session-Id", "").strip()
        if not session_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SESSION_ID", "missing X-Session-Id")
        try:
            position = int(headers.get("X-Position", "-1"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_POSITION", "invalid X-Position") from exc

        if headers.get("X-Protocol-Version") != str(PROTOCOL_VERSION):
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_PROTOCOL_VERSION", "unsupported protocol version")
        if headers.get("X-Hidden-Shape") != HIDDEN_SHAPE:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_SHAPE", f"shape must be {HIDDEN_SHAPE}")
        if headers.get("X-DType") != "fp16":
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_DTYPE", "dtype must be fp16")
        try:
            byte_length = int(headers.get("X-Byte-Length", "-1"))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_LENGTH", "invalid X-Byte-Length") from exc
        if byte_length != HIDDEN_BYTES or len(body) != HIDDEN_BYTES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_LENGTH", f"payload must be {HIDDEN_BYTES} bytes")

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
            if position != session.position_next:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "POSITION_MISMATCH",
                    f"expected position {session.position_next}, got {position}",
                )
            if position >= session.max_len:
                raise ApiError(HTTPStatus.BAD_REQUEST, "MAX_LEN_EXCEEDED", "position exceeds max_len")

            start = time.perf_counter()
            try:
                hidden_np = np.frombuffer(body, dtype=np.float16).reshape(1, 1, HIDDEN_SIZE)
                hidden = torch.from_numpy(hidden_np.copy()).to(self.device, non_blocking=True)
                pos = torch.tensor([position], dtype=torch.int64, device=self.device)
                with torch.no_grad():
                    outputs = self.wrapper(hidden, pos, *session.flat_cache())
                hidden_out = session.update_from_outputs(outputs)
                session.position_next += 1
                session.last_access_at = time.time()
                hidden_bytes = hidden_out.detach().to("cpu").contiguous().numpy().astype(np.float16).tobytes()
            except torch.cuda.OutOfMemoryError as exc:
                raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "CUDA_OOM", str(exc)) from exc

            latency_ms = (time.perf_counter() - start) * 1000.0
            response_headers = {
                "X-Session-Id": session_id,
                "X-Position": str(position),
                "X-Hidden-Shape": HIDDEN_SHAPE,
                "X-DType": "fp16",
                "X-Byte-Length": str(HIDDEN_BYTES),
                "X-Server-Latency-Ms": f"{latency_ms:.3f}",
            }
            if checksum:
                response_headers["X-Checksum"] = f"{zlib.crc32(hidden_bytes) & 0xffffffff:08x}"
            return hidden_bytes, response_headers


SERVICE: Optional[SplitService] = None


class Handler(BaseHTTPRequestHandler):
    server_version = "Qwen35SplitService/1.0"

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
            self._send_json(exc.status, {"ok": False, "error_code": exc.error_code, "message": exc.message})
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
            self._send_json(exc.status, {"ok": False, "error_code": exc.error_code, "message": exc.message})
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--session-timeout-sec", type=int, default=300)
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
        session_timeout_sec=args.session_timeout_sec,
        allow_cpu_fallback=args.allow_cpu_fallback,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
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
