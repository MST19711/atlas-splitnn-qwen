from __future__ import annotations

import json
import urllib.error
import urllib.request
import zlib

import numpy as np

PROTOCOL_VERSION = 1


class RemoteMiddleError(RuntimeError):
    pass


class RemoteMiddleClient:
    def __init__(self, server_url: str, model_name: str, hidden_size: int, max_len: int,
                 connect_timeout: float = 1.0, read_timeout: float = 30.0, checksum: bool = False):
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.hidden_bytes = hidden_size * 2
        self.hidden_shape = f"1,1,{hidden_size}"
        self.max_len = max_len
        self.timeout = max(connect_timeout, read_timeout)
        self.checksum = checksum

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
            raise RemoteMiddleError(f"HTTP {exc.code}: {body}") from exc

    def health(self) -> dict:
        req = urllib.request.Request(f"{self.server_url}/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def open(self, session_id: str) -> dict:
        return self._request_json(
            "/v1/session/open",
            {
                "session_id": session_id,
                "model": self.model_name,
                "max_len": self.max_len,
                "hidden_size": self.hidden_size,
                "dtype": "fp16",
                "protocol_version": PROTOCOL_VERSION,
            },
        )

    def close(self, session_id: str) -> dict:
        return self._request_json("/v1/session/close", {"session_id": session_id})

    def step(self, session_id: str, hidden_state: np.ndarray, position: int) -> tuple[np.ndarray, float]:
        body = hidden_state.astype(np.float16, copy=False).reshape(1, 1, self.hidden_size).tobytes()
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Session-Id": session_id,
            "X-Protocol-Version": str(PROTOCOL_VERSION),
            "X-Position": str(position),
            "X-Hidden-Shape": self.hidden_shape,
            "X-DType": "fp16",
            "X-Byte-Length": str(self.hidden_bytes),
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
                if len(payload) != self.hidden_bytes:
                    raise RemoteMiddleError(f"bad response length: {len(payload)}")
                if self.checksum:
                    want = resp.headers.get("X-Checksum")
                    got = f"{zlib.crc32(payload) & 0xffffffff:08x}"
                    if want and want.lower() != got:
                        raise RemoteMiddleError("response checksum mismatch")
                latency_ms = float(resp.headers.get("X-Server-Latency-Ms", "0"))
                return (np.frombuffer(payload, dtype=np.float16)
                        .reshape(1, 1, self.hidden_size).copy(), latency_ms)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RemoteMiddleError(f"HTTP {exc.code}: {body_text}") from exc
