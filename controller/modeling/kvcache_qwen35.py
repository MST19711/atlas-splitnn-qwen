from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import numpy as np

from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session
from scripts.qwen35_model_spec import ModelSpec

from controller.engine.constants import M, H2D, D2H


class Qwen35KvCacheError(RuntimeError):
    pass


def _check(ret: int, msg: str) -> None:
    if ret != 0:
        raise Qwen35KvCacheError(f"{msg} failed, ret={ret}")


def _s_bytes_from_spec(spec: ModelSpec) -> int:
    return (
        1
        * spec.linear_num_value_heads
        * spec.linear_key_head_dim
        * spec.linear_value_head_dim
        * 2
    )


def _c_bytes_from_spec(spec: ModelSpec) -> int:
    return 1 * spec.conv_dim * (spec.linear_conv_kernel_dim - 1) * 2


@dataclass
class _KvLayout:
    nl_dn: int
    nl_ga: int
    kv_heads: int
    head_dim: int
    s_bytes: int
    c_bytes: int
    kv_bytes: int


class _ACLSessionRuntime:
    def __init__(self, model_path: str, model_spec: ModelSpec, max_len: int):
        self.model_path = model_path
        self.model_spec = model_spec
        self.max_len = max_len
        self.acl = None
        self.mid = None
        self.desc = None
        self.out_sz = None
        self.layout = self._build_layout(model_spec, max_len)
        self._alloc: list[int] = []
        self._datasets: list[tuple[object, object, list[object], list[object]]] = []
        self._host_ids = np.empty(8, np.uint8)
        self._host_pos = np.empty(8, np.uint8)
        self._host_logits: np.ndarray | None = None
        self._di = None
        self._dp = None
        self._dl = None
        self._step = 0
        self._context = None
        self._lock = threading.Lock()

    @staticmethod
    def _build_layout(model_spec: ModelSpec, max_len: int) -> _KvLayout:
        nl_dn, nl_ga = model_spec.compute_segment(0, model_spec.num_hidden_layers)
        return _KvLayout(
            nl_dn=nl_dn,
            nl_ga=nl_ga,
            kv_heads=model_spec.num_key_value_heads,
            head_dim=model_spec.head_dim,
            s_bytes=_s_bytes_from_spec(model_spec),
            c_bytes=_c_bytes_from_spec(model_spec),
            kv_bytes=1 * model_spec.num_key_value_heads * max_len * model_spec.head_dim * 2,
        )

    def load(self) -> None:
        if self.acl is not None:
            return
        try:
            import sys

            sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
            import acl  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise Qwen35KvCacheError(f"failed to import acl: {exc}") from exc
        gc.collect()
        self.acl = acl
        _check(acl.init(), "acl.init")
        _check(acl.rt.set_device(0), "acl.rt.set_device")
        self._context, ret = acl.rt.get_context()
        _check(ret, "acl.rt.get_context")
        self.mid, ret = acl.mdl.load_from_file(self.model_path)
        _check(ret, "acl.mdl.load_from_file")
        self.desc = acl.mdl.create_desc()
        _check(acl.mdl.get_desc(self.desc, self.mid), "acl.mdl.get_desc")
        self.out_sz = acl.mdl.get_output_size_by_index(self.desc, 0)
        self._host_logits = np.empty(self.out_sz, np.uint8)
        self._di = self._alloc_ptr(8)
        self._dp = self._alloc_ptr(8)
        self._dl = self._alloc_ptr(self.out_sz)
        s_a, c_a, k_a, v_a = self._alloc_cache_set()
        s_b, c_b, k_b, v_b = self._alloc_cache_set()
        self._datasets = [
            self._create_dataset_pair(s_a, c_a, k_a, v_a, s_b, c_b, k_b, v_b),
            self._create_dataset_pair(s_b, c_b, k_b, v_b, s_a, c_a, k_a, v_a),
        ]

    def _alloc_ptr(self, size: int):
        assert self.acl is not None
        ptr, ret = self.acl.rt.malloc(size, M)
        _check(ret, "acl.rt.malloc")
        self._alloc.append(ptr)
        return ptr

    def _alloc_cache_set(self):
        layout = self.layout
        return (
            [self._alloc_ptr(layout.s_bytes) for _ in range(layout.nl_dn)],
            [self._alloc_ptr(layout.c_bytes) for _ in range(layout.nl_dn)],
            [self._alloc_ptr(layout.kv_bytes) for _ in range(layout.nl_ga)],
            [self._alloc_ptr(layout.kv_bytes) for _ in range(layout.nl_ga)],
        )

    def _create_dataset_pair(self, s_src, c_src, k_src, v_src, s_dst, c_dst, k_dst, v_dst):
        assert self.acl is not None
        ds_in = self.acl.mdl.create_dataset()
        ds_out = self.acl.mdl.create_dataset()
        bufs_in, bufs_out = [], []
        for ptr, sz in [(self._di, 8), (self._dp, 8)]:
            buf = self.acl.create_data_buffer(ptr, sz)
            bufs_in.append(buf)
            _, ret = self.acl.mdl.add_dataset_buffer(ds_in, buf)
            _check(ret, "add_dataset_buffer input")
        for ptrs, sz in [
            (s_src, self.layout.s_bytes),
            (c_src, self.layout.c_bytes),
            (k_src, self.layout.kv_bytes),
            (v_src, self.layout.kv_bytes),
        ]:
            for ptr in ptrs:
                buf = self.acl.create_data_buffer(ptr, sz)
                bufs_in.append(buf)
                _, ret = self.acl.mdl.add_dataset_buffer(ds_in, buf)
                _check(ret, "add_dataset_buffer cache input")
        buf = self.acl.create_data_buffer(self._dl, self.out_sz)
        bufs_out.append(buf)
        _, ret = self.acl.mdl.add_dataset_buffer(ds_out, buf)
        _check(ret, "add_dataset_buffer logits output")
        for ptrs, sz in [
            (s_dst, self.layout.s_bytes),
            (c_dst, self.layout.c_bytes),
            (k_dst, self.layout.kv_bytes),
            (v_dst, self.layout.kv_bytes),
        ]:
            for ptr in ptrs:
                buf = self.acl.create_data_buffer(ptr, sz)
                bufs_out.append(buf)
                _, ret = self.acl.mdl.add_dataset_buffer(ds_out, buf)
                _check(ret, "add_dataset_buffer cache output")
        return ds_in, ds_out, bufs_in, bufs_out

    def reset(self) -> None:
        self._step = 0

    def bind_thread_context(self) -> None:
        if self.acl is None or self._context is None:
            raise Qwen35KvCacheError("ACL runtime is not initialized")
        _check(self.acl.rt.set_context(self._context), "acl.rt.set_context")

    def execute(self, token_id: int, position: int) -> np.ndarray:
        assert self.acl is not None and self._host_logits is not None
        with self._lock:
            self.bind_thread_context()
            self._host_ids[:8] = np.array([token_id], np.int64).view(np.uint8)
            self._host_pos[:8] = np.array([position], np.int64).view(np.uint8)
            self.acl.rt.memcpy(self._di, 8, self._host_ids.ctypes.data, 8, H2D)
            self.acl.rt.memcpy(self._dp, 8, self._host_pos.ctypes.data, 8, H2D)
            ds_in, ds_out, _, _ = self._datasets[self._step % 2]
            self._step += 1
            _check(self.acl.mdl.execute(self.mid, ds_in, ds_out), "acl.mdl.execute")
            self.acl.rt.memcpy(self._host_logits.ctypes.data, self.out_sz, self._dl, self.out_sz, D2H)
            return self._host_logits.view(np.float16).flatten().copy()

    def close(self) -> None:
        if self.acl is None:
            return
        for ds_in, ds_out, bufs_in, bufs_out in self._datasets:
            for buf in bufs_in + bufs_out:
                self.acl.destroy_data_buffer(buf)
            self.acl.mdl.destroy_dataset(ds_in)
            self.acl.mdl.destroy_dataset(ds_out)
        self._datasets = []
        for ptr in self._alloc:
            self.acl.rt.free(ptr)
        self._alloc = []
        if self.mid is not None:
            _check(self.acl.mdl.unload(self.mid), "acl.mdl.unload")
        if self.desc is not None:
            self.acl.mdl.destroy_desc(self.desc)
        _check(self.acl.rt.reset_device(0), "acl.rt.reset_device")
        _check(self.acl.finalize(), "acl.finalize")
        self.acl = None
        self._context = None


class Qwen35KvCacheSession(Qwen35Session):
    def __init__(self, runtime: _ACLSessionRuntime):
        self.runtime = runtime
        self.position = 0
        self.closed = False
        self.runtime.reset()

    def prefill(self, input_ids: list[int]) -> np.ndarray:
        if not input_ids:
            raise ValueError("empty input_ids")
        logits = None
        for token_id in input_ids:
            logits = self.decode_next(int(token_id))
        assert logits is not None
        return logits

    def decode_next(self, token_id: int) -> np.ndarray:
        if self.closed:
            raise RuntimeError("session already closed")
        logits = self.runtime.execute(int(token_id), self.position)
        self.position += 1
        return np.asarray(logits, dtype=np.float16).reshape(1, 1, -1)

    def close(self) -> None:
        self.closed = True


class Qwen35KvCacheModel(Qwen35Model):
    def __init__(self, model_name: str, model_path: str, model_spec: ModelSpec, max_len: int):
        super().__init__(
            ModelInfo(
                model_name=model_name,
                max_len=max_len,
                vocab_size=model_spec.vocab_size,
                backend_kind="qwen35_kvcache_om",
            )
        )
        self.model_spec = model_spec
        self.runtime = _ACLSessionRuntime(model_path=model_path, model_spec=model_spec, max_len=max_len)

    def load(self) -> None:
        self.runtime.load()

    def close(self) -> None:
        self.runtime.close()

    def is_loaded(self) -> bool:
        return self.runtime.acl is not None

    def create_session(self) -> Qwen35Session:
        return Qwen35KvCacheSession(self.runtime)
