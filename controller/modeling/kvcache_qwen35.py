from __future__ import annotations

import gc
import threading
from dataclasses import dataclass

import numpy as np

from controller.cache.snapshot import CacheSnapshot
from controller.engine.constants import M, H2D, D2H
from controller.modeling.base import ModelInfo, Qwen35Model, Qwen35Session
from scripts.qwen35_model_spec import ModelSpec


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


def _device_to_host(acl, dev_ptr: int, size: int) -> np.ndarray:
    host_buf = np.empty(size, np.uint8)
    _check(acl.rt.memcpy(host_buf.ctypes.data, size, dev_ptr, size, D2H), "D2H")
    return host_buf


def _host_to_device(acl, dev_ptr: int, data: np.ndarray) -> None:
    _check(acl.rt.memcpy(dev_ptr, data.nbytes, data.ctypes.data, data.nbytes, H2D), "H2D")


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
        self._s_a: list[int] = []
        self._c_a: list[int] = []
        self._k_a: list[int] = []
        self._v_a: list[int] = []
        self._s_b: list[int] = []
        self._c_b: list[int] = []
        self._k_b: list[int] = []
        self._v_b: list[int] = []
        self._zero_cache_host: dict[int, np.ndarray] = {}

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
        self._s_a, self._c_a, self._k_a, self._v_a = s_a, c_a, k_a, v_a
        self._s_b, self._c_b, self._k_b, self._v_b = s_b, c_b, k_b, v_b
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

    def prepare_fresh(self) -> None:
        """Reset step state and zero both device-side cache sets for a new request."""
        assert self.acl is not None
        with self._lock:
            self.bind_thread_context()
            self._zero_cache_set(self._s_a, self.layout.s_bytes)
            self._zero_cache_set(self._c_a, self.layout.c_bytes)
            self._zero_cache_set(self._k_a, self.layout.kv_bytes)
            self._zero_cache_set(self._v_a, self.layout.kv_bytes)
            self._zero_cache_set(self._s_b, self.layout.s_bytes)
            self._zero_cache_set(self._c_b, self.layout.c_bytes)
            self._zero_cache_set(self._k_b, self.layout.kv_bytes)
            self._zero_cache_set(self._v_b, self.layout.kv_bytes)
            self._step = 0

    def _zero_cache_set(self, dev_ptrs: list[int], size: int) -> None:
        for dev_ptr in dev_ptrs:
            self._zero_device_ptr(dev_ptr, size)

    def _zero_device_ptr(self, dev_ptr: int, size: int) -> None:
        memset = getattr(self.acl.rt, "memset", None)
        if memset is not None:
            _check(memset(dev_ptr, size, 0, size), "acl.rt.memset")
            return
        zero_host = self._zero_cache_host.get(size)
        if zero_host is None:
            zero_host = np.zeros(size, np.uint8)
            self._zero_cache_host[size] = zero_host
        _host_to_device(self.acl, dev_ptr, zero_host)

    def bind_thread_context(self) -> None:
        if self.acl is None or self._context is None:
            raise Qwen35KvCacheError("ACL runtime is not initialized")
        _check(self.acl.rt.set_context(self._context), "acl.rt.set_context")

    def _active_cache_set(self):
        if self._step % 2 == 1:
            return self._s_b, self._c_b, self._k_b, self._v_b
        return self._s_a, self._c_a, self._k_a, self._v_a

    def snapshot_private(self, position: int) -> CacheSnapshot:
        assert self.acl is not None
        with self._lock:
            self.bind_thread_context()
            layout = self.layout
            spec = self.model_spec

            if self._step == 0:
                def _zeros(nl, size):
                    return [np.zeros(size, np.uint8) for _ in range(nl)]
                s = _zeros(layout.nl_dn, layout.s_bytes)
                c = _zeros(layout.nl_dn, layout.c_bytes)
                k = _zeros(layout.nl_ga, layout.kv_bytes)
                v = _zeros(layout.nl_ga, layout.kv_bytes)
            else:
                out_s, out_c, out_k, out_v = self._active_cache_set()
                s = [_device_to_host(self.acl, ptr, layout.s_bytes) for ptr in out_s]
                c = [_device_to_host(self.acl, ptr, layout.c_bytes) for ptr in out_c]
                k = [_device_to_host(self.acl, ptr, layout.kv_bytes) for ptr in out_k]
                v = [_device_to_host(self.acl, ptr, layout.kv_bytes) for ptr in out_v]

            kvh = spec.num_key_value_heads
            hdim = spec.head_dim
            max_l = self.max_len

            s_states = [
                buf.view(np.float16).reshape(1, spec.linear_num_value_heads,
                                             spec.linear_key_head_dim,
                                             spec.linear_value_head_dim).copy()
                for buf in s
            ]
            c_states = [
                buf.view(np.float16).reshape(1, spec.conv_dim,
                                             spec.linear_conv_kernel_dim - 1).copy()
                for buf in c
            ]
            k_states = [
                buf.view(np.float16).reshape(1, kvh, max_l, hdim).copy()
                for buf in k
            ]
            v_states = [
                buf.view(np.float16).reshape(1, kvh, max_l, hdim).copy()
                for buf in v
            ]
            return CacheSnapshot(s_states=s_states, c_states=c_states,
                                 k_states=k_states, v_states=v_states)

    def restore_private(self, snap: CacheSnapshot, position: int) -> None:
        assert self.acl is not None
        with self._lock:
            self.bind_thread_context()
            layout = self.layout
            spec = self.model_spec
            kvh = spec.num_key_value_heads
            hdim = spec.head_dim
            max_l = self.max_len
            tok_bytes = kvh * hdim * 2

            def _h2d_list(dev_ptrs, host_list):
                for dev_ptr, host_arr in zip(dev_ptrs, host_list):
                    raw = host_arr.view(np.uint8).ravel()
                    _host_to_device(self.acl, dev_ptr, raw)

            s = [snap.s_states[i] for i in range(layout.nl_dn)]
            c = [snap.c_states[i] for i in range(layout.nl_dn)]

            k_fixed = []
            v_fixed = []
            for i in range(layout.nl_ga):
                k_arr = snap.k_states[i].copy()
                k_arr[0, :, position:, :] = 0
                k_fixed.append(k_arr)
                v_arr = snap.v_states[i].copy()
                v_arr[0, :, position:, :] = 0
                v_fixed.append(v_arr)

            _h2d_list(self._s_a, s)
            _h2d_list(self._c_a, c)
            _h2d_list(self._k_a, k_fixed)
            _h2d_list(self._v_a, v_fixed)

            _h2d_list(self._s_b, s)
            _h2d_list(self._c_b, c)
            _h2d_list(self._k_b, k_fixed)
            _h2d_list(self._v_b, v_fixed)

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
        self.runtime.prepare_fresh()

    def prefill(self, input_ids: list[int], position: int = 0) -> np.ndarray:
        if not input_ids:
            raise ValueError("empty input_ids")
        self.position = position
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

    def snapshot(self) -> CacheSnapshot:
        return self.runtime.snapshot_private(self.position)

    def restore(self, snap: CacheSnapshot, position: int) -> None:
        self.runtime.restore_private(snap, position)
        self.position = position
        self.runtime.reset()


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

    def create_session(self, cache_entry=None) -> Qwen35Session:
        return Qwen35KvCacheSession(self.runtime)
