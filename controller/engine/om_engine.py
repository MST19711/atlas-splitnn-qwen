from __future__ import annotations

import numpy as np

from controller.engine.base import SplitEngine

M, H2D, D2H = 0, 1, 2
NL_DN = 3
NL_GA = 1
K_H = 16
K_DIM = 128
V_DIM = 128
CONV_D = 6144
CONV_KS = 4
KV_H = 2
HDIM = 256
HIDDEN_SIZE = 1024
S_BYTES = 1 * K_H * K_DIM * V_DIM * 2
C_BYTES = 1 * CONV_D * (CONV_KS - 1) * 2


class OmEngineError(RuntimeError):
    pass


class _ACLRuntime:
    def __init__(self, acl):
        self.acl = acl
        self._initialized = False

    def init(self) -> None:
        if self._initialized:
            return
        ret = self.acl.init()
        if ret != 0:
            raise OmEngineError(f"acl.init failed: {ret}")
        ret = self.acl.rt.set_device(0)
        if ret != 0:
            raise OmEngineError(f"acl.rt.set_device failed: {ret}")
        self._initialized = True

    def close(self) -> None:
        if not self._initialized:
            return
        self.acl.rt.reset_device(0)
        self.acl.finalize()
        self._initialized = False


class _ACLSplitSegment:
    def __init__(self, acl, model_path: str, max_len: int, input0_size: int, output0_size: int | None, input0_name: str):
        self.acl = acl
        self.max_len = max_len
        self.kv_bytes = 1 * KV_H * max_len * HDIM * 2
        self.input0_size = input0_size
        self.input0_name = input0_name

        self.mid, ret = acl.mdl.load_from_file(model_path)
        self._check(ret, "load")
        self.desc = acl.mdl.create_desc()
        self._check(acl.mdl.get_desc(self.desc, self.mid), "get_desc")
        self.output0_size = output0_size
        if self.output0_size is None:
            self.output0_size = acl.mdl.get_output_size_by_index(self.desc, 0)

        self._alloc: list[int] = []
        self._d0 = self._alloc_ptr(input0_size)
        self._dp = self._alloc_ptr(8)
        self._do = self._alloc_ptr(self.output0_size)

        self._sA, self._cA, self._kA, self._vA = self._alloc_set()
        self._sB, self._cB, self._kB, self._vB = self._alloc_set()
        self._cache_ptrs = (
            [(ptr, S_BYTES) for ptr in self._sA]
            + [(ptr, C_BYTES) for ptr in self._cA]
            + [(ptr, self.kv_bytes) for ptr in self._kA]
            + [(ptr, self.kv_bytes) for ptr in self._vA]
            + [(ptr, S_BYTES) for ptr in self._sB]
            + [(ptr, C_BYTES) for ptr in self._cB]
            + [(ptr, self.kv_bytes) for ptr in self._kB]
            + [(ptr, self.kv_bytes) for ptr in self._vB]
        )

        self._ds_in_A, self._ds_out_B = self._make_ds(
            self._d0, self._dp, self._sA, self._cA, self._kA, self._vA, self._do, self._sB, self._cB, self._kB, self._vB
        )
        self._ds_in_B, self._ds_out_A = self._make_ds(
            self._d0, self._dp, self._sB, self._cB, self._kB, self._vB, self._do, self._sA, self._cA, self._kA, self._vA
        )

        self._h0 = np.empty(input0_size, np.uint8)
        self._hp = np.empty(8, np.uint8)
        self._ho = np.empty(self.output0_size, np.uint8)
        self._zero_s = np.zeros(S_BYTES, np.uint8)
        self._zero_c = np.zeros(C_BYTES, np.uint8)
        self._zero_kv = np.zeros(self.kv_bytes, np.uint8)
        self._step = 0

    def _check(self, ret: int, msg: str) -> None:
        if ret != 0:
            raise OmEngineError(f"{msg} failed, ret={ret}")

    def _alloc_ptr(self, size: int) -> int:
        ptr, ret = self.acl.rt.malloc(size, M)
        self._check(ret, "malloc")
        self._alloc.append(ptr)
        return ptr

    def _add_buffer(self, dataset, ptr: int, size: int, tag: str):
        buf = self.acl.create_data_buffer(ptr, size)
        _, ret = self.acl.mdl.add_dataset_buffer(dataset, buf)
        self._check(ret, tag)
        return buf

    def _alloc_set(self):
        return (
            [self._alloc_ptr(S_BYTES) for _ in range(NL_DN)],
            [self._alloc_ptr(C_BYTES) for _ in range(NL_DN)],
            [self._alloc_ptr(self.kv_bytes) for _ in range(NL_GA)],
            [self._alloc_ptr(self.kv_bytes) for _ in range(NL_GA)],
        )

    def _make_ds(self, d0, dp, s_src, c_src, k_src, v_src, dout, s_dst, c_dst, k_dst, v_dst):
        ds_in = self.acl.mdl.create_dataset()
        ds_out = self.acl.mdl.create_dataset()
        self._add_buffer(ds_in, d0, self.input0_size, f"{self.input0_name} add_in")
        self._add_buffer(ds_in, dp, 8, "position add_in")
        for ptr in s_src:
            self._add_buffer(ds_in, ptr, S_BYTES, "s add_in")
        for ptr in c_src:
            self._add_buffer(ds_in, ptr, C_BYTES, "c add_in")
        for ptr in k_src:
            self._add_buffer(ds_in, ptr, self.kv_bytes, "k add_in")
        for ptr in v_src:
            self._add_buffer(ds_in, ptr, self.kv_bytes, "v add_in")

        self._add_buffer(ds_out, dout, self.output0_size, "main add_out")
        for ptr in s_dst:
            self._add_buffer(ds_out, ptr, S_BYTES, "s add_out")
        for ptr in c_dst:
            self._add_buffer(ds_out, ptr, C_BYTES, "c add_out")
        for ptr in k_dst:
            self._add_buffer(ds_out, ptr, self.kv_bytes, "k add_out")
        for ptr in v_dst:
            self._add_buffer(ds_out, ptr, self.kv_bytes, "v add_out")
        return ds_in, ds_out

    def reset(self) -> None:
        self._step = 0
        for ptr, size in self._cache_ptrs:
            if size == S_BYTES:
                src = self._zero_s
            elif size == C_BYTES:
                src = self._zero_c
            else:
                src = self._zero_kv
            self._check(self.acl.rt.memcpy(ptr, size, src.ctypes.data, size, H2D), "reset memcpy")

    def execute(self, input_bytes: bytes, position: int) -> np.ndarray:
        self._h0[: self.input0_size] = np.frombuffer(input_bytes, dtype=np.uint8, count=self.input0_size)
        self._hp[:8] = np.array([position], np.int64).view(np.uint8)
        self.acl.rt.memcpy(self._d0, self.input0_size, self._h0.ctypes.data, self.input0_size, H2D)
        self.acl.rt.memcpy(self._dp, 8, self._hp.ctypes.data, 8, H2D)

        if self._step % 2 == 0:
            ds_in, ds_out = self._ds_in_A, self._ds_out_B
        else:
            ds_in, ds_out = self._ds_in_B, self._ds_out_A
        self._step += 1

        self._check(self.acl.mdl.execute(self.mid, ds_in, ds_out), "execute")
        self.acl.rt.memcpy(self._ho.ctypes.data, self.output0_size, self._do, self.output0_size, D2H)
        return self._ho.view(np.float16)

    def close(self) -> None:
        for ptr in self._alloc:
            self.acl.rt.free(ptr)
        self.acl.mdl.unload(self.mid)


class OmSplitEngine(SplitEngine):
    def __init__(self, model_id: str, max_len: int, prefix_om: str, suffix_om: str):
        super().__init__(model_id=model_id, max_len=max_len)
        self.prefix_om = prefix_om
        self.suffix_om = suffix_om
        self.acl = None
        self.runtime: _ACLRuntime | None = None
        self.prefix: _ACLSplitSegment | None = None
        self.suffix: _ACLSplitSegment | None = None

    def load(self) -> None:
        try:
            import sys

            sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
            import acl  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OmEngineError(f"failed to import acl: {exc}") from exc
        self.acl = acl
        self.runtime = _ACLRuntime(acl)
        self.runtime.init()
        self.prefix = _ACLSplitSegment(acl, self.prefix_om, self.max_len, 8, HIDDEN_SIZE * 2, "input_ids")
        self.suffix = _ACLSplitSegment(acl, self.suffix_om, self.max_len, HIDDEN_SIZE * 2, None, "hidden_states")

    def close(self) -> None:
        if self.prefix is not None:
            self.prefix.close()
        if self.suffix is not None:
            self.suffix.close()
        self.prefix = None
        self.suffix = None
        if self.runtime is not None:
            self.runtime.close()
        self.runtime = None
        self.acl = None

    def start_session(self) -> None:
        if self.prefix is None or self.suffix is None:
            raise OmEngineError("engine not loaded")
        self.prefix.reset()
        self.suffix.reset()

    def end_session(self) -> None:
        return

    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        if self.prefix is None:
            raise OmEngineError("engine not loaded")
        input_bytes = np.array([token_id], np.int64).view(np.uint8).tobytes()
        return self.prefix.execute(input_bytes, position).reshape(1, 1, HIDDEN_SIZE)

    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        if self.suffix is None:
            raise OmEngineError("engine not loaded")
        logits = self.suffix.execute(hidden_state.astype(np.float16, copy=False).tobytes(), position)
        return logits.reshape(1, 1, -1)
