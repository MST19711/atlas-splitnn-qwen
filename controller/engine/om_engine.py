from __future__ import annotations

import gc
import threading
from pathlib import Path

import numpy as np

from controller.engine.base import SplitEngine
from scripts.qwen35_model_spec import (
    BoundEmbedHeadConfig,
    ModelSpec,
    SplitConfig,
    load_bound_embed_head_metadata,
)

M, H2D, D2H = 0, 1, 2


class OmEngineError(RuntimeError):
    pass


class _ACLHeadMatmulExecutor:
    def __init__(self, acl, runtime: "_ACLRuntime", op_model_dir: str, model_spec: ModelSpec):
        self.acl = acl
        self.runtime = runtime
        self.op_model_dir = op_model_dir
        self.model_spec = model_spec
        self._lock = threading.Lock()
        self.stream = None
        self.hidden_desc = None
        self.weight_desc = None
        self.logits_desc = None
        self.hidden_buf = None
        self.weight_buf = None
        self.logits_buf = None
        self.attr = None
        self.hidden_dev = None
        self.weight_dev = None
        self.logits_dev = None
        self.hidden_host = np.empty((1, self.model_spec.hidden_size), dtype=np.float16)
        self.logits_host = np.empty((1, self.model_spec.vocab_size), dtype=np.float16)
        self._weight_uploaded = False

    def _check(self, ret: int, msg: str) -> None:
        if ret != 0:
            raise OmEngineError(f"{msg} failed, ret={ret}")

    def load(self, tied_weight: np.memmap) -> None:
        hidden_bytes = self.model_spec.hidden_size * 2
        weight_bytes = self.model_spec.vocab_size * self.model_spec.hidden_size * 2
        logits_bytes = self.model_spec.vocab_size * 2

        self._check(self.acl.op.set_model_dir(self.op_model_dir), "acl.op.set_model_dir")
        self.stream, ret = self.acl.rt.create_stream()
        self._check(ret, "acl.rt.create_stream")

        self.hidden_dev, ret = self.acl.rt.malloc(hidden_bytes, M)
        self._check(ret, "acl.rt.malloc(hidden)")
        self.weight_dev, ret = self.acl.rt.malloc(weight_bytes, M)
        self._check(ret, "acl.rt.malloc(weight)")
        self.logits_dev, ret = self.acl.rt.malloc(logits_bytes, M)
        self._check(ret, "acl.rt.malloc(logits)")

        self._check(
            self.acl.rt.memcpy(self.weight_dev, weight_bytes, tied_weight.ctypes.data, weight_bytes, H2D),
            "acl.rt.memcpy(weight)",
        )
        self._weight_uploaded = True

        self.hidden_desc = self.acl.create_tensor_desc(1, [1, self.model_spec.hidden_size], 2)
        self.weight_desc = self.acl.create_tensor_desc(
            1, [self.model_spec.vocab_size, self.model_spec.hidden_size], 2
        )
        self.logits_desc = self.acl.create_tensor_desc(1, [1, self.model_spec.vocab_size], 2)

        self.hidden_buf = self.acl.create_data_buffer(self.hidden_dev, hidden_bytes)
        self.weight_buf = self.acl.create_data_buffer(self.weight_dev, weight_bytes)
        self.logits_buf = self.acl.create_data_buffer(self.logits_dev, logits_bytes)

        self.attr = self.acl.op.create_attr()
        self._check(self.acl.op.set_attr_bool(self.attr, "transpose_x1", False), "set transpose_x1")
        self._check(self.acl.op.set_attr_bool(self.attr, "transpose_x2", True), "set transpose_x2")

    def run(self, hidden_state: np.ndarray) -> np.ndarray:
        if not self._weight_uploaded:
            raise OmEngineError("ACL head executor not loaded")
        with self._lock:
            self.runtime.bind_thread_context()
            hidden = np.asarray(hidden_state, dtype=np.float16).reshape(1, self.model_spec.hidden_size)
            hidden_bytes = self.model_spec.hidden_size * 2
            logits_bytes = self.model_spec.vocab_size * 2
            self.hidden_host[...] = hidden
            self._check(
                self.acl.rt.memcpy(self.hidden_dev, hidden_bytes, self.hidden_host.ctypes.data, hidden_bytes, H2D),
                "acl.rt.memcpy(hidden)",
            )
            ret = self.acl.op.execute(
                "MatMul",
                [self.hidden_desc, self.weight_desc],
                [self.hidden_buf, self.weight_buf],
                [self.logits_desc],
                [self.logits_buf],
                self.attr,
                self.stream,
            )
            self._check(ret, "acl.op.execute(MatMul)")
            self._check(self.acl.rt.synchronize_stream(self.stream), "acl.rt.synchronize_stream")
            self._check(
                self.acl.rt.memcpy(self.logits_host.ctypes.data, logits_bytes, self.logits_dev, logits_bytes, D2H),
                "acl.rt.memcpy(logits)",
            )
            return self.logits_host.reshape(1, 1, self.model_spec.vocab_size).copy()

    def close(self) -> None:
        if self.attr is not None:
            self.acl.op.destroy_attr(self.attr)
        self.attr = None
        for buf_name in ("hidden_buf", "weight_buf", "logits_buf"):
            buf = getattr(self, buf_name)
            if buf is not None:
                self.acl.destroy_data_buffer(buf)
                setattr(self, buf_name, None)
        for ptr_name in ("hidden_dev", "weight_dev", "logits_dev"):
            ptr = getattr(self, ptr_name)
            if ptr is not None:
                self.acl.rt.free(ptr)
                setattr(self, ptr_name, None)
        for desc_name in ("hidden_desc", "weight_desc", "logits_desc"):
            desc = getattr(self, desc_name)
            if desc is not None:
                self.acl.destroy_tensor_desc(desc)
                setattr(self, desc_name, None)
        if self.stream is not None:
            self.acl.rt.destroy_stream(self.stream)
            self.stream = None


class _BoundEmbedHeadRuntime:
    def __init__(
        self,
        asset_dir: str,
        model_spec: ModelSpec,
        split_config: SplitConfig,
        config: BoundEmbedHeadConfig | None = None,
    ):
        self.asset_dir = Path(asset_dir)
        self.model_spec = model_spec
        self.split_config = split_config
        self.config = config
        self.tied_weight: np.memmap | None = None
        self.final_norm_weight: np.memmap | None = None
        self.head_executor: _ACLHeadMatmulExecutor | None = None

    def load(self) -> None:
        if self.config is None:
            _, _, self.config = load_bound_embed_head_metadata(str(self.asset_dir))
        assert self.config is not None
        if self.split_config.prefix_end != 0 or self.split_config.suffix_start != self.split_config.total_layers:
            raise OmEngineError(
                "bound_embed_head mode requires split 0/N/0 "
                f"(got prefix_end={self.split_config.prefix_end}, "
                f"suffix_start={self.split_config.suffix_start}, total={self.split_config.total_layers})"
            )
        if self.config.dtype != "float16":
            raise OmEngineError(f"unsupported bound asset dtype: {self.config.dtype}")

        tied_path = self.asset_dir / self.config.tied_weight_path
        norm_path = self.asset_dir / self.config.final_norm_path
        if not tied_path.exists():
            raise OmEngineError(f"missing tied weight asset: {tied_path}")
        if not norm_path.exists():
            raise OmEngineError(f"missing final norm asset: {norm_path}")

        self.tied_weight = np.memmap(
            tied_path,
            dtype=np.float16,
            mode="r",
            shape=(self.model_spec.vocab_size, self.model_spec.hidden_size),
        )
        self.final_norm_weight = np.memmap(
            norm_path,
            dtype=np.float16,
            mode="r",
            shape=(self.model_spec.hidden_size,),
        )

    def attach_head_executor(self, head_executor: _ACLHeadMatmulExecutor | None) -> None:
        self.head_executor = head_executor

    def close(self) -> None:
        if self.head_executor is not None:
            self.head_executor.close()
        self.head_executor = None
        self.tied_weight = None
        self.final_norm_weight = None

    def start_session(self) -> None:
        return

    def end_session(self) -> None:
        return

    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        del position
        if self.tied_weight is None:
            raise OmEngineError("bound embed/head assets not loaded")
        if token_id < 0 or token_id >= self.model_spec.vocab_size:
            raise OmEngineError(f"token_id out of range: {token_id}")
        hidden = np.asarray(self.tied_weight[token_id], dtype=np.float16)
        return hidden.reshape(1, 1, self.model_spec.hidden_size)

    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        del position
        if self.tied_weight is None:
            raise OmEngineError("bound embed/head assets not loaded")
        # Remote backbone already returns post-final-norm hidden states from Qwen3_5TextModel.forward().
        if self.head_executor is not None:
            return self.head_executor.run(hidden_state)
        hidden = np.asarray(hidden_state, dtype=np.float16).reshape(1, 1, self.model_spec.hidden_size)
        logits = hidden.reshape(1, self.model_spec.hidden_size).astype(
            np.float16, copy=False
        ) @ self.tied_weight.T
        return logits.astype(np.float16, copy=False).reshape(1, 1, self.model_spec.vocab_size)


class _ACLRuntime:
    def __init__(self, acl):
        self.acl = acl
        self._initialized = False
        self._context = None

    def init(self) -> None:
        if self._initialized:
            return
        ret = self.acl.init()
        if ret != 0:
            raise OmEngineError(f"acl.init failed: {ret}")
        ret = self.acl.rt.set_device(0)
        if ret != 0:
            raise OmEngineError(f"acl.rt.set_device failed: {ret}")
        context, ret = self.acl.rt.get_context()
        if ret != 0:
            raise OmEngineError(f"acl.rt.get_context failed: {ret}")
        self._context = context
        self._initialized = True

    def bind_thread_context(self) -> None:
        if not self._initialized or self._context is None:
            raise OmEngineError("ACL runtime is not initialized")
        ret = self.acl.rt.set_context(self._context)
        if ret != 0:
            raise OmEngineError(f"acl.rt.set_context failed: {ret}")

    def close(self) -> None:
        if not self._initialized:
            return
        self._context = None
        self.acl.rt.reset_device(0)
        self.acl.finalize()
        self._initialized = False


class _ACLSplitSegment:
    def __init__(self, acl, model_path: str, max_len: int,
                 input0_size: int, output0_size: int | None, input0_name: str,
                 model_spec: ModelSpec, nl_dn: int, nl_ga: int):
        self.acl = acl
        self.max_len = max_len
        self.model_spec = model_spec
        self.nl_dn = nl_dn
        self.nl_ga = nl_ga
        self._has_pos = nl_ga > 0  # position is only needed for GQA (full_attention) layers
        self.kv_bytes = 1 * model_spec.num_key_value_heads * max_len * model_spec.head_dim * 2
        self.s_bytes = (1 * model_spec.linear_num_value_heads * model_spec.linear_key_head_dim
                        * model_spec.linear_value_head_dim * 2)
        self.c_bytes = (1 * model_spec.conv_dim
                        * (model_spec.linear_conv_kernel_dim - 1) * 2)
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
        self._dp = self._alloc_ptr(8) if self._has_pos else None
        self._do = self._alloc_ptr(self.output0_size)

        self._sA, self._cA, self._kA, self._vA = self._alloc_set()
        self._sB, self._cB, self._kB, self._vB = self._alloc_set()
        self._cache_ptrs = (
            [(ptr, self.s_bytes) for ptr in self._sA]
            + [(ptr, self.c_bytes) for ptr in self._cA]
            + [(ptr, self.kv_bytes) for ptr in self._kA]
            + [(ptr, self.kv_bytes) for ptr in self._vA]
            + [(ptr, self.s_bytes) for ptr in self._sB]
            + [(ptr, self.c_bytes) for ptr in self._cB]
            + [(ptr, self.kv_bytes) for ptr in self._kB]
            + [(ptr, self.kv_bytes) for ptr in self._vB]
        )

        self._ds_in_A, self._ds_out_B = self._make_ds(
            self._d0, self._dp, self._sA, self._cA, self._kA, self._vA,
            self._do, self._sB, self._cB, self._kB, self._vB,
        )
        self._ds_in_B, self._ds_out_A = self._make_ds(
            self._d0, self._dp, self._sB, self._cB, self._kB, self._vB,
            self._do, self._sA, self._cA, self._kA, self._vA,
        )

        self._h0 = np.empty(input0_size, np.uint8)
        self._hp = np.empty(8, np.uint8)
        self._ho = np.empty(self.output0_size, np.uint8)
        self._zero_s = np.zeros(self.s_bytes, np.uint8)
        self._zero_c = np.zeros(self.c_bytes, np.uint8)
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

    def _add_buffer(self, dataset, ptr: int, size: int):
        buf = self.acl.create_data_buffer(ptr, size)
        _, ret = self.acl.mdl.add_dataset_buffer(dataset, buf)
        self._check(ret, "add_dataset_buffer")

    def _alloc_set(self):
        return (
            [self._alloc_ptr(self.s_bytes) for _ in range(self.nl_dn)],
            [self._alloc_ptr(self.c_bytes) for _ in range(self.nl_dn)],
            [self._alloc_ptr(self.kv_bytes) for _ in range(self.nl_ga)],
            [self._alloc_ptr(self.kv_bytes) for _ in range(self.nl_ga)],
        )

    def _make_ds(self, d0, dp, s_src, c_src, k_src, v_src, dout, s_dst, c_dst, k_dst, v_dst):
        ds_in = self.acl.mdl.create_dataset()
        ds_out = self.acl.mdl.create_dataset()
        self._add_buffer(ds_in, d0, self.input0_size)
        if self._has_pos:
            self._add_buffer(ds_in, self._dp, 8)
        for ptr in s_src:
            self._add_buffer(ds_in, ptr, self.s_bytes)
        for ptr in c_src:
            self._add_buffer(ds_in, ptr, self.c_bytes)
        for ptr in k_src:
            self._add_buffer(ds_in, ptr, self.kv_bytes)
        for ptr in v_src:
            self._add_buffer(ds_in, ptr, self.kv_bytes)

        self._add_buffer(ds_out, dout, self.output0_size)
        for ptr in s_dst:
            self._add_buffer(ds_out, ptr, self.s_bytes)
        for ptr in c_dst:
            self._add_buffer(ds_out, ptr, self.c_bytes)
        for ptr in k_dst:
            self._add_buffer(ds_out, ptr, self.kv_bytes)
        for ptr in v_dst:
            self._add_buffer(ds_out, ptr, self.kv_bytes)
        return ds_in, ds_out

    def reset(self) -> None:
        self._step = 0
        for ptr, size in self._cache_ptrs:
            if size == self.s_bytes:
                src = self._zero_s
            elif size == self.c_bytes:
                src = self._zero_c
            else:
                src = self._zero_kv
            self._check(self.acl.rt.memcpy(ptr, size, src.ctypes.data, size, H2D), "reset memcpy")

    def execute(self, input_bytes: bytes, position: int) -> np.ndarray:
        self._h0[: self.input0_size] = np.frombuffer(input_bytes, dtype=np.uint8,
                                                      count=self.input0_size)
        self.acl.rt.memcpy(self._d0, self.input0_size, self._h0.ctypes.data,
                           self.input0_size, H2D)
        if self._has_pos:
            self._hp[:8] = np.array([position], np.int64).view(np.uint8)
            self.acl.rt.memcpy(self._dp, 8, self._hp.ctypes.data, 8, H2D)

        if self._step % 2 == 0:
            ds_in, ds_out = self._ds_in_A, self._ds_out_B
        else:
            ds_in, ds_out = self._ds_in_B, self._ds_out_A
        self._step += 1

        self._check(self.acl.mdl.execute(self.mid, ds_in, ds_out), "execute")
        self.acl.rt.memcpy(self._ho.ctypes.data, self.output0_size, self._do,
                           self.output0_size, D2H)
        return self._ho.view(np.float16)

    def close(self) -> None:
        for ptr in self._alloc:
            self.acl.rt.free(ptr)
        self.acl.mdl.unload(self.mid)


class OmSplitEngine(SplitEngine):
    def __init__(self, model_id: str, max_len: int,
                 model_spec: ModelSpec, split_config: SplitConfig,
                 prefix_om: str | None = None, suffix_om: str | None = None,
                 mode: str = "om_split", bound_asset_dir: str | None = None,
                 bound_config: BoundEmbedHeadConfig | None = None):
        super().__init__(model_id=model_id, max_len=max_len, model_spec=model_spec)
        self.split_config = split_config
        self.prefix_om = prefix_om
        self.suffix_om = suffix_om
        self.mode = mode
        self.bound_asset_dir = bound_asset_dir
        self.bound_config = bound_config
        self.prefix_nl_dn, self.prefix_nl_ga = model_spec.compute_segment(*split_config.prefix_range)
        self.suffix_nl_dn, self.suffix_nl_ga = model_spec.compute_segment(*split_config.suffix_range)
        self.acl = None
        self.runtime: _ACLRuntime | None = None
        self.prefix: _ACLSplitSegment | None = None
        self.suffix: _ACLSplitSegment | None = None
        self.bound_runtime: _BoundEmbedHeadRuntime | None = None
        self.bound_head_executor: _ACLHeadMatmulExecutor | None = None
        self._load_lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            self._load_impl()
            self._loaded = True

    def _load_impl(self) -> None:
        if self.mode == "bound_embed_head":
            if not self.bound_asset_dir:
                raise OmEngineError("bound_embed_head mode requires bound_asset_dir")
            self.bound_runtime = _BoundEmbedHeadRuntime(
                asset_dir=self.bound_asset_dir,
                model_spec=self.model_spec,
                split_config=self.split_config,
                config=self.bound_config,
            )
            self.bound_runtime.load()
            head_op_model_dir = Path(self.bound_asset_dir) / "op_models"
            if head_op_model_dir.is_dir() and any(head_op_model_dir.glob("*.om")):
                try:
                    import sys
                    sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
                    import acl  # type: ignore
                except Exception:
                    acl = None
                if acl is not None:
                    gc.collect()
                    self.acl = acl
                    self.runtime = _ACLRuntime(acl)
                    self.runtime.init()
                    self.bound_head_executor = _ACLHeadMatmulExecutor(
                        acl=acl,
                        runtime=self.runtime,
                        op_model_dir=str(head_op_model_dir),
                        model_spec=self.model_spec,
                    )
                    assert self.bound_runtime.tied_weight is not None
                    self.bound_head_executor.load(self.bound_runtime.tied_weight)
                    self.bound_runtime.attach_head_executor(self.bound_head_executor)
            return
        if self.mode != "om_split":
            raise OmEngineError(f"unsupported om engine mode: {self.mode}")
        if not self.prefix_om or not self.suffix_om:
            raise OmEngineError("om_split mode requires prefix_om and suffix_om")
        try:
            import sys
            sys.path.insert(0, "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
            import acl  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise OmEngineError(f"failed to import acl: {exc}") from exc
        gc.collect()
        self.acl = acl
        self.runtime = _ACLRuntime(acl)
        self.runtime.init()
        self.prefix = _ACLSplitSegment(
            acl, self.prefix_om, self.max_len,
            8, self.model_spec.hidden_size * 2, "input_ids",
            self.model_spec, self.prefix_nl_dn, self.prefix_nl_ga,
        )
        self.suffix = _ACLSplitSegment(
            acl, self.suffix_om, self.max_len,
            self.model_spec.hidden_size * 2, None, "hidden_states",
            self.model_spec, self.suffix_nl_dn, self.suffix_nl_ga,
        )

    def close(self) -> None:
        if self.bound_runtime is not None:
            self.bound_runtime.close()
        self.bound_runtime = None
        self.bound_head_executor = None
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
        self._loaded = False

    def start_session(self) -> None:
        self.load()
        if self.mode == "bound_embed_head":
            assert self.bound_runtime is not None
            self.bound_runtime.start_session()
            return
        if self.prefix is None or self.suffix is None:
            raise OmEngineError("engine not loaded")
        assert self.runtime is not None
        self.runtime.bind_thread_context()
        self.prefix.reset()
        self.suffix.reset()

    def end_session(self) -> None:
        if self.mode == "bound_embed_head" and self.bound_runtime is not None:
            self.bound_runtime.end_session()
        return

    def run_prefix(self, token_id: int, position: int) -> np.ndarray:
        if self.mode == "bound_embed_head":
            if self.bound_runtime is None:
                raise OmEngineError("bound embed/head runtime not loaded")
            return self.bound_runtime.run_prefix(token_id, position)
        if self.prefix is None:
            raise OmEngineError("engine not loaded")
        assert self.runtime is not None
        self.runtime.bind_thread_context()
        input_bytes = np.array([token_id], np.int64).view(np.uint8).tobytes()
        return self.prefix.execute(input_bytes, position).reshape(1, 1, self.model_spec.hidden_size)

    def run_suffix(self, hidden_state: np.ndarray, position: int) -> np.ndarray:
        if self.mode == "bound_embed_head":
            if self.bound_runtime is None:
                raise OmEngineError("bound embed/head runtime not loaded")
            return self.bound_runtime.run_suffix(hidden_state, position)
        if self.suffix is None:
            raise OmEngineError("engine not loaded")
        assert self.runtime is not None
        self.runtime.bind_thread_context()
        logits = self.suffix.execute(hidden_state.astype(np.float16, copy=False).tobytes(), position)
        return logits.reshape(1, 1, -1)
