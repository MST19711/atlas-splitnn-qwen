from __future__ import annotations

from dataclasses import dataclass

from controller.cache.registry import PrefixCacheRegistry
from controller.engine.om_engine import OmSplitEngine
from controller.modeling.base import Qwen35Model
from controller.modeling.kvcache_qwen35 import Qwen35KvCacheModel
from controller.modeling.splitnn_qwen35 import SplitNNQwen35Model
from controller.remote_middle import RemoteMiddleClient
from scripts.qwen35_model_spec import ModelSpec, SplitConfig, load_bound_embed_head_metadata, load_metadata, parse_split


@dataclass
class BackendConfig:
    backend: str
    model_name: str
    remote_model_name: str
    tokenizer_dir: str
    max_len: int
    server_url: str
    connect_timeout: float
    read_timeout: float
    checksum: bool
    model_path: str
    split: tuple[int, int]
    prefix_onnx: str | None
    suffix_onnx: str | None
    prefix_om: str | None
    suffix_om: str | None
    bound_asset_dir: str
    model_om: str
    cache_disabled: bool = False
    cache_max_entries: int = 8
    cache_ttl_sec: int = 300
    cache_min_prefix_len: int = 8


def make_model_name(model_spec: ModelSpec, split: tuple[int, int], backend: str) -> str:
    size_map = {1024: "0.8B", 2048: "2B", 2560: "4B", 4096: "9B", 5120: "27B"}
    prefix_ct = split[0]
    suffix_ct = model_spec.num_hidden_layers - split[1]
    middle_ct = split[1] - split[0]
    size_str = size_map.get(model_spec.hidden_size, f"{model_spec.hidden_size}")
    if backend == "qwen35_kvcache_om":
        return f"qwen3.5-{size_str}-kvcache-om"
    return f"qwen3.5-{size_str}-split-{prefix_ct}-{middle_ct}-{suffix_ct}-{backend}"


def _load_split_model_spec(config: BackendConfig):
    if config.backend == "splitnn_bound_embed_head":
        model_spec, split_config, bound_config = load_bound_embed_head_metadata(config.bound_asset_dir)
        return model_spec, split_config, bound_config
    try:
        model_spec = ModelSpec.from_pretrained(config.model_path)
    except Exception:
        if config.prefix_om:
            model_spec, split_config, _, _ = load_metadata(config.prefix_om.replace(".om", ".metadata.json"))
            return model_spec, split_config, None
        raise
    split_config = SplitConfig(config.split[0], config.split[1], model_spec.num_hidden_layers)
    return model_spec, split_config, None


def _make_cache_registry(config: BackendConfig, middle_client=None) -> PrefixCacheRegistry | None:
    if config.cache_disabled:
        return None
    return PrefixCacheRegistry(
        max_entries=config.cache_max_entries,
        ttl_sec=config.cache_ttl_sec,
        min_prefix_len=config.cache_min_prefix_len,
        tag=config.backend,
        middle_client=middle_client,
    )


def create_model(config: BackendConfig) -> Qwen35Model:
    if config.backend == "qwen35_kvcache_om":
        model_spec = ModelSpec.from_pretrained(config.model_path)
        model_name = config.model_name or make_model_name(
            model_spec,
            (0, model_spec.num_hidden_layers),
            config.backend,
        )
        model = Qwen35KvCacheModel(
            model_name=model_name,
            model_path=config.model_om,
            model_spec=model_spec,
            max_len=config.max_len,
        )
        model.cache_registry = _make_cache_registry(config)
        return model

    model_spec, split_config, bound_config = _load_split_model_spec(config)
    effective_split = (split_config.prefix_end, split_config.suffix_start)
    model_name = config.model_name or make_model_name(model_spec, effective_split, config.backend)

    if config.backend == "splitnn_onnx":
        from controller.engine.onnx_engine import OnnxSplitEngine

        if not config.prefix_onnx or not config.suffix_onnx:
            raise ValueError("splitnn_onnx requires --prefix-onnx and --suffix-onnx")
        engine = OnnxSplitEngine(
            model_id=model_name,
            max_len=config.max_len,
            model_spec=model_spec,
            split_config=split_config,
            prefix_onnx=config.prefix_onnx,
            suffix_onnx=config.suffix_onnx,
        )
    else:
        om_mode = "bound_embed_head" if config.backend == "splitnn_bound_embed_head" else "om_split"
        if om_mode == "om_split" and (not config.prefix_om or not config.suffix_om):
            raise ValueError("splitnn_om requires --prefix-om and --suffix-om")
        engine = OmSplitEngine(
            model_id=model_name,
            max_len=config.max_len,
            model_spec=model_spec,
            split_config=split_config,
            prefix_om=config.prefix_om,
            suffix_om=config.suffix_om,
            mode=om_mode,
            bound_asset_dir=config.bound_asset_dir,
            bound_config=bound_config,
        )

    remote = RemoteMiddleClient(
        server_url=config.server_url,
        model_name=config.remote_model_name or model_name,
        hidden_size=model_spec.hidden_size,
        max_len=config.max_len,
        connect_timeout=config.connect_timeout,
        read_timeout=config.read_timeout,
        checksum=config.checksum,
    )
    model = SplitNNQwen35Model(
        model_name=model_name,
        max_len=config.max_len,
        vocab_size=model_spec.vocab_size,
        backend_kind=config.backend,
        engine=engine,
        remote_middle=remote,
    )
    model.cache_registry = _make_cache_registry(config, middle_client=remote)
    return model
