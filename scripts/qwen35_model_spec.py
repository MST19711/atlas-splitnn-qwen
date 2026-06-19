#!/usr/bin/env python3
"""Lightweight model spec and split config — no torch/transformers dependency.

Imported by both the x86 dev side and the Atlas board (aarch64, no PyTorch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── Model Specification ────────────────────────────────────────────────


@dataclass
class ModelSpec:
    """All architecture parameters read from model config.json."""

    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    full_attention_interval: int
    layer_types: list[str] = field(repr=False)
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelSpec":
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tc = config.text_config if hasattr(config, "text_config") else config
            data = {
                "hidden_size": tc.hidden_size,
                "vocab_size": tc.vocab_size,
                "num_hidden_layers": tc.num_hidden_layers,
                "num_attention_heads": tc.num_attention_heads,
                "num_key_value_heads": tc.num_key_value_heads,
                "head_dim": tc.head_dim,
                "intermediate_size": tc.intermediate_size,
                "linear_num_key_heads": tc.linear_num_key_heads,
                "linear_num_value_heads": tc.linear_num_value_heads,
                "linear_key_head_dim": tc.linear_key_head_dim,
                "linear_value_head_dim": tc.linear_value_head_dim,
                "linear_conv_kernel_dim": tc.linear_conv_kernel_dim,
                "full_attention_interval": getattr(tc, "full_attention_interval", 4),
                "rms_norm_eps": getattr(tc, "rms_norm_eps", 1e-6),
            }
        except Exception:
            config_path = Path(model_path) / "config.json"
            with open(config_path) as f:
                raw = json.load(f)
            tc = raw.get("text_config", raw)
            data = {
                "hidden_size": tc["hidden_size"],
                "vocab_size": tc["vocab_size"],
                "num_hidden_layers": tc["num_hidden_layers"],
                "num_attention_heads": tc["num_attention_heads"],
                "num_key_value_heads": tc["num_key_value_heads"],
                "head_dim": tc["head_dim"],
                "intermediate_size": tc["intermediate_size"],
                "linear_num_key_heads": tc["linear_num_key_heads"],
                "linear_num_value_heads": tc["linear_num_value_heads"],
                "linear_key_head_dim": tc["linear_key_head_dim"],
                "linear_value_head_dim": tc["linear_value_head_dim"],
                "linear_conv_kernel_dim": tc["linear_conv_kernel_dim"],
                "full_attention_interval": tc.get("full_attention_interval", 4),
                "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
            }

        layer_types = []
        interval = data["full_attention_interval"]
        for i in range(data["num_hidden_layers"]):
            if (i + 1) % interval == 0:
                layer_types.append("full_attention")
            else:
                layer_types.append("linear_attention")

        return cls(
            hidden_size=data["hidden_size"],
            vocab_size=data["vocab_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            linear_num_key_heads=data["linear_num_key_heads"],
            linear_num_value_heads=data["linear_num_value_heads"],
            linear_key_head_dim=data["linear_key_head_dim"],
            linear_value_head_dim=data["linear_value_head_dim"],
            linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
            full_attention_interval=interval,
            rms_norm_eps=data["rms_norm_eps"],
            layer_types=layer_types,
        )

    @property
    def conv_dim(self) -> int:
        """Dimension of the DeltaNet conv1d input (q+k+v concatenated)."""
        return (
            self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
        )

    def compute_segment(self, start: int, end: int) -> tuple[int, int]:
        """Return (nl_dn, nl_ga) for layer range [start, end)."""
        nl_dn = sum(1 for i in range(start, end) if self.layer_types[i] == "linear_attention")
        nl_ga = sum(1 for i in range(start, end) if self.layer_types[i] == "full_attention")
        return nl_dn, nl_ga

    def to_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "full_attention_interval": self.full_attention_interval,
            "rms_norm_eps": self.rms_norm_eps,
            "conv_dim": self.conv_dim,
            "layer_types": self.layer_types,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(
            hidden_size=d["hidden_size"],
            vocab_size=d["vocab_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            intermediate_size=d["intermediate_size"],
            linear_num_key_heads=d["linear_num_key_heads"],
            linear_num_value_heads=d["linear_num_value_heads"],
            linear_key_head_dim=d["linear_key_head_dim"],
            linear_value_head_dim=d["linear_value_head_dim"],
            linear_conv_kernel_dim=d["linear_conv_kernel_dim"],
            full_attention_interval=d["full_attention_interval"],
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            layer_types=d["layer_types"],
        )


# ── Split Configuration ────────────────────────────────────────────────


@dataclass
class SplitConfig:
    """Defines the layer split boundaries for Prefix / Middle / Suffix."""

    prefix_end: int
    suffix_start: int
    total_layers: int

    @classmethod
    def create(cls, prefix_layers: int, suffix_layers: int, total_layers: int) -> "SplitConfig":
        return cls(
            prefix_end=prefix_layers,
            suffix_start=total_layers - suffix_layers,
            total_layers=total_layers,
        )

    @property
    def prefix_range(self) -> tuple[int, int]:
        return (0, self.prefix_end)

    @property
    def middle_range(self) -> tuple[int, int]:
        return (self.prefix_end, self.suffix_start)

    @property
    def suffix_range(self) -> tuple[int, int]:
        return (self.suffix_start, self.total_layers)

    def to_dict(self) -> dict:
        return {
            "prefix_end": self.prefix_end,
            "suffix_start": self.suffix_start,
            "total_layers": self.total_layers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SplitConfig":
        return cls(
            prefix_end=d["prefix_end"],
            suffix_start=d["suffix_start"],
            total_layers=d["total_layers"],
        )


@dataclass
class BoundEmbedHeadConfig:
    tied_weight_path: str
    final_norm_path: str
    dtype: str = "float16"

    def to_dict(self) -> dict:
        return {
            "tied_weight_path": self.tied_weight_path,
            "final_norm_path": self.final_norm_path,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoundEmbedHeadConfig":
        return cls(
            tied_weight_path=d["tied_weight_path"],
            final_norm_path=d["final_norm_path"],
            dtype=d.get("dtype", "float16"),
        )


# ── Metadata helpers ────────────────────────────────────────────────────


def export_metadata(
    model_spec: ModelSpec,
    split_config: SplitConfig,
    segment: str,
    output_path: str,
) -> None:
    """Write a metadata.json alongside the ONNX model for board-side consumption."""
    start, end = split_config.prefix_range if segment == "prefix" else split_config.suffix_range
    nl_dn, nl_ga = model_spec.compute_segment(start, end)
    meta = {
        "model_spec": model_spec.to_dict(),
        "split_config": split_config.to_dict(),
        "segment": segment,
        "nl_dn": nl_dn,
        "nl_ga": nl_ga,
    }
    meta_path = str(Path(output_path).with_suffix(".metadata.json"))
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata: {meta_path}")


def load_metadata(meta_path: str) -> tuple[ModelSpec, SplitConfig, int, int]:
    """Load metadata.json (no PyTorch needed — safe for board)."""
    with open(meta_path) as f:
        meta = json.load(f)
    model_spec = ModelSpec.from_dict(meta["model_spec"])
    split_config = SplitConfig.from_dict(meta["split_config"])
    return model_spec, split_config, meta["nl_dn"], meta["nl_ga"]


def export_bound_embed_head_metadata(
    model_spec: ModelSpec,
    split_config: SplitConfig,
    output_dir: str,
    tied_weight_name: str = "tied_weight.bin",
    final_norm_name: str = "final_norm_weight.bin",
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_spec": model_spec.to_dict(),
        "split_config": split_config.to_dict(),
        "mode": "bound_embed_head",
        "bound_embed_head": BoundEmbedHeadConfig(
            tied_weight_path=tied_weight_name,
            final_norm_path=final_norm_name,
        ).to_dict(),
    }
    meta_path = out_dir / "bound_embed_head.metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"metadata: {meta_path}")
    return str(meta_path)


def load_bound_embed_head_metadata(
    meta_path_or_dir: str,
) -> tuple[ModelSpec, SplitConfig, BoundEmbedHeadConfig]:
    path = Path(meta_path_or_dir)
    if path.is_dir():
        path = path / "bound_embed_head.metadata.json"
    with open(path) as f:
        meta = json.load(f)
    model_spec = ModelSpec.from_dict(meta["model_spec"])
    split_config = SplitConfig.from_dict(meta["split_config"])
    bound_cfg = BoundEmbedHeadConfig.from_dict(meta["bound_embed_head"])
    return model_spec, split_config, bound_cfg
