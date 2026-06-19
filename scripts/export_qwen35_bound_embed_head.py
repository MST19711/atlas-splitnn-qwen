#!/usr/bin/env python3
"""Export Qwen3.5 tied embedding/lm_head and final norm weights for board-side bound mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qwen35_model_spec import (
    ModelSpec,
    SplitConfig,
    export_bound_embed_head_metadata,
)


def parse_split(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("split must be 'prefix_end,suffix_start', e.g. '0,24'")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model_dl/Qwen3.5-2B")
    parser.add_argument("--output-dir", default="om_out/qwen3.5_2b_bound_embed_head")
    parser.add_argument("--split", type=parse_split, default=(0, 24),
                        help="bound mode requires 0/N/0, e.g. 0,24 for 2B")
    parser.add_argument("--compile-op", action="store_true",
                        help="compile ACL single-op MatMul via ATC container")
    args = parser.parse_args()

    model_spec = ModelSpec.from_pretrained(args.model_path)
    split_config = SplitConfig(args.split[0], args.split[1], model_spec.num_hidden_layers)
    if split_config.prefix_end != 0 or split_config.suffix_start != model_spec.num_hidden_layers:
        raise ValueError(
            "bound embed/head export requires split 0/N/0 "
            f"(got prefix_end={split_config.prefix_end}, suffix_start={split_config.suffix_start}, "
            f"total_layers={model_spec.num_hidden_layers})"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    tied_weight = model.get_input_embeddings().weight.detach().cpu()
    lm_head_weight = model.get_output_embeddings().weight.detach().cpu()
    if tied_weight.data_ptr() != lm_head_weight.data_ptr():
        raise RuntimeError("model weights are not tied; bound embed/head mode requires tied weights")

    final_norm_weight = model.model.norm.weight.detach().cpu()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tied_path = out_dir / "tied_weight.bin"
    norm_path = out_dir / "final_norm_weight.bin"
    tied_weight.numpy().astype(np.float16).tofile(tied_path)
    final_norm_weight.numpy().astype(np.float16).tofile(norm_path)
    export_bound_embed_head_metadata(model_spec, split_config, str(out_dir))

    # Export ACL single-op MatMul compile config
    op_dir = out_dir / "op_models"
    op_dir.mkdir(parents=True, exist_ok=True)
    acl_op = [
        {
            "op": "MatMul",
            "input_desc": [
                {"format": "ND", "type": "float16", "shape": [1, model_spec.hidden_size]},
                {"format": "ND", "type": "float16", "shape": [model_spec.vocab_size, model_spec.hidden_size]},
            ],
            "output_desc": [
                {"format": "ND", "type": "float16", "shape": [1, model_spec.vocab_size]},
            ],
            "attr": [
                {"name": "transpose_x1", "type": "bool", "value": False},
                {"name": "transpose_x2", "type": "bool", "value": True},
            ],
        }
    ]
    config_dir = out_dir / "op_models_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    acl_op_path = config_dir / "acl_op.json"
    with open(acl_op_path, "w") as f:
        json.dump(acl_op, f, indent=2)
    print(f"exported: {acl_op_path}")

    # Compile if requested
    if args.compile_op:
        import subprocess
        compile_script = Path(__file__).resolve().parent / "compile_head_matmul.sh"
        if not compile_script.exists():
            raise FileNotFoundError(f"compile_head_matmul.sh not found at {compile_script}")
        subprocess.check_call(["bash", str(compile_script), str(out_dir)])
        # Upload compiled OM to op_models dir
        for om_file in sorted(op_dir.glob("*.om")):
            print(f"compiled: {om_file} ({om_file.stat().st_size / 1024 / 1024:.1f} MiB)")

    # Lightweight sanity check.
    sample_id = 1
    hidden = tied_weight[sample_id : sample_id + 1].reshape(1, 1, -1).to(torch.float32)
    variance = hidden.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden / torch.sqrt(variance + model_spec.rms_norm_eps)
    normalized = normalized * (1.0 + final_norm_weight.to(torch.float32).view(1, 1, -1))
    logits = normalized.to(torch.float16) @ tied_weight.transpose(0, 1)
    print(f"exported: {tied_path} ({tied_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"exported: {norm_path} ({norm_path.stat().st_size / 1024:.1f} KiB)")
    print(f"sanity logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
