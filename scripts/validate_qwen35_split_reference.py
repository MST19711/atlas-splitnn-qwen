#!/usr/bin/env python3
"""Validate Qwen3.5 SplitNN reference chain against full-model execution."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from export_qwen35_kvcache import Qwen35KVCacheWrapper
from qwen35_model_spec import ModelSpec, SplitConfig, parse_split
from qwen35_split_common import (
    MiddleWrapper,
    PrefixWrapper,
    SuffixWrapper,
    apply_qwen35_patches,
    configure_eager_attention,
)


def make_cache(model_spec: ModelSpec, nl_dn: int, nl_ga: int, max_len: int,
               device: torch.device) -> list[torch.Tensor]:
    cache = []
    conv_ks = model_spec.linear_conv_kernel_dim
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, model_spec.linear_num_value_heads,
                                   model_spec.linear_key_head_dim,
                                   model_spec.linear_value_head_dim),
                                 dtype=torch.float16, device=device))
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, model_spec.conv_dim, conv_ks - 1),
                                 dtype=torch.float16, device=device))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, model_spec.num_key_value_heads, max_len,
                                   model_spec.head_dim),
                                 dtype=torch.float16, device=device))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, model_spec.num_key_value_heads, max_len,
                                   model_spec.head_dim),
                                 dtype=torch.float16, device=device))
    return cache


def update_cache(cache: list[torch.Tensor],
                 outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return outputs[0], [x.detach() for x in outputs[1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--max-len", type=int, default=16384)
    parser.add_argument("--split", type=parse_split, default=(4, 20),
                        help="prefix_end,suffix_start")
    parser.add_argument("--prompt-ids", default="100,200,300")
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.2)
    args = parser.parse_args()

    model_spec = ModelSpec.from_pretrained(args.model_path)
    split_config = SplitConfig(args.split[0], args.split[1], model_spec.num_hidden_layers)
    full_nl_dn, full_nl_ga = model_spec.compute_segment(0, model_spec.num_hidden_layers)

    device = torch.device("cpu")
    apply_qwen35_patches()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval().to(torch.float16)
    configure_eager_attention(model)

    full = Qwen35KVCacheWrapper(model.model, args.max_len, model.lm_head).eval()
    prefix = PrefixWrapper(model.model, model_spec, split_config, args.max_len).eval()
    middle = MiddleWrapper(model.model, model_spec, split_config, args.max_len).eval()
    suffix = SuffixWrapper(model.model, model_spec, split_config, args.max_len, model.lm_head).eval()

    full_cache = make_cache(model_spec, full_nl_dn, full_nl_ga, args.max_len, device)
    prefix_cache = make_cache(model_spec, prefix.nl_dn, prefix.nl_ga, args.max_len, device)
    middle_cache = make_cache(model_spec, middle.nl_dn, middle.nl_ga, args.max_len, device)
    suffix_cache = make_cache(model_spec, suffix.nl_dn, suffix.nl_ga, args.max_len, device)

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        raise ValueError("prompt_ids is empty")

    print(f"Model: {args.model_path}, split: {args.split}, layers: {model_spec.num_hidden_layers}")
    print(f"Full: nl_dn={full_nl_dn}, nl_ga={full_nl_ga}")
    print(f"Prefix: nl_dn={prefix.nl_dn}, nl_ga={prefix.nl_ga}")
    print(f"Middle: nl_dn={middle.nl_dn}, nl_ga={middle.nl_ga}")
    print(f"Suffix: nl_dn={suffix.nl_dn}, nl_ga={suffix.nl_ga}")
    print(f"prompt_ids={prompt_ids}, decode_steps={args.decode_steps}")

    full_logits = None
    split_logits = None
    for pos, tid in enumerate(prompt_ids):
        input_ids = torch.tensor([[tid]], dtype=torch.int64, device=device)
        position = torch.tensor([pos], dtype=torch.int64, device=device)

        with torch.no_grad():
            full_out = full(input_ids, position, *full_cache)
            full_logits, full_cache = update_cache(full_cache, full_out)

            prefix_out = prefix(input_ids, position, *prefix_cache)
            hidden_l4, prefix_cache = update_cache(prefix_cache, prefix_out)

            middle_out = middle(hidden_l4, position, *middle_cache)
            hidden_l20, middle_cache = update_cache(middle_cache, middle_out)

            suffix_out = suffix(hidden_l20, position, *suffix_cache)
            split_logits, suffix_cache = update_cache(suffix_cache, suffix_out)

        diff = float(torch.max(torch.abs(full_logits - split_logits)).item())
        print(f"prefill step {pos}: max_diff={diff:.6f}")
        if not np.isfinite(diff) or diff > args.tol:
            raise AssertionError(f"prefill step {pos} diff too large: {diff}")

    current_id = int(torch.argmax(split_logits[0, 0, :]).item())
    for step in range(args.decode_steps):
        pos = len(prompt_ids) + step
        if pos >= args.max_len:
            break
        input_ids = torch.tensor([[current_id]], dtype=torch.int64, device=device)
        position = torch.tensor([pos], dtype=torch.int64, device=device)
        with torch.no_grad():
            full_out = full(input_ids, position, *full_cache)
            full_logits, full_cache = update_cache(full_cache, full_out)

            prefix_out = prefix(input_ids, position, *prefix_cache)
            hidden_l4, prefix_cache = update_cache(prefix_cache, prefix_out)

            middle_out = middle(hidden_l4, position, *middle_cache)
            hidden_l20, middle_cache = update_cache(middle_cache, middle_out)

            suffix_out = suffix(hidden_l20, position, *suffix_cache)
            split_logits, suffix_cache = update_cache(suffix_cache, suffix_out)

        diff = float(torch.max(torch.abs(full_logits - split_logits)).item())
        full_argmax = int(torch.argmax(full_logits[0, 0, :]).item())
        split_argmax = int(torch.argmax(split_logits[0, 0, :]).item())
        print(
            f"decode step {pos}: max_diff={diff:.6f}, full_argmax={full_argmax}, split_argmax={split_argmax}"
        )
        if not np.isfinite(diff) or diff > args.tol:
            raise AssertionError(f"decode step {pos} diff too large: {diff}")
        current_id = split_argmax

    print("Split reference validation: PASS")


if __name__ == "__main__":
    main()
