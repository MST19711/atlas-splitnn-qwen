#!/usr/bin/env python3
"""Validate Qwen3.5 SplitNN reference chain against full-model execution."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from export_qwen35_kvcache import Qwen35KVCacheWrapper
from qwen35_split_common import (
    CONV_D,
    CONV_KS,
    HDIM,
    HIDDEN_SIZE,
    KV_H,
    K_DIM,
    K_H,
    MAX_LEN,
    MIDDLE_NL_DN,
    MIDDLE_NL_GA,
    MiddleWrapper,
    PREFIX_NL_DN,
    PREFIX_NL_GA,
    PrefixWrapper,
    SUFFIX_NL_DN,
    SUFFIX_NL_GA,
    SuffixWrapper,
    V_DIM,
    apply_qwen35_patches,
    configure_eager_attention,
)

FULL_NL_DN = 18
FULL_NL_GA = 6


def make_cache(nl_dn: int, nl_ga: int, max_len: int, device: torch.device) -> list[torch.Tensor]:
    cache = []
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16, device=device))
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, CONV_D, CONV_KS - 1), dtype=torch.float16, device=device))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16, device=device))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16, device=device))
    return cache


def update_cache(cache: list[torch.Tensor], outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return outputs[0], [x.detach() for x in outputs[1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--prompt-ids", default="100,200,300")
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--tol", type=float, default=0.2)
    args = parser.parse_args()

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
    prefix = PrefixWrapper(model.model, args.max_len).eval()
    middle = MiddleWrapper(model.model, args.max_len).eval()
    suffix = SuffixWrapper(model.model, args.max_len, model.lm_head).eval()

    full_cache = make_cache(FULL_NL_DN, FULL_NL_GA, args.max_len, device)
    prefix_cache = make_cache(PREFIX_NL_DN, PREFIX_NL_GA, args.max_len, device)
    middle_cache = make_cache(MIDDLE_NL_DN, MIDDLE_NL_GA, args.max_len, device)
    suffix_cache = make_cache(SUFFIX_NL_DN, SUFFIX_NL_GA, args.max_len, device)

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    if not prompt_ids:
        raise ValueError("prompt_ids is empty")

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
