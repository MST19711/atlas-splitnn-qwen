#!/usr/bin/env python3
"""Validate exported SplitNN prefix/suffix ONNX against PyTorch and ORT multi-step execution."""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM

from qwen35_split_common import (
    CONV_D,
    CONV_KS,
    HDIM,
    HIDDEN_SIZE,
    KV_H,
    K_DIM,
    K_H,
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


def zero_cache_np(nl_dn: int, nl_ga: int, max_len: int) -> dict[str, np.ndarray]:
    feed = {}
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = np.zeros((1, K_H, K_DIM, V_DIM), dtype=np.float16)
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = np.zeros((1, CONV_D, CONV_KS - 1), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = np.zeros((1, KV_H, max_len, HDIM), dtype=np.float16)
    for i in range(nl_ga):
        feed[f"v_past_{i}"] = np.zeros((1, KV_H, max_len, HDIM), dtype=np.float16)
    return feed


def update_feed(feed: dict[str, np.ndarray], outputs: list[np.ndarray], nl_dn: int, nl_ga: int) -> np.ndarray:
    main = outputs[0]
    idx = 1
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = outputs[idx]
        idx += 1
    for i in range(nl_ga):
        feed[f"v_past_{i}"] = outputs[idx]
        idx += 1
    return main


def split_cache_torch_to_np(cache: list[torch.Tensor], nl_dn: int, nl_ga: int) -> dict[str, np.ndarray]:
    out = {}
    idx = 0
    for i in range(nl_dn):
        out[f"s_past_{i}"] = cache[idx].detach().cpu().numpy()
        idx += 1
    for i in range(nl_dn):
        out[f"c_past_{i}"] = cache[idx].detach().cpu().numpy()
        idx += 1
    for i in range(nl_ga):
        out[f"k_past_{i}"] = cache[idx].detach().cpu().numpy()
        idx += 1
    for i in range(nl_ga):
        out[f"v_past_{i}"] = cache[idx].detach().cpu().numpy()
        idx += 1
    return out


def make_cache_torch(nl_dn: int, nl_ga: int, max_len: int) -> list[torch.Tensor]:
    cache = []
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16))
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, CONV_D, CONV_KS - 1), dtype=torch.float16))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, KV_H, max_len, HDIM), dtype=torch.float16))
    return cache


def update_cache_torch(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    return outputs[0], [x.detach() for x in outputs[1:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--prefix-onnx", default="om_out/qwen3.5_split_prefix_max256.onnx")
    parser.add_argument("--suffix-onnx", default="om_out/qwen3.5_split_suffix_max256.onnx")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--prompt-ids", default="100,200,300")
    parser.add_argument("--decode-steps", type=int, default=5)
    parser.add_argument("--tol-hidden", type=float, default=0.1)
    parser.add_argument("--tol-logits", type=float, default=0.2)
    args = parser.parse_args()

    apply_qwen35_patches()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval().to(torch.float16)
    configure_eager_attention(model)

    prefix_pt = PrefixWrapper(model.model, args.max_len).eval()
    middle_pt = MiddleWrapper(model.model, args.max_len).eval()
    suffix_pt = SuffixWrapper(model.model, args.max_len, model.lm_head).eval()

    prefix_sess = ort.InferenceSession(args.prefix_onnx, providers=["CPUExecutionProvider"])
    suffix_sess = ort.InferenceSession(args.suffix_onnx, providers=["CPUExecutionProvider"])

    prompt_ids = [int(x.strip()) for x in args.prompt_ids.split(",") if x.strip()]
    prefix_cache_pt = make_cache_torch(PREFIX_NL_DN, PREFIX_NL_GA, args.max_len)
    middle_cache_pt = make_cache_torch(MIDDLE_NL_DN, MIDDLE_NL_GA, args.max_len)
    suffix_cache_pt = make_cache_torch(SUFFIX_NL_DN, SUFFIX_NL_GA, args.max_len)
    prefix_feed = zero_cache_np(PREFIX_NL_DN, PREFIX_NL_GA, args.max_len)
    suffix_feed = zero_cache_np(SUFFIX_NL_DN, SUFFIX_NL_GA, args.max_len)

    last_logits = None
    for pos, tid in enumerate(prompt_ids + [None] * args.decode_steps):
        if tid is None:
            tid = int(np.argmax(last_logits[0, 0, :]))
        input_ids = torch.tensor([[tid]], dtype=torch.int64)
        position = torch.tensor([pos], dtype=torch.int64)

        with torch.no_grad():
            prefix_pt_out = prefix_pt(input_ids, position, *prefix_cache_pt)
        hidden_pt, prefix_cache_pt = update_cache_torch(prefix_pt_out)

        prefix_feed["input_ids"] = np.array([[tid]], dtype=np.int64)
        prefix_feed["position"] = np.array([pos], dtype=np.int64)
        prefix_ort_out = prefix_sess.run(None, prefix_feed)
        hidden_ort = update_feed(prefix_feed, prefix_ort_out, PREFIX_NL_DN, PREFIX_NL_GA)
        diff_hidden = float(np.max(np.abs(hidden_pt.cpu().numpy().astype(np.float16) - hidden_ort)))
        print(f"step {pos}: prefix hidden diff={diff_hidden:.6f}")
        if diff_hidden > args.tol_hidden:
            raise AssertionError(f"prefix hidden diff too large at step {pos}: {diff_hidden}")

        hidden_mid = torch.from_numpy(hidden_ort.copy()).to(torch.float16)
        with torch.no_grad():
            middle_out = middle_pt(hidden_mid, position, *middle_cache_pt)
        hidden_l20, middle_cache_pt = update_cache_torch(middle_out)

        with torch.no_grad():
            suffix_pt_out = suffix_pt(hidden_l20, position, *suffix_cache_pt)
        logits_pt, suffix_cache_pt = update_cache_torch(suffix_pt_out)

        suffix_feed["hidden_states"] = hidden_l20.cpu().numpy().astype(np.float16)
        suffix_feed["position"] = np.array([pos], dtype=np.int64)
        suffix_ort_out = suffix_sess.run(None, suffix_feed)
        logits_ort = update_feed(suffix_feed, suffix_ort_out, SUFFIX_NL_DN, SUFFIX_NL_GA)
        diff_logits = float(np.max(np.abs(logits_pt.cpu().numpy().astype(np.float16) - logits_ort)))
        print(f"step {pos}: suffix logits diff={diff_logits:.6f}")
        if diff_logits > args.tol_logits:
            raise AssertionError(f"suffix logits diff too large at step {pos}: {diff_logits}")
        last_logits = logits_ort

    print("Split ORT validation: PASS")


if __name__ == "__main__":
    main()
