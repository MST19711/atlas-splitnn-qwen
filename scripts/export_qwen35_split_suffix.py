#!/usr/bin/env python3
"""Export Qwen3.5 SplitNN suffix segment (layers 20:24 + norm + lm_head)."""

import argparse
import os

import numpy as np
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
    SUFFIX_NL_DN,
    SUFFIX_NL_GA,
    SuffixWrapper,
    V_DIM,
    apply_qwen35_patches,
    build_segment_io_names,
    configure_eager_attention,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--output", default="om_out/qwen3.5_split_suffix_max256.onnx")
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    apply_qwen35_patches()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval().to(torch.float16)
    configure_eager_attention(model)
    wrapper = SuffixWrapper(model.model, args.max_len, model.lm_head).eval()

    inames, onames = build_segment_io_names("hidden_states", "logits", SUFFIX_NL_DN, SUFFIX_NL_GA)
    cache = []
    for _ in range(SUFFIX_NL_DN):
        cache.append(torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16))
    for _ in range(SUFFIX_NL_DN):
        cache.append(torch.zeros((1, CONV_D, CONV_KS - 1), dtype=torch.float16))
    for _ in range(SUFFIX_NL_GA):
        cache.append(torch.zeros((1, KV_H, args.max_len, HDIM), dtype=torch.float16))
    for _ in range(SUFFIX_NL_GA):
        cache.append(torch.zeros((1, KV_H, args.max_len, HDIM), dtype=torch.float16))

    hidden_states = torch.zeros((1, 1, HIDDEN_SIZE), dtype=torch.float16)
    position = torch.tensor([0], dtype=torch.int64)
    with torch.no_grad():
        pt = wrapper(hidden_states, position, *cache)
        print(f"PT logits: {pt[0].shape}, [{pt[0].min():.4f},{pt[0].max():.4f}]")

    torch.onnx.export(
        wrapper,
        (hidden_states, position, *cache),
        args.output,
        input_names=inames,
        output_names=onames,
        opset_version=15,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )
    print(f"ONNX: {os.path.getsize(args.output)/1024/1024:.1f} MB")

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(args.output)
    print("checker: PASS")
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    feed = {"hidden_states": np.zeros((1, 1, HIDDEN_SIZE), np.float16), "position": np.array([0], np.int64)}
    for i in range(SUFFIX_NL_DN):
        feed[f"s_past_{i}"] = np.zeros((1, K_H, K_DIM, V_DIM), np.float16)
    for i in range(SUFFIX_NL_DN):
        feed[f"c_past_{i}"] = np.zeros((1, CONV_D, CONV_KS - 1), np.float16)
    for i in range(SUFFIX_NL_GA):
        feed[f"k_past_{i}"] = np.zeros((1, KV_H, args.max_len, HDIM), np.float16)
        feed[f"v_past_{i}"] = np.zeros((1, KV_H, args.max_len, HDIM), np.float16)
    ort_out = sess.run(None, feed)
    diff = np.abs(pt[0].numpy().astype(np.float16) - ort_out[0]).max()
    print(f"PT vs ORT logits: max_diff={diff:.6f}")
    print("PASS" if diff < 0.1 else "FAIL")


if __name__ == "__main__":
    main()
