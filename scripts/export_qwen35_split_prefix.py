#!/usr/bin/env python3
"""Export Qwen3.5 SplitNN prefix segment."""

import argparse
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qwen35_split_common import (
    ModelSpec,
    PrefixWrapper,
    SplitConfig,
    apply_qwen35_patches,
    build_segment_io_names,
    configure_eager_attention,
    export_metadata,
)


def parse_split(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("split must be 'prefix_end,suffix_start', e.g. '4,20'")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    parser.add_argument("--output", default="om_out/qwen3.5_split_prefix_max256.onnx")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--split", type=parse_split, default=(4, 20),
                        help="prefix_end,suffix_start  (e.g. 4,20 for 4/16/4)")
    args = parser.parse_args()

    model_spec = ModelSpec.from_pretrained(args.model_path)
    split_config = SplitConfig(args.split[0], args.split[1], model_spec.num_hidden_layers)
    nl_dn, nl_ga = model_spec.compute_segment(*split_config.prefix_range)

    apply_qwen35_patches()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    ).eval().to(torch.float16)
    configure_eager_attention(model)
    wrapper = PrefixWrapper(model.model, model_spec, split_config, args.max_len).eval()

    inames, onames = build_segment_io_names("input_ids", "hidden_states", nl_dn, nl_ga)

    conv_ks = model_spec.linear_conv_kernel_dim
    cache = []
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, model_spec.linear_num_value_heads,
                                   model_spec.linear_key_head_dim,
                                   model_spec.linear_value_head_dim), dtype=torch.float16))
    for _ in range(nl_dn):
        cache.append(torch.zeros((1, model_spec.conv_dim, conv_ks - 1), dtype=torch.float16))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, model_spec.num_key_value_heads, args.max_len,
                                   model_spec.head_dim), dtype=torch.float16))
    for _ in range(nl_ga):
        cache.append(torch.zeros((1, model_spec.num_key_value_heads, args.max_len,
                                   model_spec.head_dim), dtype=torch.float16))

    input_ids = torch.ones((1, 1), dtype=torch.int64)
    position = torch.tensor([0], dtype=torch.int64)
    with torch.no_grad():
        pt = wrapper(input_ids, position, *cache)
        print(f"PT hidden: {pt[0].shape}, [{pt[0].min():.4f},{pt[0].max():.4f}]")

    torch.onnx.export(
        wrapper,
        (input_ids, position, *cache),
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

    onnx_model = onnx.load(args.output)
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    feed: dict = {"input_ids": np.ones((1, 1), np.int64)}
    onnx_inputs = {i.name for i in onnx_model.graph.input}
    if "position" in onnx_inputs:
        feed["position"] = np.array([0], np.int64)
    for i in range(nl_dn):
        feed[f"s_past_{i}"] = np.zeros((1, model_spec.linear_num_value_heads,
                                          model_spec.linear_key_head_dim,
                                          model_spec.linear_value_head_dim), np.float16)
    for i in range(nl_dn):
        feed[f"c_past_{i}"] = np.zeros((1, model_spec.conv_dim, conv_ks - 1), np.float16)
    for i in range(nl_ga):
        feed[f"k_past_{i}"] = np.zeros((1, model_spec.num_key_value_heads, args.max_len,
                                          model_spec.head_dim), np.float16)
        feed[f"v_past_{i}"] = np.zeros((1, model_spec.num_key_value_heads, args.max_len,
                                          model_spec.head_dim), np.float16)
    ort_out = sess.run(None, feed)
    diff = np.abs(pt[0].numpy().astype(np.float16) - ort_out[0]).max()
    print(f"PT vs ORT hidden: max_diff={diff:.6f}")
    print("PASS" if diff < 0.1 else "FAIL")

    export_metadata(model_spec, split_config, "prefix", args.output)


if __name__ == "__main__":
    main()
