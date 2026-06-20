#!/usr/bin/env python3
"""Export a pure attention segment of Qwen3.5 as ONNX (no embedding, no norm, no lm head).

Used in bound_embed_head mode when the board needs to run attention layers
between the embed lookup and/or before the final norm+head.
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from qwen35_model_spec import ModelSpec, SplitConfig, parse_split
from qwen35_split_common import (
    HiddenSegmentWrapper,
    apply_qwen35_patches,
    build_segment_io_names,
    configure_eager_attention,
    export_metadata,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--split", type=parse_split, required=True)
    parser.add_argument("--segment", choices=["prefix", "suffix", "middle"], required=True,
                        help="which segment to export")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prefix_end, suffix_start = args.split

    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    model_spec = ModelSpec.from_pretrained(args.model_path)
    split_config = SplitConfig(
        prefix_end=prefix_end, suffix_start=suffix_start,
        total_layers=model_spec.num_hidden_layers,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    configure_eager_attention(model.model, args.max_len)
    apply_qwen35_patches()

    wrapper = HiddenSegmentWrapper(
        model.model, model_spec, split_config, args.max_len, segment=args.segment,
    ).eval()

    nl_dn, nl_ga = wrapper.nl_dn, wrapper.nl_ga
    conv_ks = model_spec.linear_conv_kernel_dim

    input_names, output_names = build_segment_io_names(
        "hidden_states", "hidden_states", nl_dn, nl_ga,
    )

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

    dummy_hidden = torch.zeros((1, 1, model_spec.hidden_size), dtype=torch.float16)
    dummy_pos = torch.tensor([0], dtype=torch.int64)
    with torch.no_grad():
        pt = wrapper(dummy_hidden, dummy_pos, *cache)
        print(f"PT output shape: {pt[0].shape}, [{pt[0].min():.4f},{pt[0].max():.4f}]")

    torch.onnx.export(
        wrapper,
        (dummy_hidden, dummy_pos, *cache),
        args.output,
        input_names=input_names,
        output_names=output_names,
        opset_version=15,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )

    import onnx
    onnx.checker.check_model(args.output)
    print(f"ONNX exported: {args.output} ({os.path.getsize(args.output)/1024/1024:.1f} MB)")

    export_metadata(
        out_dir=os.path.dirname(args.output) or ".",
        onnx_path=args.output,
        model_spec=model_spec,
        split_config=split_config,
        segment=args.segment,
        nl_dn=nl_dn,
        nl_ga=nl_ga,
    )


if __name__ == "__main__":
    main()
