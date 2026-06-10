#!/usr/bin/env python3
"""
Patch Qwen3 ONNX v2: replace GQA Expand with Tile (repeat along group dim).
Expand(broadcast) and Tile(repeat) are semantically equivalent for the GQA use case.
"""

import argparse, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper


def patch_expand_to_tile(input_path: str, output_path: str, seq_len: int, head_dim: int):
    """Replace Expand nodes in self_attn with Tile nodes."""
    m = onnx.load(input_path)

    # Repeats for Tile: [1, 1, 2, 1, 1] (repeat dim 2 by factor 2)
    repeats = np.array([1, 1, 2, 1, 1], dtype=np.int64)
    patched = 0

    new_nodes = []
    new_inits = list(m.graph.initializer)

    for node in m.graph.node:
        if node.op_type == "Expand" and "self_attn" in (node.name or ""):
            src = node.input[0]  # source tensor
            # Remove old target shape from inputs (might reference dead Where)
            # Change op from Expand to Tile
            tile_repeats_name = f"{node.name}_tile_repeats"
            repeats_tensor = numpy_helper.from_array(repeats, name=tile_repeats_name)
            new_inits.append(repeats_tensor)

            new_node = helper.make_node(
                "Tile",
                inputs=[src, tile_repeats_name],
                outputs=list(node.output),
                name=node.name.replace("Expand", "Tile"),
            )
            new_nodes.append(new_node)
            patched += 1
        else:
            new_nodes.append(node)

    # Replace graph nodes and initializers
    del m.graph.node[:]; m.graph.node.extend(new_nodes)
    del m.graph.initializer[:]; m.graph.initializer.extend(new_inits)

    print(f"Patched {patched} Expand → Tile nodes")
    onnx.save(m, output_path)
    print(f"Saved: {output_path}")
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    out = args.output or args.input.replace(".onnx", "_tile.onnx")
    m = patch_expand_to_tile(args.input, out, args.seq_len, args.head_dim)

    # Validate
    print("\nValidating...")
    onnx.checker.check_model(out)
    print("  ONNX checker: PASS")

    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    N = args.seq_len
    logits = sess.run(None, {
        "input_ids": np.ones((1, N), dtype=np.int64),
        "attention_mask": np.ones((1, N), dtype=np.int64),
    })[0]
    print(f"  shape={logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
    assert logits.shape == (1, N, 151936)

    # Left-padding
    attn = np.zeros((1, N), dtype=np.int64); attn[0, -5:] = 1
    ids2 = np.ones((1, N), dtype=np.int64); ids2[0, :N-5] = 0
    logits2 = sess.run(None, {"input_ids": ids2, "attention_mask": attn})[0]
    np.testing.assert_allclose(logits[0, -5:, :], logits2[0, -5:, :], rtol=0.01, atol=0.05)
    print("  Left-padding: PASS")

    import os
    print(f"  Size: {os.path.getsize(out)/1024/1024:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
