#!/usr/bin/env python3
"""
Export Qwen3-0.6B with torch.export / dynamo-based ONNX export.
Avoids dynamic shape ops that ATC can't handle.
"""

import argparse
import torch
import numpy as np

from transformers import AutoModelForCausalLM


class StaticForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--output", default="qwen3_fp16_seq32_v2.onnx")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--validate", action="store_true", default=True)
    args = parser.parse_args()

    N = args.seq_len

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    wrapper = StaticForwardWrapper(model)

    dummy_input_ids = torch.ones((1, N), dtype=torch.int64)
    dummy_attention_mask = torch.ones((1, N), dtype=torch.int64)

    print(f"Exporting with dynamo to {args.output} (seq_len={N})...")

    # Use TorchScript-based export (works with CANN ATC)
    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        args.output,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=15,
        do_constant_folding=True,
        dynamo=False,
        verbose=False,
    )

    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Export complete: {args.output} ({size_mb:.1f} MB)")

    if args.validate:
        print("Validating with ONNX Runtime...")
        import onnx
        import onnxruntime as ort

        onnx.checker.check_model(args.output)
        print("  ONNX checker: PASS")

        session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
        input_feed = {
            "input_ids": np.ones((1, N), dtype=np.int64),
            "attention_mask": np.ones((1, N), dtype=np.int64),
        }
        logits = session.run(None, input_feed)[0]
        print(f"  shape={logits.shape}, range=[{logits.min():.4f}, {logits.max():.4f}]")
        assert logits.shape == (1, N, 151936)
        assert np.isfinite(logits).all()
        print("  Sanity: PASS")

        # Left-padding test
        attn = np.zeros((1, N), dtype=np.int64)
        attn[0, -5:] = 1
        ids2 = np.ones((1, N), dtype=np.int64)
        ids2[0, :N-5] = 0
        logits2 = session.run(None, {"input_ids": ids2, "attention_mask": attn})[0]
        np.testing.assert_allclose(
            logits[0, -5:, :], logits2[0, -5:, :], rtol=0.01, atol=0.05)
        print("  Left-padding: PASS")

    print("Done.")


if __name__ == "__main__":
    main()
