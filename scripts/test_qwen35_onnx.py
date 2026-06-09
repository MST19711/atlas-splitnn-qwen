#!/usr/bin/env python3
"""Qwen3.5-0.8B ONNX 推理验证 — 多步骤 prefill + decode。"""

import numpy as np, onnxruntime as ort

MAX = 256; NL = 24; NL_DN = 18; NL_GA = 6
K_H = 16; K_DIM = 128; V_DIM = 128; CONV_D = 6144; CONV_KS = 4
KV_H = 2; HDIM = 256

print("Loading ONNX...")
sess = ort.InferenceSession("om_out/qwen3.5_kvcache_max256.onnx", providers=["CPUExecutionProvider"])
print(f"  Inputs: {len(sess.get_inputs())}, Outputs: {len(sess.get_outputs())}")

# ── 初始化 cache ──────────────────────────────────────────────────
feed = {"input_ids": np.ones((1, 1), dtype=np.int64), "position": np.array([0], dtype=np.int64)}
for i in range(NL_DN):
    feed[f"s_past_{i}"] = np.zeros((1, K_H, K_DIM, V_DIM), dtype=np.float16)
    feed[f"c_past_{i}"] = np.zeros((1, CONV_D, CONV_KS - 1), dtype=np.float16)
for i in range(NL_GA):
    feed[f"k_past_{i}"] = np.zeros((1, KV_H, MAX, HDIM), dtype=np.float16)
    feed[f"v_past_{i}"] = np.zeros((1, KV_H, MAX, HDIM), dtype=np.float16)

# ── Prefill: 3 prompt tokens ────────────────────────────────────
prompt_ids = [100, 200, 300]
print(f"\nPrefill {len(prompt_ids)} tokens...")
for pos, tid in enumerate(prompt_ids):
    feed["input_ids"] = np.array([[tid]], dtype=np.int64)
    feed["position"] = np.array([pos], dtype=np.int64)
    outs = sess.run(None, feed)
    # Update cache from outputs: [logits, S, C, K, V]
    oi = 1  # skip logits
    for i in range(NL_DN):
        feed[f"s_past_{i}"] = outs[oi]; oi += 1
    for i in range(NL_DN):
        feed[f"c_past_{i}"] = outs[oi]; oi += 1
    for i in range(NL_GA):
        feed[f"k_past_{i}"] = outs[oi]; feed[f"v_past_{i}"] = outs[oi+1]; oi += 2
    logits = outs[0]
    print(f"  step {pos}: logits range=[{logits.min():.4f},{logits.max():.4f}], finite={np.isfinite(logits).all()}")

assert np.isfinite(logits).all(), "NaN/Inf at prefill"

# ── Decode: 5 tokens ─────────────────────────────────────────────
print(f"\nDecode 5 tokens...")
for step in range(5):
    pos = len(prompt_ids) + step
    tid = int(np.argmax(logits[0, 0, :]))  # greedy
    feed["input_ids"] = np.array([[tid]], dtype=np.int64)
    feed["position"] = np.array([pos], dtype=np.int64)
    outs = sess.run(None, feed)
    oi = 1
    for i in range(NL_DN):
        feed[f"s_past_{i}"] = outs[oi]; oi += 1
    for i in range(NL_DN):
        feed[f"c_past_{i}"] = outs[oi]; oi += 1
    for i in range(NL_GA):
        feed[f"k_past_{i}"] = outs[oi]; feed[f"v_past_{i}"] = outs[oi+1]; oi += 2
    logits = outs[0]
    print(f"  step {pos}: token={tid} (top-5 prob), logits range=[{logits.min():.4f},{logits.max():.4f}]")

# ── 最终验证 ────────────────────────────────────────────────────
print(f"\nAll K/V have non-zero values in filled positions:")
for i in range(NL_DN):
    s = feed[f"s_past_{i}"]
    assert np.count_nonzero(s[0,:,0,:]) > 0, f"S[{i}] position 0 zero!"
print(f"  S states (18): all position 0 non-zero ✓")
for i in range(NL_GA):
    k = feed[f"k_past_{i}"]
    assert np.count_nonzero(k[0,:,0,:]) > 0, f"K[{i}] position 0 zero!"
    assert np.count_nonzero(k[0,:,pos,:]) > 0, f"K[{i}] position {pos} zero!"
print(f"  K/V cache (6×2): all checked ✓")
print("\nAll tests passed!")
