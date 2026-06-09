#!/usr/bin/env python3
"""Qwen3.5-0.8B KV Cache 导出 — exact DeltaNet implementation with conv_state."""

import argparse, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5Attention, Qwen3_5DecoderLayer,
)

MAX_LEN = 256; NL = 24; NL_DN = 18; NL_GA = 6
VOCAB = 248320; K_H = 16; K_DIM = 128; V_DIM = 128
KV_H = 2; HDIM = 256; CONV_D = 6144; CONV_KS = 4


def make_attn_mask(max_len, position):
    idx = torch.arange(max_len, dtype=torch.int64)
    m = idx.unsqueeze(0).unsqueeze(0) > position
    bias = torch.full((1, 1, 1, max_len), float("-inf"), dtype=torch.float16)
    return bias.masked_fill(~m.unsqueeze(2), 0.0)


# ── DeltaNet 单步 ─────────────────────────────────────────────────
def delta_step(query, key, value, g, beta, S):
    """seq=1 DeltaNet 状态更新。l2norm + scaling + 迭代状态。"""
    eps = 1e-6
    q = query / (query.pow(2).sum(dim=-1,keepdim=True).sqrt() + eps)
    k = key / (key.pow(2).sum(dim=-1,keepdim=True).sqrt() + eps)
    q = q * (query.shape[-1] ** -0.5)

    g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
    beta_t = beta.unsqueeze(-1)
    S_new = S * g_t
    kv_mem = (S_new * k.unsqueeze(-1)).sum(dim=-2)
    delta = (value - kv_mem) * beta_t
    S_new = S_new + k.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S_new * q.unsqueeze(-1)).sum(dim=-2)
    return out, S_new


# ── Patched DeltaNet forward (接受 conv_state) ─────────────────────
def _patched_dn_fwd(self, hidden_states, attention_mask=None,
                    past_S=None, past_conv=None, position=None, **kw):
    B, T, _ = hidden_states.shape
    assert T == 1
    mqkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # CausalConv1D with state
    mqkv_t = mqkv.transpose(1, 2)           # (1,D,1)
    ks = self.conv1d.kernel_size[0]
    cs = past_conv if past_conv is not None else torch.zeros(1, mqkv.shape[-1], ks-1, dtype=mqkv.dtype)
    inp = torch.cat([cs, mqkv_t], dim=-1)
    conv = F.conv1d(inp, self.conv1d.weight, self.conv1d.bias, groups=mqkv.shape[-1])
    mqkv = F.silu(conv).transpose(1, 2)
    new_conv = inp[:, :, -ks+1:].to(hidden_states.dtype)

    q, k, v = torch.split(mqkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(B, T, -1, self.head_k_dim).transpose(1, 2)
    k = k.reshape(B, T, -1, self.head_k_dim).transpose(1, 2)
    v = v.reshape(B, T, -1, self.head_v_dim).transpose(1, 2)

    beta = b.sigmoid().transpose(1, 2)
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    g = g.transpose(1, 2)

    S = past_S if past_S is not None else torch.zeros(1, self.num_k_heads, self.head_k_dim, self.head_v_dim, dtype=v.dtype)
    out, S_new = delta_step(q[:,:,0], k[:,:,0], v[:,:,0], g[:,:,0], beta[:,:,0], S)
    out = out.to(hidden_states.dtype); S_new = S_new.to(hidden_states.dtype)

    out = self.norm(out.reshape(-1, self.head_v_dim), z[:, 0].reshape(-1, self.head_v_dim))
    out = out.reshape(B, T, -1)
    return self.out_proj(out), S_new, new_conv


# ── Attention Cache ───────────────────────────────────────────────
class _AttentionCacheWrapper:
    def __init__(self, k_buf, v_buf, position):
        self._k = k_buf; self._v = v_buf; self._pos = position
        self.present_k = k_buf; self.present_v = v_buf

    def update(self, key_states, value_states, layer_idx):
        L = self._k.shape[2]
        idx = torch.arange(L, dtype=torch.int64, device=self._k.device)
        mask = idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) == self._pos.view(1,1,1,1)
        new_k = torch.where(mask, key_states, self._k)
        new_v = torch.where(mask, value_states, self._v)
        self.present_k = new_k; self.present_v = new_v
        return new_k, new_v


# ── Wrapper (包含 conv_state) ─────────────────────────────────────
class Qwen35KVCacheWrapper(nn.Module):
    def __init__(self, text_model, max_len, lm_head):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.max_len = max_len

    def forward(self, input_ids, position, *cache_flat):
        attn_mask = make_attn_mask(self.max_len, position)
        hidden = self.model.embed_tokens(input_ids)
        pos_ids = position.unsqueeze(0)
        pos_emb = self.model.rotary_emb(hidden, pos_ids)

        # Parse: [S0..S17] [conv0..conv17] [K0..K5] [V0..K5]
        dn_states  = list(cache_flat[:NL_DN])
        conv_states= list(cache_flat[NL_DN:NL_DN+NL_DN])
        k_states   = list(cache_flat[NL_DN*2:NL_DN*2+NL_GA])
        v_states   = list(cache_flat[NL_DN*2+NL_GA:])

        pres_S = [None]*NL_DN; pres_C = [None]*NL_DN
        pres_K = [None]*NL_GA; pres_V = [None]*NL_GA
        di, gi = 0, 0

        for i, layer in enumerate(self.model.layers):
            res = hidden
            hidden = layer.input_layernorm(hidden)

            if layer.layer_type == 'linear_attention':
                hs, S_new, C_new = layer.linear_attn(
                    hidden, past_S=dn_states[di], past_conv=conv_states[di], position=position)
                dn_states[di] = S_new
                conv_states[di] = C_new
                pres_S[di] = S_new; pres_C[di] = C_new
                di += 1
            else:
                ca = _AttentionCacheWrapper(k_states[gi], v_states[gi], position)
                hs, _ = layer.self_attn(
                    hidden, attention_mask=attn_mask,
                    position_embeddings=pos_emb, position_ids=pos_ids,
                    past_key_values=ca, use_cache=True,
                )
                pres_K[gi] = ca.present_k; pres_V[gi] = ca.present_v
                gi += 1

            hidden = res + hs
            res = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = res + hidden

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *pres_S, *pres_C, *pres_K, *pres_V)


# ── 导出 ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="model/Qwen3.5-0.8B")
    p.add_argument("--output", default="om_out/qwen3.5_kvcache_max256.onnx")
    p.add_argument("--max-len", type=int, default=256)
    args = p.parse_args()
    N = args.max_len

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16,
        device_map="cpu", trust_remote_code=True,
    ).eval()
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.config._attn_implementation = "eager"
    print(f"Loaded: {model.config.num_hidden_layers} ly, vocab={model.config.vocab_size}")

    Qwen3_5GatedDeltaNet.forward = _patched_dn_fwd
    wrapper = Qwen35KVCacheWrapper(model.model, args.max_len, model.lm_head)

    # I/O names
    inames = ["input_ids", "position"]
    for i in range(NL_DN): inames.append(f"s_past_{i}")
    for i in range(NL_DN): inames.append(f"c_past_{i}")
    for i in range(NL_GA): inames.append(f"k_past_{i}")
    for i in range(NL_GA): inames.append(f"v_past_{i}")

    onames = ["logits"]
    for i in range(NL_DN): onames.append(f"s_pres_{i}")
    for i in range(NL_DN): onames.append(f"c_pres_{i}")
    for i in range(NL_GA): onames.append(f"k_pres_{i}")
    for i in range(NL_GA): onames.append(f"v_pres_{i}")

    print(f"  I/O: {len(inames)} in, {len(onames)} out")

    # Dummy cache
    cache = []
    for _ in range(NL_DN): cache.append(torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16))
    for _ in range(NL_DN): cache.append(torch.zeros((1, CONV_D, CONV_KS-1), dtype=torch.float16))
    for _ in range(NL_GA): cache.append(torch.zeros((1, KV_H, N, HDIM), dtype=torch.float16))
    for _ in range(NL_GA): cache.append(torch.zeros((1, KV_H, N, HDIM), dtype=torch.float16))

    # PT forward
    di = torch.ones((1,1), dtype=torch.int64); dp = torch.tensor([0], dtype=torch.int64)
    with torch.no_grad():
        pt = wrapper(di, dp, *cache)
        print(f"PT logits: {pt[0].shape}, [{pt[0].min():.4f},{pt[0].max():.4f}]")

    # Export
    torch.onnx.export(wrapper, (di, dp, *cache), args.output,
                      input_names=inames, output_names=onames,
                      opset_version=15, do_constant_folding=True,
                      dynamo=False, verbose=False)
    print(f"ONNX: {os.path.getsize(args.output)/1024/1024:.1f} MB")

    # ORT
    import onnx, onnxruntime as ort
    onnx.checker.check_model(args.output); print("checker: PASS")
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    feed = {"input_ids": np.ones((1,1),np.int64), "position": np.array([0],np.int64)}
    for i in range(NL_DN): feed[f"s_past_{i}"] = np.zeros((1,K_H,K_DIM,V_DIM),np.float16)
    for i in range(NL_DN): feed[f"c_past_{i}"] = np.zeros((1,CONV_D,CONV_KS-1),np.float16)
    for i in range(NL_GA): feed[f"k_past_{i}"] = feed[f"v_past_{i}"] = np.zeros((1,KV_H,N,HDIM),np.float16)
    ort_out = sess.run(None, feed)
    d = np.abs(pt[0].numpy().astype(np.float16) - ort_out[0]).max()
    print(f"PT vs ORT: max_diff={d:.6f}")
    print("PASS" if d < 0.1 else "FAIL")


if __name__ == "__main__":
    main()
