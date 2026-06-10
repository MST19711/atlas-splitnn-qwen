#!/usr/bin/env python3
"""Qwen3.5-0.8B KV Cache ONNX 导出 — 最小 monkey-patch.
只 patch 三个问题:
 1. Attention cat→Where
 2. Trilu causal mask → Where+Equal
 3. Conv1D 的 copy_ → 非 in-place
DeltaNet 的 recurrent 计算使用原生 torch 函数."""

import argparse, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet, Qwen3_5Attention, Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
    torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
)

MAX_LEN = 256; NL_DN = 18; NL_GA = 6
K_H = 16; K_DIM = 128; V_DIM = 128
KV_H = 2; HDIM = 256; CONV_D = 6144; CONV_KS = 4


# ── Helper: causal mask (Trilu-free) ───────────────────────────────
def make_attn_mask(max_len, position):
    idx = torch.arange(max_len, dtype=torch.int64)
    m = idx.unsqueeze(0).unsqueeze(0) > position
    bias = torch.full((1, 1, 1, max_len), float("-inf"), dtype=torch.float16)
    return bias.masked_fill(~m.unsqueeze(2), 0.0)


# ── Helper: GQA expand ─────────────────────────────────────────────
def _repeat_kv(x, n_rep):
    if n_rep == 1: return x
    B, H, L, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, L, D).reshape(B, H * n_rep, L, D)


# ── Patch 1: RMSNorm — type_as → to(dtype) ────────────────────────
_orig_rmsnorm_fwd = Qwen3_5RMSNorm.forward
def _patched_rmsnorm_fwd(self, x):
    output = self._norm(x.float()) * (1.0 + self.weight.float())
    return output.to(x.dtype)
Qwen3_5RMSNorm.forward = _patched_rmsnorm_fwd


# ── Patch 2: Conv state — 原生 torch_causal_conv1d_update 用 copy_,
#   ONNX 不支持。替换为返回新 state 的版本。 ──────────────────────
def _safe_conv_update(hidden_states, conv_state, weight, bias, activation):
    """Same as torch_causal_conv1d_update but returns (out, new_state)."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    inp = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_state = inp[:, :, -state_len:]
    out = F.conv1d(inp, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype), new_state


# ── Patch 3: DeltaNet forward — 用原生 recurrent, 只修 conv ──────
def _patched_dn_fwd(self, hidden_states, attention_mask=None,
                    past_S=None, past_conv=None, cache_params=None, **kw):
    """Thin wrapper: calls native logic but avoids conv state copy_."""
    B, T, _ = hidden_states.shape
    assert T == 1

    mqkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    # Conv with explicit state (avoids copy_ in torch_causal_conv1d_update)
    mqkv_t = mqkv.transpose(1, 2)
    cs = past_conv if past_conv is not None else torch.zeros(1, mqkv.shape[-1], self.conv_kernel_size - 1, dtype=mqkv.dtype)
    mqkv_t, new_conv = _safe_conv_update(mqkv_t, cs, self.conv1d.weight.squeeze(1), self.conv1d.bias, 'silu')
    mqkv = mqkv_t.transpose(1, 2)

    q, k, v = torch.split(mqkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    q = q.reshape(B, T, -1, self.head_k_dim)
    k = k.reshape(B, T, -1, self.head_k_dim)
    v = v.reshape(B, T, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    S = past_S if past_S is not None else torch.zeros(1, self.num_k_heads, self.head_k_dim, self.head_v_dim, dtype=v.dtype)

    # 原生 torch GDN recurrent（与手写 delta_step 数学等价）
    # 注意: 函数内部会做 transpose(1,2) → 传原始形状 [B,T,H,D] 即可
    out, S_new = torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, S, True, use_qk_l2norm_in_kernel=True)

    out = out.to(hidden_states.dtype); S_new = S_new.to(hidden_states.dtype)
    out = self.norm(out.reshape(-1, self.head_v_dim), z[:, 0].reshape(-1, self.head_v_dim))
    out = out.reshape(B, T, -1)
    return self.out_proj(out), S_new, new_conv


# ── Patch 4: Attention (cat→Where) ─────────────────────────────────
class AttentionCacheWrapper:
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


def _patched_attn_fwd(self, hidden_states, position_embeddings=None,
                      attention_mask=None, past_key_values=None, **kw):
    bsz, q_len, _ = hidden_states.size()
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    n_rep = self.num_key_value_groups
    key_states = _repeat_kv(key_states, n_rep)
    value_states = _repeat_kv(value_states, n_rep)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = attn_output * torch.sigmoid(gate)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# ── Wrapper ─────────────────────────────────────────────────────────
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

        s_states  = list(cache_flat[:NL_DN])
        c_states  = list(cache_flat[NL_DN:NL_DN+NL_DN])
        k_states  = list(cache_flat[NL_DN*2:NL_DN*2+NL_GA])
        v_states  = list(cache_flat[NL_DN*2+NL_GA:])

        pres_S = [None]*NL_DN; pres_C = [None]*NL_DN
        pres_K = [None]*NL_GA; pres_V = [None]*NL_GA
        di, gi = 0, 0

        for layer in self.model.layers:
            res = hidden
            hidden = layer.input_layernorm(hidden)

            if layer.layer_type == 'linear_attention':
                hs, S_new, C_new = layer.linear_attn(
                    hidden, past_S=s_states[di], past_conv=c_states[di], position=position)
                pres_S[di] = S_new; pres_C[di] = C_new
                di += 1
                hidden = hs
            else:
                ca = AttentionCacheWrapper(k_states[gi], v_states[gi], position)
                hidden, _ = layer.self_attn(
                    hidden, attention_mask=attn_mask,
                    position_embeddings=pos_emb, position_ids=pos_ids,
                    past_key_values=ca)
                pres_K[gi] = ca.present_k; pres_V[gi] = ca.present_v
                gi += 1

            hidden = res + hidden
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
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cpu", trust_remote_code=True,
    ).eval().to(torch.float16)
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        if hasattr(layer, 'self_attn'):
            layer.self_attn.config._attn_implementation = "eager"
    print(f"Loaded: {model.config.num_hidden_layers} ly, vocab={model.config.vocab_size}")

    Qwen3_5GatedDeltaNet.forward = _patched_dn_fwd
    Qwen3_5Attention.forward = _patched_attn_fwd
    wrapper = Qwen35KVCacheWrapper(model.model, args.max_len, model.lm_head)

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

    cache = []
    for _ in range(NL_DN): cache.append(torch.zeros((1, K_H, K_DIM, V_DIM), dtype=torch.float16))
    for _ in range(NL_DN): cache.append(torch.zeros((1, CONV_D, CONV_KS-1), dtype=torch.float16))
    for _ in range(NL_GA): cache.append(torch.zeros((1, KV_H, N, HDIM), dtype=torch.float16))
    for _ in range(NL_GA): cache.append(torch.zeros((1, KV_H, N, HDIM), dtype=torch.float16))

    di = torch.ones((1,1), dtype=torch.int64); dp = torch.tensor([0], dtype=torch.int64)
    with torch.no_grad():
        pt = wrapper(di, dp, *cache)
        print(f"PT logits: {pt[0].shape}, [{pt[0].min():.4f},{pt[0].max():.4f}]")

    torch.onnx.export(wrapper, (di, dp, *cache), args.output,
                      input_names=inames, output_names=onames,
                      opset_version=15, do_constant_folding=True,
                      dynamo=False, verbose=False)
    print(f"ONNX: {os.path.getsize(args.output)/1024/1024:.1f} MB")

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
