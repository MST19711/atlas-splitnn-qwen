#!/usr/bin/env python3
"""Qwen3.5 SplitNN shared helpers.

Shared by:
- ONNX export wrappers for prefix/suffix
- CUDA middle-segment service
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,
    apply_rotary_pos_emb,
    torch_recurrent_gated_delta_rule,
)

MAX_LEN = 16384
FULL_NL_DN = 18
FULL_NL_GA = 6
K_H = 16
K_DIM = 128
V_DIM = 128
KV_H = 2
HDIM = 256
CONV_D = 6144
CONV_KS = 4
HIDDEN_SIZE = 1024
VOCAB_SIZE = 248320

PREFIX_START = 0
PREFIX_END = 4
MIDDLE_START = 4
MIDDLE_END = 20
SUFFIX_START = 20
SUFFIX_END = 24

PREFIX_NL_DN = 3
PREFIX_NL_GA = 1
MIDDLE_NL_DN = 12
MIDDLE_NL_GA = 4
SUFFIX_NL_DN = 3
SUFFIX_NL_GA = 1

_PATCHED = False


def make_attn_mask(max_len: int, position: torch.Tensor) -> torch.Tensor:
    device = position.device
    idx = torch.arange(max_len, dtype=torch.int64, device=device)
    m = idx.unsqueeze(0).unsqueeze(0) > position
    bias = torch.full((1, 1, 1, max_len), float("-inf"), dtype=torch.float16, device=device)
    return bias.masked_fill(~m.unsqueeze(2), 0.0)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, heads, seq_len, dim = x.shape
    return x[:, :, None, :, :].expand(batch, heads, n_rep, seq_len, dim).reshape(
        batch, heads * n_rep, seq_len, dim
    )


def _safe_conv_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    inp = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    new_state = inp[:, :, -state_len:]
    out = F.conv1d(inp, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype), new_state


def apply_qwen35_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return

    def _patched_rmsnorm_fwd(self, x):
        output = self._norm(x.float()) * (1.0 + self.weight.float())
        return output.to(x.dtype)

    def _patched_dn_fwd(
        self,
        hidden_states,
        attention_mask=None,
        past_S=None,
        past_conv=None,
        cache_params=None,
        **kw,
    ):
        batch, seq_len, _ = hidden_states.shape
        assert seq_len == 1

        mqkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mqkv_t = mqkv.transpose(1, 2)
        if past_conv is None:
            past_conv = torch.zeros(
                batch,
                mqkv.shape[-1],
                self.conv_kernel_size - 1,
                dtype=mqkv.dtype,
                device=mqkv.device,
            )
        mqkv_t, new_conv = _safe_conv_update(
            mqkv_t,
            past_conv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
        )
        mqkv = mqkv_t.transpose(1, 2)

        q, k, v = torch.split(mqkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(batch, seq_len, -1, self.head_k_dim)
        k = k.reshape(batch, seq_len, -1, self.head_k_dim)
        v = v.reshape(batch, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if past_S is None:
            past_S = torch.zeros(
                batch,
                self.num_k_heads,
                self.head_k_dim,
                self.head_v_dim,
                dtype=v.dtype,
                device=v.device,
            )

        out, s_new = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, past_S, True, use_qk_l2norm_in_kernel=True
        )

        out = out.to(hidden_states.dtype)
        s_new = s_new.to(hidden_states.dtype)
        out = self.norm(
            out.reshape(-1, self.head_v_dim),
            z[:, 0].reshape(-1, self.head_v_dim),
        )
        out = out.reshape(batch, seq_len, -1)
        return self.out_proj(out), s_new, new_conv

    class AttentionCacheWrapper:
        def __init__(self, k_buf, v_buf, position):
            self._k = k_buf
            self._v = v_buf
            self._pos = position
            self.present_k = k_buf
            self.present_v = v_buf

        def update(self, key_states, value_states, layer_idx):
            length = self._k.shape[2]
            idx = torch.arange(length, dtype=torch.int64, device=self._k.device)
            mask = idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) == self._pos.view(1, 1, 1, 1)
            new_k = torch.where(mask, key_states, self._k)
            new_v = torch.where(mask, value_states, self._v)
            self.present_k = new_k
            self.present_v = new_v
            return new_k, new_v

    def _patched_attn_fwd(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kw,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

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

    Qwen3_5RMSNorm.forward = _patched_rmsnorm_fwd
    Qwen3_5GatedDeltaNet.forward = _patched_dn_fwd
    Qwen3_5Attention.forward = _patched_attn_fwd
    _PATCHED = True


def configure_eager_attention(model) -> None:
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    for layer in model.model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.config._attn_implementation = "eager"


def build_segment_io_names(
    input0_name: str,
    output0_name: str,
    nl_dn: int,
    nl_ga: int,
) -> tuple[list[str], list[str]]:
    inames = [input0_name, "position"]
    for i in range(nl_dn):
        inames.append(f"s_past_{i}")
    for i in range(nl_dn):
        inames.append(f"c_past_{i}")
    for i in range(nl_ga):
        inames.append(f"k_past_{i}")
    for i in range(nl_ga):
        inames.append(f"v_past_{i}")

    onames = [output0_name]
    for i in range(nl_dn):
        onames.append(f"s_pres_{i}")
    for i in range(nl_dn):
        onames.append(f"c_pres_{i}")
    for i in range(nl_ga):
        onames.append(f"k_pres_{i}")
    for i in range(nl_ga):
        onames.append(f"v_pres_{i}")
    return inames, onames


class SegmentRunner(nn.Module):
    def __init__(self, text_model, start: int, end: int, max_len: int, nl_dn: int, nl_ga: int):
        super().__init__()
        self.model = text_model
        self.layers = nn.ModuleList(list(text_model.layers[start:end]))
        self.max_len = max_len
        self.nl_dn = nl_dn
        self.nl_ga = nl_ga

    def _forward_layers(self, hidden: torch.Tensor, position: torch.Tensor, cache_flat):
        attn_mask = make_attn_mask(self.max_len, position)
        pos_ids = position.unsqueeze(0)
        pos_emb = self.model.rotary_emb(hidden, pos_ids)

        s_states = list(cache_flat[: self.nl_dn])
        c_states = list(cache_flat[self.nl_dn : self.nl_dn * 2])
        k_states = list(cache_flat[self.nl_dn * 2 : self.nl_dn * 2 + self.nl_ga])
        v_states = list(cache_flat[self.nl_dn * 2 + self.nl_ga :])

        pres_s = [None] * self.nl_dn
        pres_c = [None] * self.nl_dn
        pres_k = [None] * self.nl_ga
        pres_v = [None] * self.nl_ga
        di = 0
        gi = 0

        for layer in self.layers:
            res = hidden
            hidden = layer.input_layernorm(hidden)

            if layer.layer_type == "linear_attention":
                hidden, s_new, c_new = layer.linear_attn(
                    hidden,
                    past_S=s_states[di],
                    past_conv=c_states[di],
                    position=position,
                )
                pres_s[di] = s_new
                pres_c[di] = c_new
                di += 1
            else:
                cache = _AttentionCacheWrapper(k_states[gi], v_states[gi], position)
                hidden, _ = layer.self_attn(
                    hidden,
                    attention_mask=attn_mask,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                    past_key_values=cache,
                )
                pres_k[gi] = cache.present_k
                pres_v[gi] = cache.present_v
                gi += 1

            hidden = res + hidden
            res = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = res + hidden

        return hidden, pres_s, pres_c, pres_k, pres_v


class _AttentionCacheWrapper:
    def __init__(self, k_buf, v_buf, position):
        self._k = k_buf
        self._v = v_buf
        self._pos = position
        self.present_k = k_buf
        self.present_v = v_buf

    def update(self, key_states, value_states, layer_idx):
        length = self._k.shape[2]
        idx = torch.arange(length, dtype=torch.int64, device=self._k.device)
        mask = idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) == self._pos.view(1, 1, 1, 1)
        new_k = torch.where(mask, key_states, self._k)
        new_v = torch.where(mask, value_states, self._v)
        self.present_k = new_k
        self.present_v = new_v
        return new_k, new_v


class PrefixWrapper(SegmentRunner):
    def __init__(self, text_model, max_len: int):
        super().__init__(text_model, PREFIX_START, PREFIX_END, max_len, PREFIX_NL_DN, PREFIX_NL_GA)

    def forward(self, input_ids, position, *cache_flat):
        hidden = self.model.embed_tokens(input_ids)
        hidden, pres_s, pres_c, pres_k, pres_v = self._forward_layers(hidden, position, cache_flat)
        return (hidden, *pres_s, *pres_c, *pres_k, *pres_v)


class MiddleWrapper(SegmentRunner):
    def __init__(self, text_model, max_len: int):
        super().__init__(text_model, MIDDLE_START, MIDDLE_END, max_len, MIDDLE_NL_DN, MIDDLE_NL_GA)

    def forward(self, hidden_states, position, *cache_flat):
        hidden, pres_s, pres_c, pres_k, pres_v = self._forward_layers(hidden_states, position, cache_flat)
        return (hidden, *pres_s, *pres_c, *pres_k, *pres_v)


class SuffixWrapper(SegmentRunner):
    def __init__(self, text_model, max_len: int, lm_head):
        super().__init__(text_model, SUFFIX_START, SUFFIX_END, max_len, SUFFIX_NL_DN, SUFFIX_NL_GA)
        self.lm_head = lm_head

    def forward(self, hidden_states, position, *cache_flat):
        hidden, pres_s, pres_c, pres_k, pres_v = self._forward_layers(hidden_states, position, cache_flat)
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return (logits, *pres_s, *pres_c, *pres_k, *pres_v)
