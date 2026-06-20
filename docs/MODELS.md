# 模型对照表

---

## Qwen3-0.6B

| 属性 | 值 |
|------|-----|
| 架构 | Qwen3ForCausalLM (GQA only) |
| 参数量 | ~600M |
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 |
| head_dim | 128 |
| vocab_size | 151936 |

**部署方案:**
- KV Cache (256 tok): 1.5 GB OM, ~3.6 tok/s
- 需 `enable_thinking=False`（模型不支持）

---

## Qwen3.5-0.8B

| 属性 | 值 |
|------|-----|
| 架构 | Qwen3_5ForConditionalGeneration (DeltaNet + GQA) |
| 参数量 | ~800M |
| hidden_size | 1024 |
| num_hidden_layers | 24 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 16 |
| full_attention_interval | 4 |
| vocab_size | 248320 |
| tie_word_embeddings | true |

**层分布:** 18 DeltaNet (layers 0-17) + 6 Full Attention (layers 18-23)

**部署方案:**
- KV Cache (256 tok): 1.9 GB OM, ~4.7 tok/s
- KV Cache (4096 tok): 2.0 GB OM
- SplitNN (4/16/4): Prefix+Suffix ~1.4 GB OM
- SplitNN (4/16/4, 16K): Prefix+Suffix ~2.8 GB OM

---

## Qwen3.5-2B

| 属性 | 值 |
|------|-----|
| 架构 | Qwen3_5ForConditionalGeneration |
| 参数量 | ~2B |
| hidden_size | 2048 |
| intermediate_size | 6144 |
| num_hidden_layers | 24 |
| num_attention_heads | 8 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 16 |
| full_attention_interval | 4 |
| vocab_size | 248320 |
| tie_word_embeddings | true |

**部署方案:**
- SplitNN 参数绑定 (0/24/0): tied_weight 970 MB, ~5.3 tok/s

---

## Qwen3.5-4B

| 属性 | 值 |
|------|-----|
| 架构 | Qwen3_5ForConditionalGeneration |
| 参数量 | ~4B |
| hidden_size | 2560 |
| intermediate_size | 9728 |
| num_hidden_layers | 32 |
| num_attention_heads | 10 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| linear_num_key_heads | 10 |
| linear_num_value_heads | 10 |
| full_attention_interval | 4 |
| vocab_size | 248320 |
| tie_word_embeddings | true |

**部署方案:**
- SplitNN OM (1/30/1, 16K): Prefix+Suffix ~2.8 GB OM, ~1.1 tok/s
