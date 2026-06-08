# Qwen3-0.6B 在华为 Atlas 200I DK A2 上的部署

## 目录

1. [环境规格](#1-环境规格)
2. [项目结构](#2-项目结构)
3. [第一轮：静态窗口模型 (seq=32)](#3-第一轮静态窗口模型-seq32)
4. [第二轮：KV Cache 模型 (max_len=256)](#4-第二轮-kv-cache-模型-max_len256)
5. [关键问题与解决方案](#5-关键问题与解决方案)
6. [复现指南](#6-复现指南)
7. [性能数据](#7-性能数据)

---

## 1. 环境规格

### 硬件

| 项目 | 规格 |
|------|------|
| 开发板 | 华为 Atlas 200I DK A2 |
| 处理器 | Ascend310B4 NPU + 4核 ARM CPU |
| NPU 内存 | 4 GB (实际可用约 3.5 GB) |
| 系统内存 | 3.4 GB |
| 操作系统 | Ubuntu 22.04 aarch64 |

### 软件

| 项目 | 版本 |
|------|------|
| CANN (开发机) | 8.0.RC3 (x86_64, 容器内运行) |
| CANN (开发板) | 8.0.RC3 (aarch64) |
| 开发板 Python | 3.10.12 + torch(CPU) + transformers |
| 开发机 Python | 3.10.* (pixi 管理) |
| PyTorch | 2.12 (开发机) |
| Transformers | 5.10.2 |
| ONNX opset | 15 |
| 容器运行时 | Podman |

### 模型

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-0.6B Instruct |
| 精度 | FP16 |
| 权重文件 | model.safetensors (1.5 GB) |
| 层数 | 28 |
| Q-heads | 16 |
| K/V-heads | 8 |
| hidden_size | 1024 |
| head_dim | 128 |
| vocab_size | 151936 |
| max_position_embeddings | 40960 |

### CANN 安装包下载

| 文件 | 大小 | 说明 |
|------|------|------|
| `Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run` | 1.9 GB | 基础工具链 |
| `Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip` | 683 MB | 310B 内核包 |

下载地址（华为 Ascend 社区）：
- 基础工具链: `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run`
- 310B 内核包: `https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip`

---

## 2. 项目结构

```
Embedded_FinalHW/
├── model/Qwen3-0.6B/                 # 模型权重 + tokenizer 文件
│   ├── config.json                   # 模型配置
│   ├── model.safetensors             # FP16 权重
│   ├── tokenizer.json                # BPE tokenizer
│   ├── tokenizer_config.json         # 含 chat_template
│   └── vocab.json, merges.txt
│
├── scripts/                          # ONNX 导出 & ATC 转换
│   ├── export_fp16.py                # seq=N 静态窗口导出
│   ├── export_kvcache.py             # KV Cache 导出（monkey-patch）
│   ├── patch_onnx.py                 # GQA Expand→Tile 修补
│   ├── download_model.py             # 下载 Qwen3 模型
│   └── podman_convert.sh             # Podman ATC 转换
│
├── board/                            # 开发板推理脚本
│   ├── gen_text_seq32.py             # seq=32 滑动窗口推理
│   ├── gen_text_kvcache.py           # KV Cache 推理
│   └── acl_verify.py                 # ACL 单次推理验证
│
├── docker/                           # 容器定义
│   └── Containerfile.v2              # CANN 8.0 + 310B 内核
│
├── om_out/                           # 生成的 OM 文件
├── logs/                             # ATC 转换日志
├── pixi.toml / pixi.lock             # Python 环境
├── ARCHIVE.md                        # 本文档
└── AGENTS.md
```

---

## 3. 第一轮：静态窗口模型 (seq=32)

### 3.1 设计思路

不使用 KV Cache，每次把完整序列（left-padded 到 32）作为输入，利用 causal attention 内部的 mask 让模型只关注实际 token。

**Left-padding + causal mask 原理**：
```
输入:  [PAD...PAD,  a,  b,  c,  d]   ← 32 个位置，实际 4 个 token
mask:  [0......0,    1,  1,  1,  1]   ← 0=忽略，1=参与注意力

Causal Attention 保证位置 i 只看 0..i
→ 左边 padding 的 0 不影响右边真实 token 的计算
→ 每步模型能看到完整上下文（最多 32 token）

序列超过 32 时：滑动窗口取最后 32 个 token
```

### 3.2 ONNX 导出

**文件**: `scripts/export_fp16.py`

核心代码：
```python
class StaticForwardWrapper(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,        # 不使用 KV cache
            return_dict=True,
        )
        return outputs.logits
```

**导出命令**：
```bash
pixi run python scripts/export_fp16.py --seq-len 32 \
    --output om_out/qwen3_fp16_seq32.onnx
```

**验证 (ONNX Runtime)**：
```python
session = ort.InferenceSession("qwen3_fp16_seq32.onnx")
logits = session.run(None, {
    "input_ids": np.ones((1,32), np.int64),
    "attention_mask": np.ones((1,32), np.int64),
})[0]
# shape: (1, 32, 151936), dtype: float16
# left-padding 一致性: 最后 5 个位置的 logits 与纯 causal 一致
```

### 3.3 ONNX 修补：GQA Expand → Tile

**问题**：Qwen3 使用 GQA（16 Q-heads / 8 KV-heads）。每层 attention 中用 `Expand` 把 K/V 从 8 头重复到 16 头。PyTorch ONNX 导出时，Expand 的目标 shape `[1, 8, 2, 32, 128]` 通过 `Where+Equal+ConstantOfShape` 动态计算。ATC 编译器无法静态推断这些动态 shape。

**shape 含义**：
```
[1,  8, 2, 32,  128]
 │   │  │  │    └── head_dim: KV 头维度 (hidden/kv_heads = 1024/8)
 │   │  │  └────── seq_len: 序列长度
 │   │  └───────── repeat: GQA 分组因子 (16/8=2)
 │   └──────────── num_kv_heads: K/V 头数
 └──────────────── batch: 批次大小
```

**解决**: 将 28 层 × 2 (K+V) = 56 个 `Expand` 替换为 `Tile` + 静态常量。

**文件**: `scripts/patch_onnx.py`

核心代码：
```python
repeats = np.array([1, 1, 2, 1, 1], dtype=np.int64)  # 在分组维重复2次

for node in m.graph.node:
    if node.op_type == "Expand" and "self_attn" in node.name:
        new_node = helper.make_node("Tile",
            inputs=[node.input[0], tile_repeats_name],
            outputs=list(node.output))
```

**运行**：
```bash
pixi run python scripts/patch_onnx.py \
    qwen3_fp16_seq32.onnx --output qwen3_fp16_seq32_tile.onnx --seq-len 32
```

### 3.4 ATC 转换

```bash
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
OUTPUT_PREFIX=om_out/qwen3_fp16_seq32_tile \
bash scripts/podman_convert.sh
```

### 3.5 开发板推理

**文件**: `board/gen_text_seq32.py`

生成循环（伪代码）：
```python
buffer = prompt_tokens[-32:]
for step in range(max_tokens):
    ids, mask = pad_left(buffer, n_real)         # left-pad 到 32
    logits = model.execute(ids, mask)            # → (32, vocab)
    next_token = sample(logits[n_real-1])        # 最后有效位置的 logit
    buffer.append(next_token)                     # 滑动窗口
```

**结果**：3.6 tok/s，连贯中文

---

## 4. 第二轮：KV Cache 模型 (max_len=256)

### 4.1 设计思路

seq=32 可扩展到 256 但 O(N²) 太慢。KV Cache 让每步只算新 token 的 K/Q/V，注意力看所有历史 Key（O(N)）。

**单模型方案**：Prefill 和 Decode 共用同一个 OM 模型。

### 4.2 Monkey-patch Qwen3Attention

**核心操作**：将每层 attention 中的 `torch.cat(past_k, new_k)` 替换为 `torch.where`。

**文件**: `scripts/export_kvcache.py`

```python
def insert_to_cache(cache, new_kv, position):
    """
    cache:  (1, 8, 256, 128)  ← 预分配的固定大小 K/V 缓冲区
    new_kv: (1, 8, 1, 128)    ← 新 token 的单步 K/V
    position: scalar tensor    ← 当前 token 在序列中的位置

    用 torch.where 将 new_kv 写入 cache 的 position 位置。
    ONNX: Equal + Where —— 全部静态 shape，ATC 可编译。
    """
    L = cache.shape[2]
    idx = torch.arange(L, dtype=torch.int64, device=cache.device)
    mask = idx.unsqueeze(0).unsqueeze(0).unsqueeze(-1) == position.view(1,1,1,1)
    return torch.where(mask, new_kv, cache)
```

**设计关键**：`Where` 原生支持 broadcasting。`mask` shape 为 `(1,1,256,1)`，`new_kv` 为 `(1,8,1,128)`，自动广播到 `(1,8,256,128)`。不需要 `Tile` 或 `Expand`。

**Monkey-patch 三层架构**：

1. **`Qwen3Attention.forward()`** — 替换为 `_patched_attention_forward`，接受 `past_k, past_v, position` 参数，调用 `insert_to_cache`
2. **`Qwen3DecoderLayer.forward()`** — 替换为 `_patched_decoder_forward`，透传 K/V 参数
3. **`KVCacheWrapper.forward()`** — 将 56 个 K/V 组织为显式 I/O，传递给各层

**Wrapper 签名**：
```python
class KVCacheWrapper(nn.Module):
    def forward(self,
        input_ids:  (1, 1) int64,       # 单个 token
        position:   (1,)  int64,         # 当前在序列中的位置
        *kv_past:   56 × (1,8,256,128)   # 所有层的 K/V
    ) -> (
        logits:     (1, 1, 151936),      # 下一个 token 的预测
        *kv_present: 56 × (1,8,256,128)  # 更新后的 K/V
    )
```

**Causal mask**：
```python
def make_attn_mask(max_len, position):
    idx = torch.arange(L)
    mask = idx.unsqueeze(0).unsqueeze(0) > position  # key_pos > current → mask
    bias = torch.full((1, 1, 1, L), float("-inf"))
    return bias.masked_fill(~mask.unsqueeze(2), 0.0)  # (1, 1, 1, L)
```

### 4.3 验证

**ORT 多步生成验证**：
```python
# 初始化空 KV 缓存
kv = {f'k_{i}': zeros(1,8,256,128) for i in range(28)}
kv.update({f'v_{i}': ...})

# 3 步 prefill + 5 步 decode
for pos in range(8):
    feed = {'input_ids': ..., 'position': np.array([pos]), **kv}
    outs = sess.run(None, feed)
    # 更新 KV 从输出
    for i in range(28):
        kv[f'k_{i}'] = outs[1 + 2*i]
        kv[f'v_{i}'] = outs[1 + 2*i + 1]

# 验证: K[0] 在位置 0/1/2 有非零值，说明 prefill 正确填入
```

**结果**: PyTorch baseline == PyTorch patch == ORT（FP16 精度内，max_diff=0.000）

### 4.4 CANN 容器环境

**关键发现**：CANN 8.0 的 ATC 支持 Ascend310B4，但需要安装 `Ascend-cann-kernels-310b` 内核包。注意区分：
- `310P` 内核包：含 Ascend310P1/P3，**不含 B4**
- `310B` 内核包：含 Ascend310B4 算子数据

**容器构建 (docker/Containerfile.v2)**：

```dockerfile
FROM localhost/cann-atc-ubuntu22

# pip3 = CANN 8.0 compiler 依赖
RUN apt-get install -y python3-pip

# 安装 CANN
COPY cann-toolkit.run opp-kernel-310b.run /tmp/
RUN echo "y" | /tmp/cann-toolkit.run --quiet --install --install-path=/usr/local/Ascend/cann
RUN echo "y" | /tmp/opp-kernel-310b.run --quiet --full --install-path=/usr/local/Ascend/cann

# TBE 编译器路径硬编码修复
RUN ln -sfn /usr/local/Ascend/cann/ascend-toolkit/8.0.RC3 \
           /usr/local/Ascend/CANN-1.84

# 环境变量
ENV CANN_BASE=/usr/local/Ascend/cann/ascend-toolkit/8.0.RC3
ENV ASCEND_TOOLKIT_HOME=${CANN_BASE}
ENV ASCEND_OPP_PATH=${CANN_BASE}/opp
ENV PATH=${CANN_BASE}/x86_64-linux/bin:${CANN_BASE}/compiler/bin:${PATH}
ENV LD_LIBRARY_PATH=${CANN_BASE}/x86_64-linux/lib64:${CANN_BASE}/compiler/lib64:...
ENV PYTHONPATH=${CANN_BASE}/python/site-packages:${CANN_BASE}/compiler/python/site-packages
```

**构建命令**：
```bash
podman build -f docker/Containerfile.v2 -t cann-atc-ubuntu22:v4 docker/
```

### 4.5 ATC 转换

```bash
# 自动生成 INPUT_SHAPE (58 inputs)
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")

MODEL_ONNX=om_out/qwen3_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
bash scripts/podman_convert.sh
```

**OM 规格**：
- 输入：58（input_ids + position + 28×past_K + 28×past_V）
- 输出：57（logits + 28×present_K + 28×present_V）
- 大小：1.5 GB

### 4.6 开发板推理

**文件**: `board/gen_text_kvcache.py`

**ACL 内存管理流程**：
```
1. acl.rt.malloc(KV_BYTES) × 56        ← 预分配 device K/V 缓冲
2. 每步:
   a. 构建 input dataset: input_ids(8B) + position(8B) + 56个K/V(512KB each)
   b. 构建 output dataset: logits(303KB) + 56个K/V
   c. acl.mdl.execute()
   d. D2H: logits → 采样
   e. H2D: 更新后的 K/V 从 output 复制到 input buffer
3. acl.rt.free() × 56
```

**生成循环**：
```python
# Prefill
for pos, tid in enumerate(prompt_ids):
    model.execute(int(tid), pos, kv_dev)

# Decode
for step in range(max_new):
    logits = model.execute(current_id, pos, kv_dev)
    next_token = sample(logits)
```

**K/V 缓冲区统计**：每个 tensor `(1,8,256,128) float16 = 512KB`，56 个 = ~28MB

**结果**：连贯中文输出，0.8 tok/s

---

## 5. 关键问题与解决方案

### 5.1 GQA Expand 动态 shape

- **现象**: ATC 报 `Expand_1: Data shape are not compatible`
- **原因**: PyTorch ONNX 用 `Where+Equal+ConstantOfShape` 动态计算 GQA 的重复 shape
- **修复**: `patch_onnx.py` 将 56 个 `Expand` 替换为 `Tile` + 静态常量 `[1,1,2,1,1]`
- **文件**: `scripts/patch_onnx.py`

### 5.2 TBE ccec 路径硬编码

- **现象**: `FileNotFoundError: /usr/local/Ascend/CANN-1.84/x86_64-linux/ccec_compiler/bin/ccec`
- **原因**: `tbe/tvm/contrib/ccec.py:44` 硬编码绝对路径
- **修复**: 容器内 `ln -sf <实际CANN路径> /usr/local/Ascend/CANN-1.84`
- **文件**: `docker/Containerfile.v2:16-17`

### 5.3 310P vs 310B 内核包

- **现象**: 用 310P 内核编译的 OM 在开发板加载失败 (`ret=500002`)
- **原因**: 开发板是 Ascend310B4，需 310B 内核包（310P 包只含 P1/P3）
- **修复**: 下载 `Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip`
- **文件**: `docker/Containerfile.v2:7`

### 5.4 Qwen3 thinking 模板

- **现象**: 模型生成大量 `<think>...</think>` 标签
- **原因**: Qwen3 0.6B 自带 reasoning，尺寸太小效果不好
- **修复**: `apply_chat_template(..., enable_thinking=False)` — 空 `<think>` 块信号
- **文件**: `board/gen_text_kvcache.py:163`

### 5.5 NPU 进程残留

- **现象**: ACL 进程被 kill 后 NPU 内存不释放，后续 `load_from_file` 卡住
- **原因**: D 状态下进程被 kill，驱动资源无法自动回收
- **修复**: 开发板重启

### 5.6 ACL Python API 差异

- **现象**: `acl.mdl.add_dataset_buffer` 返回值预期 int，实际为 tuple
- **实际返回**: `(dataset_ptr, ret_code)` — 需解包 `_, ret = ...`
- **文件**: `board/gen_text_kvcache.py` + `board/acl_verify.py`

---

## 6. 复现指南

### 6.1 环境准备 (开发机 x86_64)

```bash
cd Embedded_FinalHW

# Python 环境
pixi install

# 下载模型
pixi run python scripts/download_model.py

# 下载 CANN 8.0 安装包
cd cann_install
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run"
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip"
cd ..

# 准备容器素材
cp cann_install/Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run docker/cann-toolkit.run
cd cann_install
unzip -o Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip -d _310b
unzip -o _310b/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip -d _310b_inner
cp _310b_inner/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.run ../docker/opp-kernel-310b.run
cd ..

# 构建容器
podman build -f docker/Containerfile.v2 -t cann-atc-ubuntu22:v4 docker/
```

### 6.2 导出 seq=32 模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_fp16.py --seq-len 32 \
    --output om_out/qwen3_fp16_seq32.onnx

# 2. Patch GQA Expand → Tile
pixi run python scripts/patch_onnx.py \
    om_out/qwen3_fp16_seq32.onnx \
    --output om_out/qwen3_fp16_seq32_tile.onnx --seq-len 32

# 3. ATC 转 OM
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
bash scripts/podman_convert.sh

# 4. SCP 到开发板
sshpass -p 'Mind@123' scp om_out/qwen3_fp16_seq32_tile.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp model/Qwen3-0.6B/tokenizer* model/Qwen3-0.6B/vocab* \
    model/Qwen3-0.6B/merges* root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp board/gen_text_seq32.py root@192.168.137.100:/root/slm_deploy/

# 5. 运行
ssh root@192.168.137.100 "
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 /root/slm_deploy/gen_text_seq32.py --prompt '你好'
"
```

### 6.3 导出 KV Cache 模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

# 2. 验证 (ORT 多步)
pixi run python -c "
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('om_out/qwen3_kvcache_max256.onnx')
# ... prefill + decode loop
"

# 3. ATC 转 OM
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")
MODEL_ONNX=om_out/qwen3_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
bash scripts/podman_convert.sh

# 4. SCP + 运行
sshpass -p 'Mind@123' scp om_out/qwen3_kvcache_max256.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp board/gen_text_kvcache.py \
    root@192.168.137.100:/root/slm_deploy/
ssh root@192.168.137.100 "
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python3 /root/slm_deploy/gen_text_kvcache.py --prompt '你好'
"
```

### 6.4 开发板环境确认

```bash
ssh root@192.168.137.100  # 密码: Mind@123

source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, transformers, numpy; print('OK')"
npu-smi info
```

---

## 7. 性能数据

| 方案 | 上下文 | 每步耗时 | 速度 | NPU 内存 | OM 大小 |
|------|--------|---------|------|---------|---------|
| seq=1 | 1 token | 125 ms | 8.0 tok/s | ~3.3 GB | 1.5 GB |
| seq=32 (Tile) | 32 token | 280 ms | 3.6 tok/s | ~3.3 GB | 1.5 GB |
| KV Cache (256) | 256 token | 1200 ms | 0.8 tok/s | ~3.4 GB | 1.5 GB |

### 每步耗时估算 (KV Cache, 1200ms)

```
Q/K/V 投影 + QK-Norm:     ~50 ms
RoPE 位置编码:             ~10 ms
insert_to_cache (Where):   ~5 ms
注意力 (Q×full_K^T):       ~500 ms  ← 主瓶颈 (1×256 attention)
MLP 前馈网络:              ~500 ms  ← 28×2×1024×3072
输出投影 (lm_head):         ~50 ms
ACL H2D/D2H (28MB K/V):    ~10 ms
Python 开销:                ~75 ms
```

**注意力的 O(max_len) 开销**：当前模型对所有 256 个位置的完整 K/V 做注意力。真正高效 KV Cache 应每步只算新 Q 与所有 K 的点积（增量注意力）。

---

*文档最后更新: 2026-06-08*
