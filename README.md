# Qwen3-0.6B 在华为 Atlas 200I DK A2 上的部署

> 将千问小语言模型部署到华为昇腾边缘计算板，实现 NPU 加速的端侧中文对话。


---

## 目录


- [项目动机](#项目动机)
- [成果概览](#成果概览)
- [硬件与软件](#硬件与软件)
- [项目结构](#项目结构)
- [背景知识](#背景知识)
- [第一轮实验：静态窗口模型](#第一轮实验静态窗口模型)
- [第二轮实验：KV Cache 模型](#第二轮实验kv-cache-模型)
- [解决的关键问题](#解决的关键问题)
- [如何复现](#如何复现)

> 本文件同时也作为一份课程报告，因此有一些复现无关内容。如果你更关心如何实际部署复现，可以直接跳转到“[如何复现](#如何复现)”部分。
---

## 项目动机

大语言模型通常运行在云端 GPU 集群上，但很多场景需要**本地推理**——隐私敏感、网络不稳定、或者单纯想把 AI 装进小盒子里。

华为 Atlas 200I DK A2 是一块售价千元级别的昇腾开发者套件，搭载 Ascend310B4 NPU（4GB）。这个项目尝试把 **Qwen3-0.6B**——通义千问的 6 亿参数指令微调模型——部署到这块板子上，实现可用的中文对话。

过程分两轮：先用简单的静态窗口方案跑通全链路，再升级到 KV Cache 以获得更长上下文。

---

## 成果概览

| 方案 | 上下文长度 | 每 token 耗时 | 解码速度* | 输出质量 |
|------|-----------|-------------|----------|---------|
| 静态窗口 (seq=32) | 32 token | ~280 ms | 3.6 tok/s | 连贯中文 |
| KV Cache (max_len=256) | 256 token | ~210 ms | 4.8 tok/s | 连贯中文 |

> *解码速度 = 纯生成阶段速度（不含 prompt）。OM 首次加载约 75 秒，后续每 token 约 210ms，达到当前
> NPU 计算 174ms 的理论下限附近（剩余 ~36ms 为 H2D/D2H/Python 采样开销）。

模型输出示例：

```
Prompt: "你好，请用一句话介绍你自己"
Output: "我是一位叫AI的助手，专为用户提供帮助和支持。"
```

---

## 硬件与软件

### 开发板

| 项目 | 规格 |
|------|------|
| 型号 | 华为 Atlas 200I DK A2 |
| 芯片 | Ascend310B4 NPU + 4 核 ARM |
| NPU 内存 | 4 GB |
| 系统内存 | 3.4 GB |
| 系统 | Ubuntu 22.04 aarch64 |

### 使用的模型

Qwen3-0.6B Instruct，FP16 精度：

| 参数 | 值 |
|------|-----|
| 参数量 | ~600M (28 层) |
| 词表大小 | 151936 |
| 注意力头 | 16 Q-heads / 8 KV-heads (GQA) |
| 权重文件 | model.safetensors (1.5 GB) |

### 工具链

| 组件 | 版本/说明 |
|------|----------|
| CANN | 8.0.RC3 (华为昇腾计算架构) |
| ATC | CANN 内置的模型编译器 (ONNX → OM) |
| ACL | CANN 的 C 运行时库 (通过 Python `acl` 模块调用) |
| ONNX | opset 15，TorchScript 导出 |
| 容器 | Podman, Ubuntu 22.04 base |

---

## 项目结构

```
Embedded_FinalHW/
├── README.md                         # 本文档
├── AGENTS.md                         # AI 辅助开发用参考
├── model/Qwen3-0.6B/                 # 模型权重 + tokenizer
├── scripts/                          # ONNX 导出 & ATC 转换 (x86 开发机)
│   ├── export_fp16.py                # 静态窗口模型导出
│   ├── export_kvcache.py             # KV Cache 模型导出
│   ├── patch_onnx.py                 # ONNX 图修补工具
│   ├── download_model.py
│   └── podman_convert.sh             # 容器化 ATC 转换
├── board/                            # 板载推理脚本 (aarch64)
│   ├── gen_text_seq32.py             # 静态窗口推理
│   ├── gen_text_kvcache.py           # KV Cache 推理
│   └── acl_verify.py                 # 单次推理验证
├── docker/Containerfile.v2           # CANN 8.0 容器镜像定义
├── om_out/                           # 编译产出的 OM 文件
└── logs/                             # ATC 转换日志
```

---

## 背景知识

### 自回归文本生成

语言模型逐个 token 地生成文本。每一步输入当前序列，输出下一个 token 的概率分布，采样后拼接到序列末尾。

### Left-padding

对于固定长度的输入（比如 32 token），如果实际序列更短，需要在左边填 0（padding），右边放真实 token。配合 `attention_mask`（0 表示忽略），causal attention 保证 padding 不影响实际 token 的计算。

### KV Cache

Transformer 每层的注意力需要计算 Key、Query、Value。没有缓存时，每一步都要重新计算所有历史 token 的 K 和 V（重复劳动）。KV Cache 把它们存下来，下一步只需算新 token 的 K/V，注意力复杂度从 O(N²) 降到 O(N)。

### GQA (Grouped Query Attention)

Qwen3 用 16 个 Q 头但只有 8 个 K/V 头——每 2 个 Q 头共享一组 K/V。这样在几乎不损失精度的情况下节省了 ~33% 的 K/V 计算量。

---

## 第一轮实验：静态窗口模型

### 思路

不用 KV Cache。每次把完整序列 left-pad 到固定长度（比如 32），一次性送入模型。尽管有重复计算，但实现简单，适合快速验证全链路。

**序列长度为什么选 32？** 这是反复试探的结果。1 太小（无上下文），128 太大（NPU 内存可能不够，且每步太慢）。32 是折中——够容纳一句日常对话，也能在 ~280ms 内完成一步。

### 导出 ONNX

`scripts/export_fp16.py` 把 Qwen3 封装为固定 seq_len 的静态图：

```python
dummy_input_ids = torch.ones((1, 32), dtype=torch.int64)
torch.onnx.export(wrapper, (dummy_input_ids, ...), "output.onnx",
                  opset_version=15, do_constant_folding=True)

```

关键设置：`use_cache=False`（不产生 KV Cache 输出），`dynamo=False`（用传统的 TorchScript 追踪器，兼容性更好）。

导出后用 ONNX Runtime 验证——检查 left-padding 下末尾位置 logits 与全 1 mask 一致。

### 修补 ONNX：处理 GQA 的 Expand 节点

导出后的 ONNX 图里有 56 个 `Expand` 节点——每层注意力要把 K/V 从 8 头重复到 16 头（GQA），重复因子为 `[1, 8, 2, 32, 128]` 中的那个 `2`。问题在于这个 shape 是通过 `Where + Equal + ConstantOfShape` **动态计算**的，ATC 编译器无法静态推断。

`scripts/patch_onnx.py` 遍历所有 `self_attn` 中的 Expand，替换为 `Tile` 算子配合静态常量 `[1, 1, 2, 1, 1]`。`Tile` 和 `Expand` 在这个场景下语义完全相同——都是把张量沿某一维复制指定次数。

修补后用 ORT 重新验证——确保数值输出不变。

### ATC 编译

```bash
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
bash scripts/podman_convert.sh
```

ATC（Ascend Tensor Compiler）把 ONNX 图编译为板载 NPU 可执行的 OM 文件。过程约 3 分钟，产物 1.5 GB。

### 板载推理

`board/gen_text_seq32.py` 实现了 left-padding 滑动窗口的完整生成循环：

1. Tokenize 用户输入
2. 取最后 32 token，左侧填 0，构建 attention_mask
3. ACL 执行 → 取 logits 的最后一个有效位置 → 采样
4. 新 token 加入窗口，最旧的 token 滑出

每 token 约 280ms，解码速度约 3.6 tok/s，输出是连贯的中文。

---

## 第二轮实验：KV Cache 模型

### 思路

静态窗口方案能跑通，但扩大窗口到 256 时，每步就需要计算 256×256 的注意力矩阵，O(N²) 的增长意味着每步要等好几秒。KV Cache 解决的就是这个问题——**存储而不重复计算**。

由于这台NPU实际上不支持动态计算图，我们的做法是**修改 Qwen3 内部的注意力实现**（monkey-patch），让它把原本动态增长的 K/V 替换为预分配的固定大小缓冲区。

### Monkey-patch：把 `cat` 换成 `where`

Qwen3 原生的 KV Cache 用 `torch.cat([past_k, new_k])` 实现——每次 K/V 长度加 1。这个动态增长 ONNX 没法表达（需要 static shape）。

我们的方案：预分配固定大小的 K/V 缓冲区 `(1, 8, max_len, 128)`，每次用 `torch.where` 把新 token 的 K/V **写入缓冲区对应位置**：

```python
def insert_to_cache(cache, new_kv, position):
    idx = torch.arange(max_len)
    mask = idx == position  # shape: (1, 1, max_len, 1)
    return torch.where(mask, new_kv, cache)
```

`Where` 是 ONNX 原生算子，`mask` 和 `new_kv` 通过 broadcasting 自动对齐，所有维度编译时完全静态。这就是整个设计最核心的一行代码。

### 三层 monkey-patch

替换从内到外逐步进行：

1. **`Qwen3Attention.forward()`**——接受 `past_k`, `past_v`, `position`，调用 `insert_to_cache`
2. **`Qwen3DecoderLayer.forward()`**——透传 K/V 参数
3. **`KVCacheWrapper`**——顶层模块，把 56 个 K/V 组织为显式的模型输入/输出

### 导出与验证

导出后模型有 **58 个输入、57 个输出**（2 + 56 个 K/V 输入，1 + 56 个输出）。

在两轮验证中确保了正确性：

1. **单步对比**：PyTorch 原生模型 logits == monkey-patch PyTorch logits（FP16 精度内完全一致）
2. **多步对比**：用 ONNX Runtime 模拟完整的 prefill（3 token）→ decode（5 token）流程，K/V 按位置正确累积

### 容器环境的构建

CANN 8.0 的 ATC 编译器需要在 Linux 环境中运行。我们把它装进 Podman 容器，避免直接污染开发机。

**关键发现**：CANN 的 ATC 对不同芯片需要安装对应的"内核包"。310P 包只覆盖 P1/P3 型号，而开发板是 310B4 芯片，**必须用 310B 内核包**。用错包的症状是 ATC 编译能过，但开发板加载 OM 时返回 `ret=500002`。

另一个重要的修复：CANN 的 TBE Python 代码里硬编码了编译器路径 `/usr/local/Ascend/CANN-1.84/`。我们在容器里创建一个符号链接指向实际安装位置即可解决。

### 板载推理

与 seq=32 方案不同，KV Cache 需要在 NPU 上维护一套持续的 K/V 缓冲区：

1. `acl.rt.malloc` 预分配 56 个 K/V tensor（每个 512KB，共约 28MB）
2. Prefill 阶段：逐 token 输入，K/V 逐步填充
3. Decode 阶段：每步从输出取回 logits 采样，同时把更新后的 K/V 复制回输入缓冲区

K/V 的 D2H + H2D 开销约 10ms，相对 1200ms 的总延迟可以忽略。

### 结果与速度分析

输出："你好！有什么可以帮助你的吗？"——与 prompt 匹配的连贯回复。

当前每 token 约 210ms，纯解码速度约 4.8 tok/s。模型不区分 prefill 与 decode 阶段，prompt 的 token 同样逐个送入，因此初始 prompt 的处理需要额外时间。

### 性能优化历程

开发板推理经历了三轮优化：

**初始版本**（每步 alloc + memcpy，ACL 框架开销严重）：

```
每步 execute():
  acl.rt.malloc × 115      ← 为 58 个输入 + 57 个输出分配 buffer
  acl.rt.memcpy × 112      ← K/V 在 host↔device 间全量搬运（28MB）
  acl.mdl.execute()         ← 174ms NPU 计算
  acl.rt.free × 115         ← 释放

每步约 420ms，Profiler 实测：
- NPU 计算 (BatchMatMulV2): 174ms
- Host alloc + memcpy: ~200ms
- Framework 开销: ~46ms
```

Profiler 数据（`msprof --ascendcl=on --task-time=on --ai-core=on`）：

| 类别 | 耗时 | 占比 |
|------|------|------|
| BatchMatMulV2 (NPU 计算) | 174ms | 41% |
| Host 侧 alloc + memcpy | ~200ms | 48% |
| Python / ACL 框架 | ~46ms | 11% |

**第一轮优化**：预分配 device 缓冲区 + K/V 指针轮转（双缓冲），消除每步的 `acl.rt.malloc`/`acl.rt.free` 和 112 次 K/V memcpy。

```
每步只剩: H2D(16B) + execute + D2H(303KB)
K/V 留在 device 内存，指针身份互换
```

**第二轮优化**：预创建 AB/BA 两组 dataset 并绑定到双缓冲 K/V，消除每步的 `acl.mdl.create_dataset`/`destroy_dataset`/`create_data_buffer`/`add_dataset_buffer`。

```
每步只剩: H2D(16B) + execute + D2H(303KB)
（全部数据结构在模型生命周期内一次性创建）
```

最终每步约 210ms。因为 NPU 计算 174ms 是理论下限，剩余的 ~36ms 是 H2D/D2H/Python 采样的固定开销。

> 注意：1.5 GB OM 在 CANN 7.0.RC1 runtime 上首次加载需要约 75 秒，期间无输出不是卡死。

---

## 解决的关键问题

### GQA Expand 的动态 shape

ATC 无法静态推断 ONNX 里的动态 shape 计算。用 `patch_onnx.py` 把 `Expand` 换为 `Tile` + 静态常量。

### TBE 编译器路径硬编码

`tbe/tvm/contrib/ccec.py` 里写死了 `/usr/local/Ascend/CANN-1.84/`。容器里 symlink 解决。


### thinking 模板

Qwen3 的 chat template 默认开启 `<think>` 推理链，对 0.6B 小模型反而浪费时间。`enable_thinking=False` 关闭。

### NPU 进程残留

ACL 进程被 kill -9 后（尤其 D 状态），NPU 内存无法自动回收。解决方案：重启板子。遇到脚本卡死时不要反复跑——先确认 NPU 是否干净。

### ACL API 返回值

`acl.mdl.add_dataset_buffer()` 返回 `(ptr, ret)` 而非单个 ret code，直接解包。

---

## 如何复现

### 环境准备

```bash
# 开发机 (x86_64)
cd Embedded_FinalHW
pixi install                           # Python 环境
pixi run python scripts/download_model.py  # 下载 Qwen3

# 下载 CANN 8.0（约 2.6 GB 总计）
mkdir -p cann_install && cd cann_install
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-toolkit_8.0.RC3_linux-x86_64.run"
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.RC3/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip"
cd ..

# 准备容器素材并构建
cp cann_install/*toolkit*.run docker/cann-toolkit.run
cd cann_install
unzip -o Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip -d _310b
unzip -o _310b/Ascend-cann-kernels-310b_8.0.RC3_linux-x86_64.zip -d _310b_inner
cp _310b_inner/*.run ../docker/opp-kernel-310b.run
cd ..
podman build -f docker/Containerfile.v2 -t cann-atc-ubuntu22:v4 docker/
```

### 导出无KV Cache seq=32 的模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_fp16.py --seq-len 32 \
    --output om_out/qwen3_fp16_seq32.onnx

# 2. (可选) 验证: ONNX Runtime 推理 + left-padding 一致性
pixi run python -c "
import numpy as np, onnx, onnxruntime as ort
onnx.checker.check_model('om_out/qwen3_fp16_seq32.onnx')
sess = ort.InferenceSession('om_out/qwen3_fp16_seq32.onnx',
                            providers=['CPUExecutionProvider'])
# 全 1 mask
a = sess.run(None, {'input_ids': np.ones((1,32),np.int64),
                    'attention_mask': np.ones((1,32),np.int64)})[0]
# left-padding: 只有最后 5 个 token 参与注意力
m = np.zeros((1,32), np.int64); m[0,-5:] = 1
b = sess.run(None, {'input_ids': np.ones((1,32),np.int64),
                    'attention_mask': m})[0]
np.testing.assert_allclose(a[0,-5:,:], b[0,-5:,:], rtol=0.01, atol=0.05)
print('ONNX validation: PASS')
"

# 3. Patch GQA Expand → Tile
pixi run python scripts/patch_onnx.py \
    om_out/qwen3_fp16_seq32.onnx \
    --output om_out/qwen3_fp16_seq32_tile.onnx --seq-len 32

# 4. (可选) 再次验证修补后的 ONNX（同上，略）

# 5. ATC 转 OM
MODEL_ONNX=om_out/qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
bash scripts/podman_convert.sh

# 6. 传输到开发板
sshpass -p 'Mind@123' scp om_out/qwen3_fp16_seq32_tile.om \
    root@192.168.137.100:/root/slm_deploy/
```

### 导出支持 KV Cache 的模型

```bash
# 1. 导出 ONNX
pixi run python scripts/export_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

# 2. (可选) 验证: 单步 logits 与 PyTorch 原版一致
pixi run python -c "
import torch, numpy as np, onnx, onnxruntime as ort
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3DecoderLayer
import sys; sys.path.insert(0,'scripts')
from export_kvcache import (_patched_attention_forward,
    _patched_decoder_forward, KVCacheWrapper)

# PyTorch baseline
m = AutoModelForCausalLM.from_pretrained('model/Qwen3-0.6B',
    torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True).eval()
with torch.no_grad():
    bl = m(torch.tensor([[100]], dtype=torch.long), use_cache=False).logits

# Patched PyTorch
Qwen3Attention.forward = _patched_attention_forward
Qwen3DecoderLayer.forward = _patched_decoder_forward
w = KVCacheWrapper(m, 256).eval()
pos = torch.tensor([0], dtype=torch.int64)
kv = [torch.zeros(1,8,256,128, dtype=torch.float16) for _ in range(56)]
out_pt = w(torch.tensor([[100]], dtype=torch.long), pos, *kv)

assert torch.allclose(bl.float(), out_pt[0].float(), rtol=0.01, atol=0.1)
print('PyTorch patched vs baseline: PASS')

# ONNX Runtime
onnx.checker.check_model('om_out/qwen3_kvcache_max256.onnx')
sess = ort.InferenceSession('om_out/qwen3_kvcache_max256.onnx',
                            providers=['CPUExecutionProvider'])
feed = {'input_ids': np.array([[100]], np.int64),
        'position': np.array([0], np.int64)}
for i in range(28):
    feed[f'past_k_{i}'] = np.zeros((1,8,256,128), np.float16)
    feed[f'past_v_{i}'] = np.zeros((1,8,256,128), np.float16)
ort_out = sess.run(None, feed)[0]
assert np.allclose(out_pt[0].numpy().astype(np.float16), ort_out, rtol=0.01, atol=0.1)
print('ORT vs PyTorch: PASS')
"

# 3. (可选) 验证: ORT 多步生成（K/V 缓存正确积累）
pixi run python -c "
import numpy as np, onnxruntime as ort
sess = ort.InferenceSession('om_out/qwen3_kvcache_max256.onnx')
kv = {}
for i in range(28):
    kv[f'k_{i}'] = np.zeros((1,8,256,128), np.float16)
    kv[f'v_{i}'] = np.zeros((1,8,256,128), np.float16)
# 3 步 prefill + 2 步 decode
for pos in range(5):
    feed = {'input_ids': np.array([[pos+100]], np.int64),
            'position': np.array([pos], np.int64)}
    for i in range(28):
        feed[f'past_k_{i}'] = kv[f'k_{i}']
        feed[f'past_v_{i}'] = kv[f'v_{i}']
    outs = sess.run(None, feed)
    for i in range(28):
        kv[f'k_{i}'] = outs[1 + 2*i]
        kv[f'v_{i}'] = outs[1 + 2*i + 1]
# 检查前 3 个位置有非零 K 值
assert np.count_nonzero(kv['k_0'][0,:,0,:]) > 0, 'Position 0 unfilled'
assert np.count_nonzero(kv['k_0'][0,:,2,:]) > 0, 'Position 2 unfilled'
print('Multi-step KV accumulation: PASS')
"

# 4. ATC 转 OM
INPUT_SHAPE=$(pixi run python -c "
import onnx
m = onnx.load('om_out/qwen3_kvcache_max256.onnx')
print(';'.join(i.name+':'+','.join(str(d.dim_value)
    for d in i.type.tensor_type.shape.dim) for i in m.graph.input))
")
MODEL_ONNX=om_out/qwen3_kvcache_max256.onnx \
INPUT_SHAPE="$INPUT_SHAPE" \
bash scripts/podman_convert.sh

# 5. 传输到开发板
sshpass -p 'Mind@123' scp om_out/qwen3_kvcache_max256.om \
    root@192.168.137.100:/root/slm_deploy/
```


### 准备开发板环境（离线恢复）

如果开发板刚重置、没有公网连接，只要板上出厂 Ascend runtime/pyACL 还在，就可以按下面流程离线恢复到可运行 KV Cache 模型的状态。
本项目实测的重置后环境是：Ubuntu 22.04 aarch64、Python 3.10.6、`npu-smi 23.0.rc3`、CANN Toolkit `7.0.RC1`。
虽然 OM 是用 CANN 8.0.RC3 编译的，但这个 `qwen3_kvcache_max256.om` 在该 CANN 7.0.RC1 runtime 上可以成功加载并执行；这只说明当前模型兼容，不代表所有 CANN 8 产物都能在 CANN 7 上运行。

#### 1. 板端基础检查

在**开发机**上执行：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
python3 --version
npu-smi info
find /usr/local/Ascend -maxdepth 5 -name set_env.sh -print
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl; print(\"acl OK\")"
'
```

期望结果：

- `npu-smi info` 能看到 `310B4`
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 后 `import acl` 成功
- 如果 `import acl` 失败，先修复或重装 Ascend runtime/Toolkit；Python 依赖无法替代 pyACL

#### 2. 准备 aarch64 离线 Python wheel

如果开发机已有 `tmp/*.whl` 和 `tmp/get-pip.py`，可直接使用。没有的话，在**有网络的开发机**上下载到 `tmp/`：

```bash
mkdir -p tmp
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 \
    --only-binary=:all: \
    "numpy==1.26.4" "transformers==4.53.3" \
    "tokenizers==0.21.4" "torch==2.1.0" \
    "safetensors" "huggingface-hub" "requests" "pyyaml" \
    "regex" "tqdm" "filelock" "fsspec" "packaging" \
    "typing-extensions" "sympy" "networkx" "jinja2"
curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
python3 -m pip download --dest tmp pip setuptools wheel
```

说明：

- 固定 `numpy==1.26.4`，避免 NumPy 2.x 和旧版 CANN/pyACL 组合出现兼容风险
- `torch` 在板端只用于 tokenizer/transformers 依赖链，不参与 NPU 推理
- 如果 `torch==2.1.0` 下载不到 aarch64 wheel，可换用本地已验证的 `torch-2.1.0-cp310-cp310-manylinux2014_aarch64.whl`

#### 3. 传输离线依赖并安装

在**开发机**上执行：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'

sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py \
    root@192.168.137.100:/root/slm_deploy/wheels/

sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
cd /root/slm_deploy
python3 wheels/get-pip.py --no-index --find-links=/root/slm_deploy/wheels \
    pip setuptools wheel
python3 -m pip install --no-index --find-links=/root/slm_deploy/wheels \
    --force-reinstall "numpy==1.26.4" transformers torch
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers, torch; print(numpy.__version__, transformers.__version__, torch.__version__)"
'
```

### 传输文件到开发板

以下操作在**开发机**上执行：

```bash
# Tokenizer 文件（Qwen3 BPE tokenizer）
sshpass -p 'Mind@123' scp \
    model/Qwen3-0.6B/tokenizer.json \
    model/Qwen3-0.6B/tokenizer_config.json \
    model/Qwen3-0.6B/vocab.json \
    model/Qwen3-0.6B/merges.txt \
    model/Qwen3-0.6B/config.json \
    model/Qwen3-0.6B/generation_config.json \
    root@192.168.137.100:/root/slm_deploy/

# 推理脚本
sshpass -p 'Mind@123' scp \
    board/gen_text_seq32.py \
    board/gen_text_kvcache.py \
    board/run_kvcache.sh \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 'chmod +x /root/slm_deploy/run_kvcache.sh'

# OM 模型文件（先确保已完成 ATC 转换，文件在 om_out/ 下）
sshpass -p 'Mind@123' scp om_out/qwen3_fp16_seq32_tile.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3_kvcache_max256.om \
    root@192.168.137.100:/root/slm_deploy/qwen3_kvcache_max256_b4.om
```

可选：校验大文件传输是否完整。

```bash
sha256sum om_out/qwen3_kvcache_max256.om
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 \
    'sha256sum /root/slm_deploy/qwen3_kvcache_max256_b4.om'
```

### 运行推理

在**开发板**上执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy

# seq=32 滑动窗口模型
python3 gen_text_seq32.py --prompt "你好，请介绍一下你自己" --max-tokens 50

# KV Cache 模型 (max_len=256)
python3 gen_text_kvcache.py --prompt "你好，请介绍一下你自己" --max-tokens 50

# 或使用封装好的入口
./run_kvcache.sh --prompt "你好，请介绍一下你自己" --max-tokens 50
```

首次运行前可做一个短冒烟测试：

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
    root@192.168.137.100 '
cd /root/slm_deploy
./run_kvcache.sh --prompt "你好" --max-tokens 2
'
```

实测输出应包含：

```text
I/O: 58 in, 57 out
[Prompt: 13 tokens]
[step 1] post-execute
...
你好！
Done.
```

注意：1.5 GB OM 首次 `acl.mdl.load_from_file` 可能需要约 75 秒，短时间无输出不一定是卡死。

**可选参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 各脚本内置默认路径 | OM 文件路径 |
| `--tokenizer-dir` | `/root/slm_deploy` | tokenizer 文件目录 |
| `--max-tokens` | 30 | 最大生成 token 数 |
| `--temperature` | 0.7 | 采样温度，0 为贪心解码 |
| `--top-k` | 40 | Top-K 采样 |
| `--top-p` | 0.9 | Nucleus 采样 |

**常见运行时问题**：

- **模型加载卡住**：NPU 可能有之前进程的残留状态，重启板子 (`reboot`)
- **输出乱码**：确认 prompt 不要超过模型窗口（seq=32 时最多 ~15 个中文字）。KV Cache 模型也有 256 token 上限
- **速度异常慢**：用 `npu-smi info` 检查 NPU 内存是否接近满载——如果 ~95% 说明重复加载了多个模型实例

---

*最后更新：2026-06-09*
