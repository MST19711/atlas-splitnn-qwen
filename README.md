# Qwen3 / Qwen3.5 在华为 Atlas 200I DK A2 上的部署

[English](./README_EN.md) | [实验报告](./REPORT.md)

将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 边缘计算板，实现 NPU 加速的端侧中文对话。

---

## 成果概览

| 方案 | 模型 | 上下文 | OM 大小 |
|------|------|--------|---------|
| Qwen3 KV Cache | Qwen3-0.6B | 256 tok | 1.5 GB |
| Qwen3.5 KV Cache | Qwen3.5-0.8B | 256 tok | 1.9 GB |
| Qwen3.5 KV Cache | Qwen3.5-0.8B | 1024 tok | 1.9 GB |
| **Qwen3.5 SplitNN** | **Qwen3.5-4B (1/30/1)** | **16K tok** | **Prefix+Suffix ~2.8 GB** |
| **Qwen3.5 SplitNN 参数绑定** | **Qwen3.5-2B (0/24/0)** | **8K tok** | **板端共享 tied weight 970MB + 单算子 head OM 14KB** |

> SplitNN 方案在开发机 ONNX 后端已联调通过，4B 模型 16K 上下文可稳定推理。板端 OM 部署待 ATC 编译后实测。
>
> 目前仓库已新增一条可实际部署到开发板的 `Qwen3.5-2B split 0/24/0` 参数绑定链路：板端提供 OpenAI 兼容控制器，主机承担全部 Transformer 主干层，板端仅执行 `embedding + tied lm_head`。

---

## 硬件与工具链

| 组件 | 说明 |
|------|------|
| 开发板 | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| 模型 | Qwen3-0.6B / Qwen3.5-0.8B / Qwen3.5-4B (FP16) |
| CANN | 7.0.0 (ATC 编译容器) / 7.0.RC1 (板端 runtime) |
| ONNX | opset 15, TorchScript 导出 |
| 容器 | Podman + Rocky Linux 9, 镜像 `cann-atc-rocky:v7` |
| Python | pixi 管理 (x86), pip (板端 aarch64) |

---

## 项目结构

```
├── model/                    # 模型权重 + tokenizer
├── scripts/                  # ONNX 导出 & ATC 转换 (x86 dev)
│   ├── qwen35_model_spec.py       # ModelSpec/SplitConfig (无 torch 依赖)
│   ├── qwen35_split_common.py     # SplitNN 共享代码（Wrappers + Patches）
│   ├── export_qwen35_split_prefix.py   # Prefix ONNX 导出（支持 --split）
│   ├── export_qwen35_split_suffix.py   # Suffix ONNX 导出（支持 --split）
│   ├── export_qwen35_bound_embed_head.py  # 导出参数绑定资产（tied weight + metadata）
│   ├── export_qwen3_kvcache.py         # Qwen3 KV Cache 导出
│   ├── export_qwen35_kvcache.py          # Qwen3.5 DeltaNet KV Cache 导出
│   ├── gen_input_shape.py        # ONNX → ATC INPUT_SHAPE 辅助
│   ├── podman_convert.sh         # 容器化 ATC 转换
├── board/                    # 板端推理 (aarch64)
│   ├── gen_text_qwen3_kvcache.py       # Qwen3 KV Cache 推理
│   ├── gen_text_qwen35_kvcache.py        # Qwen3.5 DeltaNet KV Cache 推理
│   ├── gen_text_qwen35_splitnn.py        # Qwen3.5 SplitNN 推理（复用 OmSplitEngine）
│   └── run_openai_split_controller_bound_2b.sh  # 板端 OpenAI 控制器启动脚本（2B 参数绑定）
│   └── run_qwen3_kvcache.sh
├── controller/               # SplitNN 控制器（OpenAI API + 可插拔前后段引擎）
│   ├── openai_split_controller.py   # FastAPI 入口
│   ├── orchestrator.py              # 消息编排、采样、生成循环
│   ├── remote_middle.py             # 与远端中段服务的 HTTP 协议
│   ├── schemas.py                   # Pydantic 数据模型 (含 enable_thinking)
│   └── engine/
│       ├── base.py                  # 引擎抽象基类
│       ├── onnx_engine.py           # ONNX Runtime 引擎
│       └── om_engine.py             # OM (NPU) 引擎
├── docker/                   # ATC 容器构建
│   └── Containerfile.v2-cann7
├── server/                   # CUDA 主机服务
│   └── qwen35_split_service.py      # 中段 HTTP 服务（支持 --split）
├── om_out/                   # ATC 编译产物 (*.om, *.onnx)
└── logs/                     # 编译日志
```

---

## 快速开始

### 新增：Qwen3.5-2B SplitNN 参数绑定链路

这条链路的目标是以尽可能小的板端改动支持参数绑定：

- 主机：
  - 运行 `Qwen3.5-2B` 的全部 24 层 Transformer 主干
  - 提供中段 HTTP 服务
- 开发板：
  - 运行 OpenAI 兼容控制器
  - `run_prefix()` 执行 embedding lookup
  - `run_suffix()` 执行 tied `lm_head`

控制器外部接口不变，仍然使用：

```text
POST /v1/chat/completions
GET  /v1/models
GET  /healthz
```

### 1. 准备 2B 参数绑定资产

在主机上导出板端需要的共享权重与元数据：

```bash
cd /home/CX_Li/EF_clean

pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-2B \
  --output-dir qwen3.5_2b_bound_embed_head \
  --split 0,24
```

导出目录至少包含：

```text
qwen3.5_2b_bound_embed_head/
├── tied_weight.bin
├── final_norm_weight.bin
└── bound_embed_head.metadata.json
```

### 2. 编译板端 `lm_head` 单算子模型

参数绑定模式下，板端 `run_suffix()` 默认可退回 CPU `numpy`，但为了可用速度，实际部署时应为真实 head shape 编译 ACL single-op `MatMul`：

```bash
mkdir -p tmp_singleop_matmul_qwen35_2b_head/run/out/test_data/config
cat > tmp_singleop_matmul_qwen35_2b_head/run/out/test_data/config/acl_op.json <<'EOF'
[
  {
    "op": "MatMul",
    "input_desc": [
      {"format": "ND", "type": "float16", "shape": [1, 2048]},
      {"format": "ND", "type": "float16", "shape": [248320, 2048]}
    ],
    "output_desc": [
      {"format": "ND", "type": "float16", "shape": [1, 248320]}
    ],
    "attr": [
      {"name": "transpose_x1", "type": "bool", "value": false},
      {"name": "transpose_x2", "type": "bool", "value": true}
    ]
  }
]
EOF
```

使用现有 CANN 7 容器编译：

```bash
podman run --rm --network=host --http-proxy=false \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= \
  -v /home/CX_Li/EF_clean:/workspace:Z \
  -w /workspace/tmp_singleop_matmul_qwen35_2b_head/run/out \
  localhost/cann-atc-rocky:v7 \
  bash -lc 'atc --singleop=test_data/config/acl_op.json \
    --soc_version=Ascend310B4 \
    --output=op_models'
```

编译结果示例：

```text
op_models/0_MatMul_1_2_1_2048_1_2_248320_2048_1_2_1_248320.om
```

### 3. 同步到开发板

```bash
sshpass -p 'Mind@123' ssh -F /dev/null -o StrictHostKeyChecking=no \
  root@192.168.137.100 'mkdir -p /root/slm_deploy/qwen3.5_2b_bound_embed_head/op_models'

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  qwen3.5_2b_bound_embed_head/* \
  root@192.168.137.100:/root/slm_deploy/qwen3.5_2b_bound_embed_head/

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  tmp_singleop_matmul_qwen35_2b_head/run/out/op_models/* \
  root@192.168.137.100:/root/slm_deploy/qwen3.5_2b_bound_embed_head/op_models/

sshpass -p 'Mind@123' scp -F /dev/null -o StrictHostKeyChecking=no \
  controller/openai_split_controller.py \
  controller/orchestrator.py \
  controller/remote_middle.py \
  controller/schemas.py \
  controller/engine/om_engine.py \
  scripts/qwen35_model_spec.py \
  board/run_openai_split_controller_bound_2b.sh \
  root@192.168.137.100:/root/slm_deploy/
```

如果板端目录结构已经和仓库同步过，只需要额外更新：

- `controller/engine/om_engine.py`
- `controller/openai_split_controller.py`
- `controller/orchestrator.py`
- `scripts/qwen35_model_spec.py`
- `board/run_openai_split_controller_bound_2b.sh`
- `qwen3.5_2b_bound_embed_head/`

### 4. 启动主机侧中段服务

```bash
cd /home/CX_Li/EF_clean

pixi run python server/qwen35_split_service.py \
  --host 127.0.0.1 \
  --port 18080 \
  --model-path model_dl/Qwen3.5-2B \
  --split 0,24 \
  --device cuda:0 \
  --max-len 8192 \
  --session-timeout-sec 60 \
  --max-sessions 1
```

推荐保留 `--max-sessions 1`。当前这条 `0/24/0` 参数绑定链路默认按“单开发板控制器 + 单主机中段服务”部署，额外并发只会放大 session cache 与 HTTP 请求处理开销。

再建立反向 SSH 映射，使开发板通过自身 `127.0.0.1:28080` 访问主机 `18080`：

```bash
sshpass -p 'Mind@123' ssh -F /dev/null \
  -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 \
  root@192.168.137.100
```

### 5. 启动开发板 OpenAI 控制器

登录开发板后执行：

```bash
cd /root/slm_deploy
./run_openai_split_controller_bound_2b.sh
```

后台运行：

```bash
cd /root/slm_deploy
nohup ./run_openai_split_controller_bound_2b.sh >/tmp/bound_controller.log 2>&1 &
tail -f /tmp/bound_controller.log
```

### 6. 手动测试 OpenAI 接口

健康检查：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/models
```

非流式：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "请用三句话介绍一下你自己"}
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": false
  }'
```

流式：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "你好，请简单和我打个招呼"}
    ],
    "temperature": 0.0,
    "max_tokens": 32,
    "stream": true
  }'
```

如果想观察纯 SSE 数据，可以再接一段：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "你好，请简单和我打个招呼"}
    ],
    "temperature": 0.0,
    "max_tokens": 32,
    "stream": true
  }' | sed -n 's/^data: //p'
```

开启 thinking：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-2b-split-0-24-0-om",
    "messages": [
      {"role": "user", "content": "请先认真思考，再简短回答：1+1为什么等于2？"}
    ],
    "temperature": 0.0,
    "max_tokens": 128,
    "stream": false,
    "enable_thinking": true
  }'
```

`Qwen3.5-2B` 当前部署默认不建议开启 thinking。实际联调中发现 2B thinking 模式容易出现长时间思维链循环、乱码和收尾失败；控制器当前会直接拒绝这一路径，建议显式使用 `enable_thinking=false`。

### 6.1 断连与清理行为

当前控制器已补充“请求方断连后自动取消”的处理：

- 流式请求如果前端程序崩溃、浏览器标签页关闭、或命令行 `Ctrl+C` 中断，控制器会在当前 token step 结束后停止生成
- 非流式请求如果客户端提前断开，也会设置取消标志并尽快退出生成循环
- 退出时会自动执行远端 `session/close`，并调用本地 `engine.end_session()`

需要注意的是，取消粒度仍是“当前 step 完成后”而不是立即抢占。也就是说，如果控制器正阻塞在一次 `remote_middle.step()` 中，仍需等这一小步返回后才能进入清理逻辑。

### 7. 当前状态

目前这条 `Qwen3.5-2B split 0/24/0` 参数绑定链路已经完成以下验证：

- 开发板控制器可正常启动，不再卡在 `Waiting for application startup`
- 非流式 OpenAI 兼容请求可正常返回文本
- 流式 OpenAI 兼容请求可完整返回 SSE chunk 与 `[DONE]`
- 客户端中途断开后，控制器不会再无限继续生成；会在当前 step 完成后自动关闭远端 session
- 板端实际生成路径为：
  - `run_prefix()`：共享 tied weight 的 embedding lookup
  - `remote_middle.step()`：主机侧全部 24 层 Transformer 主干
  - `run_suffix()`：开发板 ACL single-op `MatMul` 执行 tied `lm_head`
- 主机侧中段服务已补充以下保护：
  - session 超时回收与显式 cache 释放
  - `CUDA_OOM` 时主动回收故障 session
  - `max_sessions` 并发限制
  - 单线程 `HTTPServer`，避免 `/v1/session/step` 高频请求持续创建线程
- 主机侧注意力缓存更新已改为原地写回，不再每个 token 重建整块 K/V cache
- GQA 计算已移除“物理复制 KV 头”的实现，改为按组直接计算，降低系统内存与显存抖动

### 7.1 最近修复摘要

这次新增的修复主要集中在两条线上：

1. 控制器稳定性
   - 支持客户端断连后的自动取消与 session 清理
   - `Qwen3.5-2B` thinking 模式显式禁用，避免已知的长思维链循环问题

2. 主机侧中段服务内存稳定性
   - session 生命周期改为显式释放 cache，而不是仅依赖字典删除
   - `Qwen3_5Attention` 的 K/V 更新改为原地 `copy_()` 写回
   - GQA 不再把 2 个 KV heads 扩成 8 个 query heads 的物理副本
   - HTTP 服务从 `ThreadingHTTPServer` 改为 `HTTPServer`，避免每 token step 产生线程级系统内存膨胀

### 1. 环境准备

#### 开发机 (x86_64)

```bash
cd EF && pixi install

# 下载模型权重
hf download Qwen/Qwen3-0.6B --local-dir model/Qwen3-0.6B
hf download Qwen/Qwen3.5-0.8B --local-dir model/Qwen3.5-0.8B

# 下载 CANN 7.0.0 安装包到 docker/ 目录
wget -O docker/Ascend-cann-toolkit_7.0.0_linux-x86_64.run \   # ATC 编译器 (~1.6GB)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-toolkit_7.0.0_linux-x86_64.run"
wget -O docker/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm \  # 310B 内核 (~351MB, 必须用 310B 而非 310P)
  "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.0/Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm"

# 构建 ATC 容器（需 --network=host 确保容器内 dnf/pip 可联网）
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

#### 下载板端 Python wheel（在有网络的开发机上执行）

板端推理脚本实际只依赖 `numpy` + `transformers`。**必须用 `--no-deps` 分别下载**，原因见下方说明。

```bash
mkdir -p tmp

# 主要依赖
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "numpy==1.26.4" "transformers==4.57.6" "tokenizers==0.22.2" \
    "huggingface_hub==0.34.6" "safetensors" "requests" "pyyaml" "regex" "tqdm" \
    "filelock" "fsspec" "packaging" "typing-extensions" \
    "certifi" "charset-normalizer" "idna" "urllib3"

# jinja2 + markupsafe：apply_chat_template(enable_thinking=False) 需要
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "markupsafe"
python3 -m pip download --dest tmp --platform any \
    --python-version 310 --only-binary=:all: --no-deps \
    "jinja2"

# SplitNN 控制器额外依赖（板端 OpenAI API）
mkdir -p tmp/board_controller_wheels
python3 -m pip download --dest tmp/board_controller_wheels --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "fastapi" "uvicorn" "pydantic" "pydantic-core" "starlette" \
    "annotated-types" "typing-inspection" "anyio" "h11" "click" \
    "sniffio" "exceptiongroup"

# pip 自举：板端出厂无 pip，且板端 pip.conf 指向不可达的豆瓣镜像
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "pip" "setuptools" "wheel"

curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

> **为什么 `--no-deps`？** `huggingface-hub>=0.34` 在 aarch64 平台要求 `hf-xet>=1.1.3`，但 PyPI 上 `hf-xet` 的 aarch64 wheel 最高只有 `0.1.x`。`hf-xet` 仅用于 HuggingFace Hub 并行下载加速，板端只通过 `AutoTokenizer.from_pretrained(local_dir)` 读取本地 tokenizer 文件，以 `--no-deps` 安装 huggingface-hub 安全可行。

### 2. 导出 ONNX → 编译 OM

```bash
# 导出 ONNX（可自定义 --max-len）
pixi run python scripts/export_qwen3_kvcache.py --max-len 256 \
    --output om_out/qwen3_kvcache_max256.onnx

pixi run python scripts/export_qwen35_kvcache.py --max-len 256 \
    --output om_out/qwen3.5_kvcache_max256.onnx

# ATC 编译（统一命令，自动读取 ONNX 的 input shape）
# 注意：INPUT_SHAPE 值包含分号，必须先 export 再运行，不能内联展开
INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/qwen3.5_kvcache_max256.onnx)
export INPUT_SHAPE MODEL_ONNX="om_out/qwen3.5_kvcache_max256.onnx" OUTPUT_PREFIX="om_out/qwen3.5_kvcache_max256"
bash scripts/podman_convert.sh
```

可选参数：

| 参数 | 说明 |
|------|------|
| `--max-len` | KV Cache 上下文长度（默认 256，可选 128/512/1024 等） |
| `--output` | ONNX 输出路径 |
| `IMAGE=` | ATC 容器镜像（默认 `cann-atc-rocky:v7`） |
| `OUTPUT_PREFIX=` | OM 输出路径前缀 |

### 3. 板端环境

开发板为 Ubuntu 22.04 aarch64，出厂预装 CANN 7.0.RC1。板端 `/root/.pip/pip.conf` 配置了豆瓣镜像（无外网不可达），需用 pip wheel 自举安装。

#### 传输文件

```bash
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/wheels'
sshpass -p 'Mind@123' scp tmp/*.whl tmp/get-pip.py root@192.168.137.100:/root/slm_deploy/wheels/
sshpass -p 'Mind@123' ssh root@192.168.137.100 'mkdir -p /root/slm_deploy/board_controller_wheels'
sshpass -p 'Mind@123' scp tmp/board_controller_wheels/*.whl \
    root@192.168.137.100:/root/slm_deploy/board_controller_wheels/
```

#### 在开发板上安装

```bash
cd /root/slm_deploy/wheels

# 1. 用 pip wheel 自举安装 pip（get-pip.py 因镜像不可达会失败）
python3 -c "
import zipfile, sys, os
whl = [f for f in os.listdir('.') if f.startswith('pip-')][0]
zf = zipfile.ZipFile(whl)
zf.extractall('/tmp/_pip')
sys.path.insert(0, '/tmp/_pip')
import pip._internal
pip._internal.main(['install', '--no-deps', '--no-index', '--force-reinstall', whl])
"

# 2. 基础库（无上层依赖）
python3 -m pip install --no-deps --no-index --find-links=. \
    numpy-*.whl typing_extensions-*.whl packaging-*.whl filelock-*.whl \
    fsspec-*.whl tqdm-*.whl regex-*.whl safetensors-*.whl

# 3. huggingface-hub（--no-deps 跳过 hf-xet）
python3 -m pip install --no-deps --no-index --find-links=. \
    huggingface_hub-0.34.6-*.whl

# 4. requests → tokenizers → transformers
python3 -m pip install --no-deps --no-index --find-links=. \
    requests-*.whl charset_normalizer-*.whl tokenizers-*.whl transformers-*.whl

# 5. jinja2 + markupsafe（chat template 模板渲染需要）
python3 -m pip install --no-deps --no-index --find-links=. \
    markupsafe-*.whl jinja2-*.whl

# 6. 放宽 tokenizers 版本上限
sed -i 's/tokenizers>=0.22.0,<=0.23.0/tokenizers>=0.22.0,<=0.23.1/' \
  /usr/local/lib/python3.10/dist-packages/transformers/dependency_versions_table.py

# 7. 验证
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -c "import acl, numpy, transformers, tokenizers, huggingface_hub, safetensors, jinja2; print('OK')"

# 8. SplitNN 控制器额外依赖
cd /root/slm_deploy/board_controller_wheels
python3 -m pip install --no-deps --no-index --find-links=. \
    annotated_types-*.whl exceptiongroup-*.whl sniffio-*.whl anyio-*.whl \
    click-*.whl h11-*.whl typing_inspection-*.whl pydantic_core-*.whl \
    pydantic-*.whl starlette-*.whl uvicorn-*.whl fastapi-*.whl

# 9. 控制器依赖验证
python3 -c "import fastapi, uvicorn, pydantic; print('controller deps OK')"
```

> **关于 `hf-xet`**：`huggingface-hub>=0.34` 在 aarch64 上依赖 `hf-xet>=1.1.3`，但 PyPI 上 `hf-xet` 仅 `0.1.x` 提供 aarch64 wheel。`hf-xet` 仅用于 HuggingFace Hub 并行下载，板端只通过 `from_pretrained(local_dir)` 读本地 tokenizer 文件，不会触发下载路径。

可选：关闭桌面服务释放 ~120MB 内存：

```bash
systemctl stop sddm && systemctl disable sddm
pkill -f xfce4-power-manager
pkill -f xfce4-notifyd
pkill -f tumblerd
```

### 4. 部署并运行

从开发机传输文件：

```bash
# Qwen3 KV Cache
scp om_out/qwen3_kvcache_max256_cann7.om root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen3_kvcache.py          root@192.168.137.100:/root/slm_deploy/

# Qwen3.5 KV Cache（tokenizer 与 Qwen3 不兼容，需单独传输）
scp om_out/qwen3.5_kvcache_max256.om         root@192.168.137.100:/root/slm_deploy/
scp board/gen_text_qwen35_kvcache.py          root@192.168.137.100:/root/slm_deploy/
scp model/Qwen3.5-0.8B/tokenizer.json model/Qwen3.5-0.8B/tokenizer_config.json \
    model/Qwen3.5-0.8B/chat_template.jinja     root@192.168.137.100:/root/slm_deploy/
```

> **tokenizer 兼容性**：Qwen3 使用 `vocab.json` + `merges.txt`，Qwen3.5 使用 `tokenizer.json`。两者互不兼容，部署时注意不要互相覆盖。

在开发板上运行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy

# Qwen3 KV Cache
python3 gen_text_qwen3_kvcache.py \
    --model qwen3_kvcache_max256_cann7.om --prompt "你好" --max-tokens 50

# Qwen3.5 KV Cache
python3 gen_text_qwen35_kvcache.py \
    --model qwen3.5_kvcache_max256.om --tokenizer-dir /root/slm_deploy \
    --prompt "你好" --max-tokens 50
```

## SplitNN 通用化架构

SplitNN 体系现已从硬编码的 `4/16/4` 切分演进为**参数化架构**，支持：

- **任意 Qwen3.5 模型尺寸**（0.8B / 2B / 4B / 9B / 27B）
- **自定义上下文长度**（256 / 16K / 任意）
- **自定义切分方案**（通过 `--split prefix_end,suffix_start` 指定）
- **Thinking 开关**（`enable_thinking` 选项透传至 chat template）

### 核心组件

| 组件 | 路径 | 作用 |
|------|------|------|
| `ModelSpec` | `scripts/qwen35_model_spec.py` | 从 `config.json` 动态读取所有架构参数 |
| `SplitConfig` | `scripts/qwen35_model_spec.py` | 切分点参数化，自动计算各段 DN/GA 层数 |
| `metadata.json` | 导出时附带 | 板端无 PyTorch，通过 JSON 获取模型参数 |

### 使用 `--split`

```bash
# 0.8B 经典 4/16/4 切分（默认）
pixi run python scripts/export_qwen35_split_prefix.py --split 4,20 --max-len 256

# 4B 模型，前后各 1 层（板端内存最优）
pixi run python scripts/export_qwen35_split_prefix.py \
  --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384
```

## SplitNN 原型（板端 + CUDA 主机）

首版 SplitNN 使用 `4 / 16 / 4` 切分（Qwen3.5-0.8B），现扩展为通用参数化体系：
- 开发板运行前段和后段（两个 OM 或 ONNX）
- CUDA 主机运行中间段（PyTorch）
- 板端与主机通过 `HTTP/1.1 + application/octet-stream` 传输 `(1,1,hidden_size)` 的 `fp16 hidden state`

### 设计动机

引入 SplitNN 的原因不是单纯为了"分布式"，而是为了把不同硬件各自擅长的部分拆开：

- 开发板负责前后段，尽量贴近最终端侧部署形态
- 主机负责中间段的大部分计算量，减轻板端算力和内存压力
- 两端之间只传输单步 hidden state，而不传完整 cache，从而把网络负担控制在每 token 约 `hidden_size × 2` 字节往返

对 Qwen3.5 而言，`full_attention_interval=4` 的周期结构使得切分可以灵活选择边界——`4 / 16 / 4`（0.8B 经典）、`1 / 30 / 1`（4B 内存最优）等多种方案均可。

### 系统职责划分

SplitNN 中一共分成三部分：

1. **前段（prefix）**
   - 输入：`token_id + position + 前段 cache`
   - 输出：`hidden_state`
   - 运行位置：开发板 OM 或开发机 ONNX

2. **中段（middle）**
   - 输入：`hidden_state + position + 中段 cache`
   - 输出：`hidden_state`
   - 运行位置：主机 `server/qwen35_split_service.py`

3. **后段（suffix）**
   - 输入：`hidden_state + position + 后段 cache`
   - 输出：`logits`
   - 运行位置：开发板 OM 或开发机 ONNX

其中：
- **前后段 cache** 留在本地执行端
- **中段 cache** 留在远端 server
- 网络上传输的只有 `(1, 1, hidden_size)` 的 `fp16 hidden state`

### 原型验证状态

在引入控制器之前，原始 SplitNN 原型已经完成了三层验证：

1. **纯 PyTorch reference 对齐**
   - `prefix -> middle -> suffix`
   - 与完整 `Qwen3.5 KV Cache` 模型逐 token 对齐

2. **前后段 ONNX 多步 ORT 校验**
   - 验证 prefix hidden 和 suffix logits 的数值正确性

3. **本地模拟联调**
   - `ONNX prefix/suffix + middle server`
   - 已经可以按真实 prompt 生成并解码出正常中文文本

因此，控制器并不是从零开始设计，而是建立在“SplitNN 原型本身已经可工作”的基础上。

### 1. 导出前段 / 后段 ONNX

```bash
# 0.8B 经典 4/16/4 切分（默认）
pixi run python scripts/export_qwen35_split_prefix.py \
    --max-len 256 --output om_out/qwen3.5_split_prefix_max256.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --max-len 256 --output om_out/qwen3.5_split_suffix_max256.onnx

# 4B 模型 1/30/1 切分 + 16K 上下文
pixi run python scripts/export_qwen35_split_prefix.py \
    --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384 \
    --output om_out/qwen3.5_4b_split_prefix_max16384.onnx

pixi run python scripts/export_qwen35_split_suffix.py \
    --model-path model_dl/Qwen3.5-4B --split 1,31 --max-len 16384 \
    --output om_out/qwen3.5_4b_split_suffix_max16384.onnx
```

本地参考链路校验：

```bash
pixi run python scripts/validate_qwen35_split_reference.py
```

导出后 ORT 多步校验：

```bash
pixi run python scripts/validate_qwen35_split_ort.py \
    --prefix-onnx om_out/qwen3.5_split_prefix_max256.onnx \
    --suffix-onnx om_out/qwen3.5_split_suffix_max256.onnx
```

然后分别用现有 `scripts/gen_input_shape.py` + `scripts/podman_convert.sh` 编译成 `.om`。

### 2. 启动 CUDA 主机服务

```bash
# 0.8B 经典 4/16/4
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model/Qwen3.5-0.8B \
    --device cuda:0 --max-len 16384

# 4B 1/30/1 切分 + 16K 上下文
pixi run python server/qwen35_split_service.py \
    --host 0.0.0.0 --port 18080 \
    --model-path model_dl/Qwen3.5-4B \
    --split 1,31 --device cuda:0 --max-len 16384
```

健康检查：

```bash
curl http://<server-ip>:18080/v1/health
```

### 3. 开发板运行 SplitNN

将以下文件传到板端 `/root/slm_deploy/`：
- `om_out/qwen3.5_split_prefix_max256.om`
- `om_out/qwen3.5_split_suffix_max256.om`
- `om_out/qwen3.5_split_prefix_max16384.om`
- `om_out/qwen3.5_split_suffix_max16384.om`
- `board/gen_text_qwen35_splitnn.py`
- `board/run_qwen35_splitnn.sh`
- `board/run_qwen35_splitnn_16k.sh`
- `model/Qwen3.5-0.8B/tokenizer.json`
- `model/Qwen3.5-0.8B/tokenizer_config.json`
- `model/Qwen3.5-0.8B/chat_template.jinja`

短上下文（256）运行：

```bash
python3 -u /root/slm_deploy/gen_text_qwen35_splitnn.py \
    --server-url http://<server-ip>:18080 \
    --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max256.om \
    --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max256.om
```

长上下文（16K）运行：

```bash
cd /root/slm_deploy
./run_qwen35_splitnn_16k.sh
```

> `run_qwen35_splitnn_16k.sh` 默认使用 `qwen3.5_split_prefix_max16384.om`、`qwen3.5_split_suffix_max16384.om` 和 `--max-len 16384`。

## SplitNN 控制器（OpenAI API）

在原始 SplitNN 原型之上，仓库额外提供了一个“控制器中间层”：
- 对外提供 OpenAI 兼容的 `/v1/chat/completions`
- 对内管理 tokenizer、chat template、采样、生成循环
- 前后段通过统一引擎接口切换：
  - `OnnxSplitEngine`：开发机仿真
  - `OmSplitEngine`：开发板部署
- 中间 16 层继续通过 `server/qwen35_split_service.py` 远端执行

### 控制器架构

```
OpenAI Client
    |
    v
controller/openai_split_controller.py
    |
    +-- orchestrator.py      # messages -> prompt -> prefill/decode -> OpenAI 响应
    +-- remote_middle.py     # 与 middle server 的 open/step/close 协议
    +-- engine/onnx_engine.py
    `-- engine/om_engine.py
```

### 控制器特点

- **无状态多轮**：每次请求都基于完整 `messages` 重新 prefill，不跨请求复用 cache
- **前后段 cache**：由本地引擎实例内部管理
- **中段 cache**：由远端 middle server 按 `session_id` 管理
- **流式输出**：支持 `stream=true` 的 SSE 响应
- **Thinking 开关**：支持 `enable_thinking` 参数控制思考模式
- **灵活切分**：通过 `--split` / `--model-path` 适配不同模型尺寸和切分方案

### 开发机启动（ONNX 后端）

先启动 middle server：

```bash
# 4B 1/30/1 长上下文
pixi run python server/qwen35_split_service.py \
    --host 127.0.0.1 --port 18080 \
    --model-path model_dl/Qwen3.5-4B \
    --split 1,31 --device cuda:0 --max-len 16384
```

再启动控制器：

```bash
# 4B 1/30/1 + 16K ONNX 后端
pixi run python controller/openai_split_controller.py \
    --host 127.0.0.1 --port 8000 \
    --engine onnx \
    --model-path model_dl/Qwen3.5-4B \
    --model-name qwen3.5-4b-split-1-30-1-onnx \
    --remote-model-name "Qwen3.5-2560B-split-1-30-1" \
    --split 1,31 --max-len 16384 \
    --server-url http://127.0.0.1:18080 \
    --prefix-onnx om_out/qwen3.5_4b_split_prefix_max16384.onnx \
    --suffix-onnx om_out/qwen3.5_4b_split_suffix_max16384.onnx
```

### 开发板启动（OM 后端）

#### 板端文件布局

板端 `/root/slm_deploy/` 需按以下结构组织，确保模块间相互引用正确：

```
/root/slm_deploy/
├── gen_text_qwen35_splitnn.py     # 板端推理入口（独立脚本）
├── scripts/                        # Python 包（需 __init__.py）
│   ├── __init__.py
│   └── qwen35_model_spec.py       # ModelSpec/SplitConfig/load_metadata
├── controller/
│   ├── __init__.py
│   └── engine/
│       ├── __init__.py
│       ├── base.py                 # SplitEngine 抽象基类
│       └── om_engine.py            # OmSplitEngine（ACL NPU 推理）
├── qwen3.5_split_prefix_max256.om         # OM 模型文件
├── qwen3.5_split_prefix_max256.metadata.json   # 配套元数据
├── qwen3.5_split_suffix_max256.om
├── qwen3.5_split_suffix_max256.metadata.json
├── tokenizer.json                 # Qwen3.5 tokenizer（与 Qwen3 不兼容）
├── tokenizer_config.json
└── chat_template.jinja
```

**关键引用链：**
- `gen_text_qwen35_splitnn.py` → `qwen35_model_spec.load_metadata()` → 从 `.metadata.json` 读取模型参数（无需 PyTorch）
- `gen_text_qwen35_splitnn.py` → `controller.engine.om_engine.OmSplitEngine` → ACL 管理 .om 文件
- `controller/engine/base.py` → `scripts.qwen35_model_spec.ModelSpec`（需 `scripts/__init__.py`）

**注意：**
- `scripts/`、`controller/`、`controller/engine/` 都必须有 `__init__.py`（可为空文件），否则 Python 包导入失败
- `.metadata.json` 文件名需与对应 `.om` 前缀一致（如 `qwen3.5_split_prefix_max256.om` → `...metadata.json`）
- 首次运行前需 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 ACL 环境

#### 传输文件

```bash
# 核心代码
sshpass -p 'Mind@123' scp board/gen_text_qwen35_splitnn.py \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp scripts/qwen35_model_spec.py \
    root@192.168.137.100:/root/slm_deploy/scripts/
sshpass -p 'Mind@123' scp controller/__init__.py \
    root@192.168.137.100:/root/slm_deploy/controller/
sshpass -p 'Mind@123' scp controller/engine/{__init__.py,base.py,om_engine.py} \
    root@192.168.137.100:/root/slm_deploy/controller/engine/

# OM 模型 + 元数据
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_prefix_max256.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_prefix_max256.metadata.json \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_suffix_max256.om \
    root@192.168.137.100:/root/slm_deploy/
sshpass -p 'Mind@123' scp om_out/qwen3.5_split_suffix_max256.metadata.json \
    root@192.168.137.100:/root/slm_deploy/

# Tokenizer
sshpass -p 'Mind@123' scp model/Qwen3.5-0.8B/{tokenizer.json,tokenizer_config.json,chat_template.jinja} \
    root@192.168.137.100:/root/slm_deploy/
```

#### 建立网络隧道并运行

如果开发板无法直接访问主机 `18080` 端口，先在开发机建立反向隧道：

```bash
# 开发机执行（将板端 28080 转发到本机 18080）
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no -N \
  -R 28080:127.0.0.1:18080 root@192.168.137.100 &
```

板端运行（通过隧道 `127.0.0.1:28080` 访问中间服务）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /root/slm_deploy
python3 gen_text_qwen35_splitnn.py \
    --server-url http://127.0.0.1:28080 \
    --prefix-model /root/slm_deploy/qwen3.5_split_prefix_max256.om \
    --suffix-model /root/slm_deploy/qwen3.5_split_suffix_max256.om \
    --tokenizer-dir /root/slm_deploy \
    --prompt "你好" --max-tokens 50
```

### 当前验证状态

- `4 / 16 / 4` SplitNN 原型本身已通过 reference / ORT / 本地模拟三层验证
- **SplitNN 通用化完成**：通过 `ModelSpec` + `SplitConfig` 支持任意模型尺寸和切分方案
- **4B 模型 1/30/1 切分 16K 上下文**本地 ONNX 联调通过（非流式 + 流式 + 多轮 + thinking）
- **端边协同实测数据**：

| 模型 | 切分 | 上下文 | Prefill | Decode | 说明 |
|------|------|--------|---------|--------|------|
| Qwen3.5-0.8B | 4/16/4 | 256 | 13 tok / 3.8s | 1.8 tok/s | 短问答 + thinking |
| Qwen3.5-4B | 1/30/1 | 16K | 13 tok / 63s | 0.3 tok/s | 板端 1GA层 是瓶颈 |
| Qwen3.5-4B | 1/30/1 | 16K | 319 tok | — | 长上下文 ✓ |
| **Qwen3.5-4B** | **0/32/0** | **16K** | **13 tok / 5.1s** | **1.2 tok/s** | **GA offload → 4x 提速** |
| **Qwen3.5-4B** | **0/32/0** | **16K** | **419 tok** | — | 长上下文 ✓ |

- **0/32/0 切分**：板端仅 embedding + lm_head（无任何注意力层），所有 32 层 GA/DN 在主机 GPU 执行，decode 从 0.3 提升到 1.2 tok/s
- 瓶颈已从板端 NPU GA 注意力转移到服务器 GPU 全模型推理 + 网络延迟
- **板端脚本**已重构为复用 `OmSplitEngine`，通过 `metadata.json` 获取模型参数
- `max_len=16384` 的 CUDA middle server 已完成实际测速，单 token 中段吞吐约 `9 tok/s`
- 开发板 OM 后端的 SSE 流式收尾仍待继续排查，当前建议优先使用非流式接口做板端联调

两种推理脚本：

| 脚本 | 对应模型 | 关键参数 |
|------|---------|---------|
| `gen_text_qwen3_kvcache.py` | Qwen3 KV Cache | `--model X.om --prompt` |
| `gen_text_qwen35_kvcache.py` | Qwen3.5 KV Cache | `--model X.om --tokenizer-dir /root/slm_deploy` |

---

## 相关文档

- [REPORT.md](./REPORT.md) — 详细实验报告（设计思路、踩坑记录、性能优化）
- [AGENTS.md](./AGENTS.md) — AI 辅助开发速查
