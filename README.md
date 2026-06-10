# Qwen3 / Qwen3.5 在华为 Atlas 200I DK A2 上的部署

[English](./README_EN.md) | [实验报告](./REPORT.md)

将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 边缘计算板，实现 NPU 加速的端侧中文对话。

---

## 成果概览

| 方案 | 上下文 | prefill | decode | OM 大小 |
|------|--------|---------|--------|---------|
| Qwen3 静态窗口 | 32 tok | — | 3.6 tok/s | 1.5 GB |
| Qwen3 KV Cache | 256 tok | ~5.6s | 3.6 tok/s | 1.5 GB |
| Qwen3.5 KV Cache | 256 tok | ~52s | 3.7 tok/s | 1.9 GB |
| Qwen3.5 KV Cache | 1024 tok | ~67s | 3.6 tok/s | 1.9 GB |

---

## 硬件与工具链

| 组件 | 说明 |
|------|------|
| 开发板 | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| 模型 | Qwen3-0.6B / Qwen3.5-0.8B (FP16) |
| CANN | 7.0.0 (ATC 编译容器) / 7.0.RC1 (板端 runtime) |
| ONNX | opset 15, TorchScript 导出 |
| 容器 | Podman + Rocky Linux 9, 镜像 `cann-atc-rocky:v7` |
| Python | pixi 管理 (x86), pip (板端 aarch64) |

---

## 项目结构

```
├── model/                    # 模型权重 + tokenizer
├── scripts/                  # ONNX 导出 & ATC 转换 (x86 dev)
│   ├── export_qwen3_static.py        # Qwen3 静态窗口导出
│   ├── export_qwen3_kvcache.py     # Qwen3 KV Cache 导出
│   ├── export_qwen35_kvcache.py      # Qwen3.5 DeltaNet KV Cache 导出
│   ├── patch_qwen3_static_onnx.py         # GQA Expand→Tile 修补 (静态窗口专用)
│   ├── gen_input_shape.py    # ONNX → ATC INPUT_SHAPE 辅助
│   ├── podman_convert.sh     # 容器化 ATC 转换
│   └── download_qwen3.py     # HF 模型下载
├── board/                    # 板端推理 (aarch64)
│   ├── gen_text_qwen3_static.py     # 静态窗口推理
│   ├── gen_text_qwen3_kvcache.py   # Qwen3 KV Cache 推理
│   ├── gen_text_qwen35_kvcache.py    # Qwen3.5 DeltaNet KV Cache 推理
│   └── run_qwen3_kvcache.sh
├── docker/                   # ATC 容器构建
│   └── Containerfile.v2-cann7
├── om_out/                   # ATC 编译产物 (*.om)
└── logs/                     # 编译日志
```

---

## 快速开始

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

# pip 自举：板端出厂无 pip，且板端 pip.conf 指向不可达的豆瓣镜像
python3 -m pip download --dest tmp --platform manylinux2014_aarch64 \
    --python-version 310 --implementation cp --abi cp310 --only-binary=:all: --no-deps \
    "pip" "setuptools" "wheel"

curl -L https://bootstrap.pypa.io/get-pip.py -o tmp/get-pip.py
```

> **为什么 `--no-deps`？** `huggingface-hub>=0.34` 在 aarch64 平台要求 `hf-xet>=1.1.3`，但 PyPI 上 `hf-xet` 的 aarch64 wheel 最高只有 `0.1.x`。`hf-xet` 仅用于 HuggingFace Hub 并行下载加速，板端只通过 `AutoTokenizer.from_pretrained(local_dir)` 读取本地 tokenizer 文件，以 `--no-deps` 安装 huggingface-hub 安全可行。

### 2. 导出 ONNX → 编译 OM

```bash
# 导出 ONNX（三选一，可自定义 --max-len）
pixi run python scripts/export_qwen3_static.py \
    --output om_out/qwen3_seq32.onnx

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

> 静态窗口模型编译前需先运行 `patch_qwen3_static_onnx.py`（修复 GQA Expand 节点）：
> ```bash
> pixi run python scripts/patch_qwen3_static_onnx.py om_out/qwen3_seq32.onnx
> ```

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

三种推理脚本：

| 脚本 | 对应模型 | 关键参数 |
|------|---------|---------|
| `gen_text_qwen3_static.py` | Qwen3 静态窗口 | `--prompt`, `--max-tokens` |
| `gen_text_qwen3_kvcache.py` | Qwen3 KV Cache | `--model X.om --prompt` |
| `gen_text_qwen35_kvcache.py` | Qwen3.5 KV Cache | `--model X.om --tokenizer-dir /root/slm_deploy` |

---

## 相关文档

- [REPORT.md](./REPORT.md) — 详细实验报告（设计思路、踩坑记录、性能优化）
- [AGENTS.md](./AGENTS.md) — AI 辅助开发速查
