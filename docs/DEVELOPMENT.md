# 开发指南

环境配置、工具链、导出流程与常用命令。

---

## 开发板

- IP: `192.168.137.100`, 用户: `root`, 密码: `Mind@123`
- 型号: Atlas 200I DK A2 (Ascend310B4, 4GB NPU)
- 出厂预装: `acl` (CANN 7.0.RC1 runtime)
- 需离线安装: `numpy`, `transformers`, `tokenizers`, `huggingface-hub`, `jinja2`, `markupsafe`, `fastapi`, `uvicorn`, `pydantic`
- NPU 进程被 kill 后驱动不清理 → 重启板子

### SSH / SCP

```bash
# SSH
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 '<cmd>'

# SCP
sshpass -p 'Mind@123' scp <local> root@192.168.137.100:/root/slm_deploy/
```

---

## Python 环境 (x86 开发机)

- pixi 管理: `pixi run python <script>`
- 添加包: `pixi add <pkg>` (conda), `pixi add --pypi <pkg>` (pip)
- `pixi.toml` 在项目根

当前依赖：

```toml
[dependencies]
python = "3.10.*"
transformers = ">=5.10.2,<6"
onnx = ">=1.21.0,<2"
onnxruntime = ">=1.26.0,<2"
numpy = ">=2.2.6,<3"
safetensors = ">=0.7.0,<0.8"

[pypi-dependencies]
torch = "==2.11.0+cu128"
onnxsim = ">=0.6.4"
fastapi = ">=0.116.1"
uvicorn = ">=0.35.0"
pydantic = ">=2.11.7"
```

---

## ATC 转换

```bash
MODEL_ONNX=om_out/model.onnx \
INPUT_SHAPE="name1:d1,d2;name2:d1,d2" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh
```

- 镜像: `cann-atc-rocky:v7`, CANN 7.0 (实际 7.1.0.3.220)
- `soc_version=Ascend310B4`
- 需传入 `INPUT_SHAPE`, `MODEL_ONNX`, 可选 `OUTPUT_PREFIX`, `IMAGE`

**注意**: `INPUT_SHAPE` 包含分号分隔的多个 shape 定义，shell 内联展开会把分号当命令分隔符。必须 `export INPUT_SHAPE` 后运行，不能内联。

---

## ATC 容器构建

下载 CANN 7.0.0 安装包到 `docker/`：

1. toolkit: `Ascend-cann-toolkit_7.0.0_linux-x86_64.run` (~1.6GB)
2. kernel: `Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm` (~351MB)

```bash
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

---

## 导出 ONNX

### Qwen3 KV Cache

```bash
pixi run python scripts/export_qwen3_kvcache.py \
  --max-len 256 --output om_out/qwen3_kvcache_max256.onnx
```

### Qwen3.5 KV Cache

```bash
pixi run python scripts/export_qwen35_kvcache.py \
  --max-len 256 --output om_out/qwen3.5_kvcache_max256.onnx
```

### SplitNN

```bash
# 前缀
pixi run python scripts/export_qwen35_split_prefix.py \
  --model-path model/Qwen3.5-0.8B --max-len 16384 --split 4,20 \
  --output om_out/qwen3.5_split_prefix_max16384.onnx

# 后缀
pixi run python scripts/export_qwen35_split_suffix.py \
  --model-path model/Qwen3.5-0.8B --max-len 16384 --split 4,20 \
  --output om_out/qwen3.5_split_suffix_max16384.onnx
```

### 参数绑定

```bash
pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-2B \
  --output-dir qwen3.5_2b_bound_embed_head \
  --split 0,24 --compile-op
```

---

## 验证

### KV Cache ORT 多步验证

```bash
pixi run python scripts/validate_qwen35_kvcache_ort.py \
  om_out/qwen3.5_kvcache_max256.onnx
```

### SplitNN ORT 多步验证

```bash
pixi run python scripts/validate_qwen35_split_ort.py \
  --model-path model/Qwen3.5-0.8B --split 4,20 --max-len 256
```

### SplitNN 参考验证 (vs full model)

```bash
pixi run python scripts/validate_qwen35_split_reference.py \
  --model-path model/Qwen3.5-0.8B --split 4,20 --max-len 256
```

---

## 模型

| 模型 | 目录 | 说明 |
|------|------|------|
| Qwen3-0.6B | `model/Qwen3-0.6B/` | 28 层 GQA |
| Qwen3.5-0.8B | `model/Qwen3.5-0.8B/` | 24 层 DeltaNet+GQA |
| Qwen3.5-2B | `model_dl/Qwen3.5-2B/` | 24 层，2048 hidden |
| Qwen3.5-4B | `model_dl/Qwen3.5-4B/` | 32 层，2560 hidden |

模型权重在 `.gitignore` 中，需单独下载。

---

## 运行测试

```bash
pixi run python -m pytest tests/ -v
```
