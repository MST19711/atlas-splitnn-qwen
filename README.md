# Qwen3 / Qwen3.5 在华为 Atlas 200I DK A2 上的部署

[English](./README_EN.md)

将通义千问小语言模型部署到华为昇腾 Atlas 200I DK A2 边缘计算板，实现 NPU 加速的端侧中文对话。

---

## 三种引擎模式

| 方案 | 当前已测试组合 | 上下文（已测试） | 板端执行（示例） | 外部依赖 | 适用场景 |
|------|------|--------|----------|----------|----------|
| **KV Cache 纯板端** | 0.8B | 256/4096 tok | 全部 Transformer 层 | 无 | 入门、独立部署 |
| **SplitNN OM** | 0.8B / 4B | 8K / 16K tok | 前后若干层 + tokenizer/head | CUDA 主机 + SSH | 大模型、长上下文 |
| **SplitNN 参数绑定** | 2B / 4B | 8K / 16K tok | embedding / head + 可选前后注意力段 | CUDA 主机 + SSH | 降低板端 OM 体积、灵活切分 |

说明：
- 表中是当前仓库**已经验证通过**的典型组合，不表示模型大小、上下文窗口或板端执行范围被引擎模式写死
- `splitnn_om` / `splitnn_bound_embed_head` 与切分点、`max_len`、板端承担的前后段范围在代码层面都是可配置的
- 更详细的组合关系见 [设计决策](./docs/DESIGN.md) 与 [架构设计](./docs/ARCHITECTURE.md)

---

## 模型对照

| 模型 | 参数量 | hidden_size | 层数 | 速度 | 文档 |
|------|--------|-------------|------|------|------|
| Qwen3-0.6B | ~600M | 1024 | 28 GQA | 3.6 tok/s | [模型详情](./docs/MODELS.md) |
| Qwen3.5-0.8B | ~800M | 1024 | 24 (18DN+6GA) | ~4.7 tok/s | [模型详情](./docs/MODELS.md) |
| Qwen3.5-2B | ~2B | 2048 | 24 (18DN+6GA) | ~5.3 tok/s | [模型详情](./docs/MODELS.md) |
| Qwen3.5-4B | ~4B | 2560 | 32 (24DN+8GA) | ~1.1 tok/s | [模型详情](./docs/MODELS.md) |

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [快速开始](./docs/QUICKSTART.md) | 三步完成纯板端部署 |
| [架构设计](./docs/ARCHITECTURE.md) | 三层架构、三种引擎模式、生成流水线 |
| [部署详解](./docs/DEPLOYMENT.md) | 板端文件布局、SSH隧道、启动脚本 |
| [开发指南](./docs/DEVELOPMENT.md) | 环境配置、工具链、导出流程、验证 |
| [设计决策](./docs/DESIGN.md) | Monkey-patch、双缓冲、切分策略、无状态多轮 |
| [实验记录](./docs/EXPERIMENTS.md) | 十个阶段的开发历史 |
| [踩坑速查](./docs/GOTCHAS.md) | 常见问题与解决方案 |
| [模型对照](./docs/MODELS.md) | 所有支持模型的详细参数 |

---

## 硬件与工具链

| 组件 | 说明 |
|------|------|
| 开发板 | Atlas 200I DK A2 (Ascend310B4, 4GB NPU) |
| CANN | 7.0.0 (ATC 编译) / 7.0.RC1 (板端 runtime) |
| ONNX | opset 15, TorchScript 导出 |
| 容器 | Podman + Rocky Linux 9, 镜像 `cann-atc-rocky:v7` |
| Python | pixi 管理 (x86), pip (板端 aarch64) |

---

## 项目结构

```
├── model/                    # 模型权重 + tokenizer
├── scripts/                  # ONNX 导出 & ATC 转换 (x86)
│   ├── qwen35_model_spec.py       # ModelSpec/SplitConfig (无 torch 依赖)
│   ├── qwen35_split_common.py     # SplitNN 共享代码
│   ├── export_*.py                # 各导出脚本
│   ├── validate_*.py              # 验证脚本
│   ├── gen_input_shape.py         # ONNX → ATC INPUT_SHAPE
│   └── podman_convert.sh          # 容器化 ATC 转换
├── board/                    # 板端推理 (aarch64)
│   ├── kvcache_runner.py          # KV Cache 推理基类
│   ├── gen_text_*.py              # 推理脚本
│   └── run_*.sh                   # 启动脚本
├── controller/               # OpenAI API 控制器
│   ├── openai_controller.py       # FastAPI 入口
│   ├── modeling/                  # 模型抽象层
│   ├── engine/                    # 推理引擎层
│   ├── generation/                # 生成循环与采样
│   └── tokenization/              # Tokenizer 适配
├── server/                   # CUDA 主机中段服务
├── om_out/                   # ATC 编译产物
├── docker/                   # ATC 容器构建
├── tests/                    # 单元测试
└── docs/                     # 项目文档
```

---

## 快速开始

```bash
# 1. 导出 ONNX
pixi run python scripts/export_qwen35_kvcache.py \
  --max-len 256 --output om_out/qwen3.5_kvcache_max256.onnx

# 2. ATC 编译
export INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py \
  om_out/qwen3.5_kvcache_max256.onnx)
MODEL_ONNX=om_out/qwen3.5_kvcache_max256.onnx \
  bash scripts/podman_convert.sh

# 3. 上传并启动 (板端)
cd /root/slm_deploy && bash run_kvcache_4096.sh

# 4. 测试
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-0.8B-kvcache-om","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

详见 [快速开始](./docs/QUICKSTART.md)。

OpenAI 请求参数、模型名查询、流式/非流式示例见 [快速开始](./docs/QUICKSTART.md) 中的“OpenAI 请求参数与示例”小节。
