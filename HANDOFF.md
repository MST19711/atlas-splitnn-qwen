# Qwen3.5-0.8B 开发板部署卡点

## 现象

`acl.mdl.load_from_file()` 在 CANN 7.0.RC1 runtime 上挂死（进入 D 态），dmesg 显示 `drv_soft_fault: err_type=0xa (os-memory)` 后 driver 自动 recover。

CANN 7.0.RC1 无法加载 CANN 8.0.RC3 编译的 OM。

## 环境

| 项目 | 值 |
|------|-----|
| 开发板 | Atlas 200I DK A2 (Ubuntu 22.04 aarch64) |
| CANN | 7.0.RC1 (出厂自带) |
| NPU | Ascend310B4 |
| 用于编译的 CANN | 8.0.RC3 (容器 cann-atc-ubuntu22:v4) |

## 状态

| OM 文件 | 编译 soc_version | CANN 7.0 能否加载 | 备注 |
|---------|-----------------|-------------------|------|
| qwen3_kvcache_max256.om | Ascend310B4 | **能** | 仅含标准 MatMul/Where |
| qwen3.5_kvcache_max256.om | Ascend310B4 | **不能** | 含 DeltaNet 算子 |

## ONNX 模型规格

| 项目 | 值 |
|------|-----|
| 输入 | 50 (input_ids + position + 18×S + 18×conv + 6×K + 6×V) |
| 输出 | 49 (logits + 18×S + 18×conv + 6×K + 6×V) |
| 大小 | 1921 MB |
| opset | 15 |
| 导出脚本 | `scripts/export_qwen35.py` |
| ONNX 文件 | `om_out/qwen3.5_kvcache_max256.onnx` |
| OM 文件 | `om_out/qwen3.5_kvcache_max256.om` |
| 板端脚本 | `board/gen_text_qwen35.py` |

## 关键文件

```
Embedded_FinalHW/
├── scripts/
│   ├── export_qwen35.py          ← Qwen3.5 ONNX 导出 (monkey-patch)
│   └── test_qwen35_onnx.py       ← ONNX 推理验证 (本地 CPU)
├── board/
│   └── gen_text_qwen35.py        ← 板端推理脚本 (50 I/O)
├── om_out/
│   ├── qwen3.5_kvcache_max256.onnx  ← ONNX (1921 MB)
│   └── qwen3.5_kvcache_max256.om    ← OM (1921 MB, CANN 8.0 编译)
└── AGENTS.md                     ← 板子信息
```

## 数学等价性验证

已在本地（x86, ONNX Runtime）通过：

- 原始 PyTorch vs Patched PyTorch (逐 token): **diff=0.0234** (FP16 精度内)
- PT vs ORT (ONNX 导出保真度): **diff=0.031**
- 实际生成: 输入"你好" → 输出"你好！我是通义千问..."

## 可能路线

1. **升级板载 CANN 到 8.0.RC3** — 需要 NPU 驱动 + 固件的匹配升级
2. **用 CANN 7.0 ATC 重新编译** — x86 上 7.0 ATC 尝试失败 (Op store init failed)，可能某些环境变量或 OPP 路径不对
3. **简化 ONNX 图** — 把 DeltaNet 的算子用更基础的操作重新表达（可能会有精度损失）
