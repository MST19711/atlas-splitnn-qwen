# CLAUDE.md

## 交互语言
- 请使用**中文**与我对话。
- 阅读网页请使用playwright mcp，不要webfetch防止动态网页无法加载
- 对于需要等待的任务，告诉我多少时间之后唤醒你然后停止输出

## 开发板信息
- **用户名**: root@192.168.137.100
- **密码**: Mind@123
- **型号**: 华为 Atlas 200I DK A2 (Ubuntu 22.04 aarch64)

## Python 环境
- 使用 **pixi** 管理本机所有 Python 环境相关事务
- `pixi.toml` 在项目根目录

### pixi 使用规则
1. **所有 Python 包必须通过 `pixi add` 安装**
2. 安装 conda-forge 包: `pixi add <package>`
3. 安装 PyPI 包: `pixi add --pypi <package>`
4. 运行命令: `pixi run python <script>`

## SSH 连接
- 所有开发板 SSH/SCP 操作直接使用 `sshpass`：
  - SSH: `sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 '<command>'`
  - SCP: `sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no <local> root@192.168.137.100:/root/slm_deploy/`

## 项目目录结构

```
Embedded_FinalHW/
├── model/Qwen3-0.6B/          # 模型权重 + tokenizer
├── scripts/
│   ├── export_fp16.py          # FP16 ONNX 导出 (seq=N, use_cache=False)
│   ├── export_kvcache.py       # 🆕 KV Cache ONNX 导出
│   ├── patch_onnx.py           # ONNX 修补 (GQA Expand→Tile 等)
│   ├── download_model.py       # 下载模型
│   └── podman_convert.sh       # Podman ATC 转换
├── board/
│   ├── gen_text_kvcache.py     # 🆕 KV Cache 推理
│   ├── gen_text_seq32.py       # seq=32 滑动窗口推理 (当前可用)
│   └── acl_verify.py           # ACL 单次推理验证
├── docker/Containerfile.cann-atc
├── om_out/                     # OM 产物
├── logs/
├── pixi.toml / pixi.lock
└── AGENTS.md
```

## 开发板部署: Qwen3-0.6B FP16

### 当前状态

| 模型 | 上下文 | 速度 | 状态 |
|------|--------|------|------|
| seq=1 OM | 单 token | 8 tok/s | 仅基准测试，无上下文 |
| seq=32 OM (tile) | 32-token 滑动窗口 | 3.6 tok/s | ✅ 可用，连贯中文 |
| KV Cache OM | max_len=N | 未部署 | 🔴 开发中 |

### 已部署到开发板

- `/root/slm_deploy/qwen3_fp16_seq1.om` (1.5 GB)
- `/root/slm_deploy/qwen3_fp16_seq32_tile.om` (1.5 GB)
- `/root/slm_deploy/board/` — 推理脚本
- `/root/slm_deploy/tokenizer.json` 等 tokenizer 文件
- 开发板已装 `torch` (CPU) + `transformers` (仅用于 tokenizer)

### ATC 转换

```bash
# 用法示例
MODEL_ONNX=qwen3_fp16_seq32_tile.onnx \
INPUT_SHAPE="input_ids:1,32;attention_mask:1,32" \
OUTPUT_PREFIX=om_out/qwen3_fp16_seq32_tile \
bash scripts/podman_convert.sh
```

容器内关键修复：`ln -sf /workspace/cann8_install/ascend-toolkit/8.0.RC3 /usr/local/Ascend/CANN-1.84`（TBE ccec 编译器路径硬编码）

### ONNX 修补说明
Qwen3 使用 GQA（16 Q-heads vs 8 KV-heads），ONNX 导出时 K/V 头的 Expand 目标 shape 为动态计算（Where + ConstantOfShape），ATC 无法静态推断。`scripts/patch_onnx.py` 将其替换为 Tile 算子 + 静态常量 `[1, 8, 2, 32, 128]`。
