# 部署详解

板端文件布局、SSH 隧道配置、shell 脚本说明与各个部署模式的详细步骤。

---

## 板端文件布局

### KV Cache 纯板端

```
/root/slm_deploy/
├── controller/                  # 控制器代码（完整目录）
│   ├── openai_controller.py
│   ├── schemas.py
│   ├── modeling/
│   │   ├── base.py, factory.py
│   │   └── kvcache_qwen35.py
│   ├── engine/
│   │   ├── base.py, constants.py
│   │   └── om_engine.py
│   ├── generation/              # runner, config, strategies, logits_processors
│   └── tokenization/qwen35.py
├── scripts/qwen35_model_spec.py  # ModelSpec 结构定义（无 torch 依赖）
├── qwen3.5_kvcache_max256.om     # KV Cache OM 模型（1.9GB）
├── config.json                   # Qwen3.5 模型配置
├── tokenizer.json, vocab.json, merges.txt, tokenizer_config.json, chat_template.jinja
└── run_kvcache_4096.sh
```

### SplitNN OM

```
/root/slm_deploy/
├── controller/                   # 控制器代码
├── scripts/qwen35_model_spec.py
├── qwen3.5_split_prefix_max16384.om + .metadata.json
├── qwen3.5_split_suffix_max16384.om + .metadata.json
├── tokenizer 文件
└── run_openai_split_controller_om_16k.sh
```

### SplitNN 参数绑定（示例：2B）

```
/root/slm_deploy/
├── controller/
├── scripts/qwen35_model_spec.py
├── qwen3.5_2b_bound_embed_head/
│   ├── tied_weight.bin           # (2048×248320, FP16, ~970MB)
│   ├── final_norm_weight.bin
│   ├── bound_embed_head.metadata.json
│   └── op_models/                # ACL single-op MatMul 产物
├── qwen3.5_2b_prefix_dn_8k.om    # 可选：纯注意力 prefix OM (split > 0 时)
├── qwen3.5_2b_suffix_ga_8k.om    # 可选：纯注意力 suffix OM (split < N 时)
├── tokenizer 文件 (model_2b/)
└── run_openai_split_controller_bound_2b.sh
```

> Bound 模式不再强制 `split=0/N/0`。若前缀有注意力层，需上传对应的 prefix OM；若后缀有注意力层，需上传对应的 suffix OM。OM 文件由 `scripts/export_qwen35_middle.py` 导出。

---

## 板端启动脚本

所有启动脚本统一使用 `board/setup_ascend_env.sh` 初始化 Ascend 环境，通过 `exec python3` 替换 shell 进程。

| 脚本 | 后端 | 说明 |
|------|------|------|
| `run_kvcache_4096.sh` | `qwen35_kvcache_om` | 纯板端 KV Cache，4096 tok，100MB cache 限制 |
| `run_openai_split_controller_om.sh` | `splitnn_om` | SplitNN OM，当前脚本示例为较短上下文 |
| `run_openai_split_controller_om_16k.sh` | `splitnn_om` | SplitNN OM，当前脚本示例为 16K |
| `run_openai_split_controller_bound_2b.sh` | `splitnn_bound_embed_head` | 参数绑定，当前脚本示例为 2B `0/24/0` |
| `run_openai_split_controller_bound_4b.sh` | `splitnn_bound_embed_head` | 参数绑定，当前脚本示例为 4B `0/32/0` |
| `run_qwen3_kvcache.sh` | — | 裸 Qwen3 KV Cache 推理（调试用） |
| `run_qwen35_splitnn.sh` | — | 裸 SplitNN 推理（调试用） |
| `run_qwen35_splitnn_16k.sh` | — | 裸 SplitNN 16K 推理（调试用） |

说明：
- 上表描述的是仓库内现有脚本的默认参数，不表示上下文长度或板端执行范围被引擎模式固定写死
- `splitnn_om` / `splitnn_bound_embed_head` 都可通过更换资产、`--split` 和 `--max-len` 调整板端承担的前后段范围与上下文窗口

---

## SSH 隧道

SplitNN 方案需要在开发机和板端之间建立反向隧道，使板端能通过 `127.0.0.1:28080` 访问开发机的 `127.0.0.1:18080`。

```bash
sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no \
  -o ExitOnForwardFailure=yes \
  -N -R 28080:127.0.0.1:18080 \
  root@192.168.137.100
```

- `-N`: 不执行远程命令
- `-R 28080:127.0.0.1:18080`: 将远端 28080 端口转发到本地 18080
- `-o ExitOnForwardFailure=yes`: 端口转发失败立即退出

---

## 参数绑定编译

参数绑定模式下板端 LM Head 使用 ACL single-op `MatMul` 执行，需要为真实 head shape 编译 OM。以下命令只是当前验证过的示例：

```bash
# 一键编译（需 ATC 容器）
pixi run python scripts/export_qwen35_bound_embed_head.py \
  --model-path model_dl/Qwen3.5-2B \
  --output-dir om_out/qwen3.5_2b_bound_embed_head \
  --split 0,24 --compile-op

# 或单独编译已有资产
bash scripts/compile_head_matmul.sh om_out/qwen3.5_2b_bound_embed_head
```

---

## 内存优化

板端 ATLAS 200I DK A2 出厂自带桌面环境，推理场景可安全关闭：

```bash
systemctl stop sddm && systemctl disable sddm
pkill -f xfce4-power-manager
pkill -f xfce4-notifyd
pkill -f tumblerd
```

不影响网络管理、文件系统、音频、通信等系统服务。效果: 665 MiB → 544 MiB，释放约 121 MiB。

---

## NPU 状态检查

```bash
# 启动前确认
npu-smi info

# 应显示 Health: OK
# 若为 Alarm → reboot
```

NPU 进程被 kill 后驱动不自动清理，必须重启板子。
