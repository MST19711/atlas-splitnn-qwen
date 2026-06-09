# AGENTS.md

## 语言
使用中文

## 开发板
- `root@192.168.137.100`, 密码 `Mind@123`, Atlas 200I DK A2 (Ascend310B4)
- SSH: `sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 '<cmd>'`
- SCP: `sshpass -p 'Mind@123' scp <local> root@192.168.137.100:/root/slm_deploy/`
- 板载已装: `torch(cpu) + transformers` (仅 tokenizer), numpy, acl
- NPU 进程被 kill 后驱动不清理 → 重启板子

## Python 环境
- pixi 管理 (`pixi run python <script>`)
- `pixi add <pkg>` (conda), `pixi add --pypi <pkg>` (pip)
- `pixi.toml` 在项目根

## 项目结构
```
scripts/        # ONNX 导出 + ATC 转换 (x86 dev)
  export_fp16.py      seq=N 静态导出
  export_kvcache.py   KV Cache 导出 (monkey-patch Qwen3Attention)
  patch_onnx.py       GQA Expand→Tile
  download_model.py
  podman_convert.sh
board/          # 板载推理 (aarch64)
  gen_text_seq32.py     seq=32 滑动窗口
  gen_text_kvcache.py   KV Cache (max_len=256)
  acl_verify.py         ACL 验证
docker/Containerfile.v2 # CANN 8.0 + 310B 内核, 镜像: cann-atc-ubuntu22:v4
model/Qwen3-0.6B/       # 模型权重 + tokenizer
om_out/ logs/
ARCHIVE.md              # 完整文档 (人类阅读)
```

## ATC 转换
```bash
# 示例
MODEL_ONNX=om_out/model.onnx \
INPUT_SHAPE="name1:d1,d2;name2:d1,d2" \
bash scripts/podman_convert.sh
```
- 镜像 `cann-atc-ubuntu22:v4`, CANN 内置, soc_version=`Ascend310B4`
- 需传入 INPUT_SHAPE, MODEL_ONNX, 可选 OUTPUT_PREFIX

## 当前模型
| 模型 | 文件 | 速度 | 上下文 |
|------|------|------|--------|
| seq=32 Tile | om_out/qwen3_fp16_seq32_tile.om | 3.6 tok/s | 32 tok |
| KV Cache | om_out/qwen3_kvcache_max256.om | 4.8 tok/s | 256 tok |

## 踩坑速查
1. **ACL API**: `acl.mdl.add_dataset_buffer(ds,buf)` 返回 tuple `(ptr,ret)`, 需 `_, ret = ...`
2. **TBE ccec**: `tbe/tvm/contrib/ccec.py` 硬编码 `/usr/local/Ascend/CANN-1.84/` → 容器内 symlink
3. **310B内核**: 必须用 `Ascend-cann-kernels-310b` (非 310P 或其他型号), 否则 soc_version=Ascend310B4 失败
4. **GQA Expand**: seq=N 导出后需 `patch_onnx.py` 修复 56 个动态 Expand
5. **thinking**: Qwen3 0.6B 需 `enable_thinking=False`
6. **NPU 泄漏**: kill 后内存不释放 → reboot
7. **pip3**: CANN 8.0 compiler 依赖, Containerfile 已处理

## 导出 KV Cache 模型
```bash
pixi run python scripts/export_kvcache.py --max-len 256 --output om_out/qwen3_kvcache_max256.onnx
# ORT 多步验证通过 → ATC → SCP → board/gen_text_kvcache.py
```
