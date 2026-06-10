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
  export_fp16.py        seq=N 静态导出
  export_kvcache.py     KV Cache 导出 (monkey-patch Qwen3Attention)
  export_qwen35.py      Qwen3.5 KV Cache 导出 (4 个轻量 patch: cat→Where, Trilu, RMSNorm, conv)
  patch_onnx.py         GQA Expand→Tile
  podman_convert.sh
board/          # 板载推理 (aarch64)
  gen_text_seq32.py       seq=32 滑动窗口
  gen_text_kvcache.py     Qwen3 KV Cache (max_len=256)
  gen_text_qwen35.py      Qwen3.5 KV Cache (max_len=256, DeltaNet)
  acl_verify.py           ACL 验证
docker/
  Containerfile.v2-cann7  # CANN 7.0 + 310B 内核 (Rocky 9), 镜像: cann-atc-rocky:v7
model/
  Qwen3-0.6B/             # Qwen3 模型权重 + tokenizer
om_out/ logs/
```

## ATC 转换
```bash
# 示例
MODEL_ONNX=om_out/model.onnx \
INPUT_SHAPE="name1:d1,d2;name2:d1,d2" \
IMAGE=localhost/cann-atc-rocky:v7 \
bash scripts/podman_convert.sh
```
- 镜像 `cann-atc-rocky:v7`, CANN 7.0 (实际安装版本 7.1.0.3.220)
- soc_version=`Ascend310B4`
- 需传入 INPUT_SHAPE, MODEL_ONNX, 可选 OUTPUT_PREFIX, IMAGE

## ATC 容器构建
```bash
# 下载 CANN 7.0.0 安装包到 docker/
# 1. toolkit: Ascend-cann-toolkit_7.0.0_linux-x86_64.run (1.6GB)
# 2. kernel: Ascend-cann-kernels-310b-7.0.0-linux.noarch.rpm (351MB)
podman build --network=host -t localhost/cann-atc-rocky:v7 \
    -f docker/Containerfile.v2-cann7 docker/
```

## 当前模型
| 模型 | 文件 | 速度 | 上下文 |
|------|------|------|--------|
| Qwen3 KV Cache | om_out/qwen3_kvcache_max256_cann7.om | 3.6 tok/s | 256 tok |
| Qwen3.5 KV Cache | om_out/qwen3.5_kvcache_max256.om | 3.7 tok/s | 256 tok |

## 踩坑速查
1. **ACL API**: `acl.mdl.add_dataset_buffer(ds,buf)` 返回 tuple `(ptr,ret)`, 需 `_, ret = ...`
2. **TBE ccec**: `tbe/tvm/contrib/ccec.py` 硬编码 `/usr/local/Ascend/CANN-1.84/` → 容器内 symlink
3. **310B内核**: 必须用 `Ascend-cann-kernels-310b` (非 310P 或其他型号), 否则 soc_version=Ascend310B4 失败
4. **GQA Expand**: seq=N 导出后需 `patch_onnx.py` 修复 56 个动态 Expand
5. **thinking**: Qwen3 0.6B 需 `enable_thinking=False`
6. **NPU 泄漏**: kill 后内存不释放 → reboot
7. **numpy<2**: CANN 7 TBE 不兼容 numpy 2.x, 需要 `pip3 install 'numpy<2'`
8. **TBE 依赖**: 需 `pip3 install attrs cloudpickle psutil synr tornado` (te 包依赖)
9. **gcc-c++**: CCE 编译器需要 C++ 标准库头文件, 容器需 `dnf install gcc-c++`
10. **tokenizer 混乱**: Qwen3 和 Qwen3.5 的 tokenizer 文件互不兼容, SCP 时注意不要互相覆盖
11. **桌面服务浪费内存**: sddm/xfce4-power-manager/xfce4-notifyd/tumblerd 可安全关闭, 回收 ~120 MiB RAM

## 内存优化 (板端)
板端 ATLAS 200I DK A2 出厂自带桌面环境, 推理场景可安全关闭:
```bash
systemctl stop sddm && systemctl disable sddm
pkill -f xfce4-power-manager
pkill -f xfce4-notifyd
pkill -f tumblerd
```
- 不影响: 网络管理 (NetworkManager/systemd-networkd), 文件系统 (udisks2/gvfs), 音频 (pulseaudio/pipewire), 通信 (ModemManager), 更新 (unattended-upgrades)
- 效果: 665 MiB → 544 MiB, 释放约 121 MiB

## 导出 KV Cache 模型
```bash
# Qwen3
pixi run python scripts/export_kvcache.py --max-len 256 --output om_out/qwen3_kvcache_max256.onnx

# Qwen3.5
pixi run python scripts/export_qwen35.py --max-len 256 --output om_out/qwen3.5_kvcache_max256.onnx

# ORT 多步验证 → ATC → SCP → board/gen_text_kvcache.py
```
