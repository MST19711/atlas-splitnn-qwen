# 踩坑速查

开发过程中遇到的常见问题和解决方案。

---

## ACL API

1. **`acl.mdl.add_dataset_buffer(ds, buf)` 返回 tuple** `(ptr, ret)`，需 `_, ret = ...` 接收返回值
2. **`acl.rt.memcpy` 方向常量**: `M=0` (设备间), `H2D=1` (主机到设备), `D2H=2` (设备到主机)

---

## ATC 编译

3. **TBE ccec**: `tbe/tvm/contrib/ccec.py` 硬编码 `/usr/local/Ascend/CANN-1.84/` → 容器内需要 symlink
4. **310B 内核**: 必须使用 `Ascend-cann-kernels-310b`（非 310P 或其他型号），否则 `soc_version=Ascend310B4` 失败
5. **numpy<2**: CANN 7 TBE 不兼容 numpy 2.x，需要 `pip3 install 'numpy<2'`
6. **TBE 依赖**: 需 `pip3 install attrs cloudpickle psutil synr tornado`（te 包依赖）
7. **gcc-c++**: CCE 编译器需要 C++ 标准库头文件，容器需 `dnf install gcc-c++`
8. **ATC INPUT_SHAPE 不能内联展开**: KV Cache 模型包含 50-58 个分号分隔的 shape 定义，shell 内联 `VAR=$(cmd) bash script.sh` 会把分号当命令分隔符。必须 `export INPUT_SHAPE` 后运行脚本
9. **静态窗口模型不可用**: Qwen3 静态窗口 (seq=32) 在 CANN 7.1.0.3.220 + Ascend310B4 下 ATC 编译产物输出错误，仅 KV Cache 方案可用
10. **容器构建需 `--network=host`**: 容器内 `dnf install`/`pip install` 需要联网

---

## 板端环境

11. **板端 pip 安装必须 `--no-deps`**: `huggingface-hub>=0.34` 依赖 `hf-xet>=1.1.3`，但 PyPI 无 aarch64 wheel
12. **板端需 jinja2**: `apply_chat_template(enable_thinking=False)` 触发 jinja2 模板渲染，板端需额外安装 `jinja2` + `markupsafe`（aarch64 wheel）

---

## Tokenizer 兼容性

13. **Qwen3 / Qwen3.5 tokenizer 互不兼容**: Qwen3 用 `vocab.json`+`merges.txt`，Qwen3.5 用 `tokenizer.json`。SCP 时注意不要互相覆盖

---

## NPU 运行时

14. **NPU 泄漏**: kill 后内存不释放 → reboot
15. **thinking**: Qwen3 0.6B 需 `enable_thinking=False`（模型本身不支持）
16. **异常退出后 NPU 变 Alarm**: 必须 `reboot`，无其他恢复手段

---

## 内存优化

17. **桌面服务浪费内存**: sddm/xfce4-power-manager/xfce4-notifyd/tumblerd 可安全关闭，回收 ~120 MiB RAM

---

## 模型加载

18. **KV Cache OM 加载约 210 秒**: 1.9GB OM 从磁盘到 NPU 内存
19. **prefill 阶段按 prompt 长度逐 token 执行**: 长 prompt 需数十秒
20. `npu-smi info` 确认 Health: OK 后再启动推理
21. **Qwen3.5-4B 参数绑定不建议用 NPU GatherV2 做 embedding**: `tied_weight.bin` 较大时板端可能出现 `acl.op.execute(GatherV2) failed, ret=100024`。推荐 `0/32/0` 下直接使用 CPU embedding lookup，仅保留 NPU `MatMul` 做 lm_head
22. **中段服务健康检查路径是 `/v1/health`**: `server/qwen35_split_service.py` 不提供 `/healthz`
