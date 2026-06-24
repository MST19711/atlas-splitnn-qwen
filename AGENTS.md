# AGENTS.md

## 语言
使用中文

## 开发板
- `root@192.168.137.100`, 密码 `Mind@123`, Atlas 200I DK A2 (Ascend310B4)
- SSH: `sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 '<cmd>'`
- SCP: `sshpass -p 'Mind@123' scp <local> root@192.168.137.100:/root/slm_deploy/`
- NPU 进程被 kill 后驱动不清理 → 重启板子

## Python 环境
- pixi 管理 (`pixi run python <script>`)
- `pixi add <pkg>` (conda), `pixi add --pypi <pkg>` (pip)

## 常用命令

### 导出 ONNX
```bash
# Qwen3.5 KV Cache
pixi run python scripts/export_qwen35_kvcache.py --max-len 256 --output om_out/qwen3.5_kvcache_max256.onnx

# Qwen3.5 SplitNN
pixi run python scripts/export_qwen35_split_prefix.py --model-path model/Qwen3.5-0.8B --max-len 256 --split 4,20 --output om_out/qwen3.5_split_prefix_max256.onnx
pixi run python scripts/export_qwen35_split_suffix.py --model-path model/Qwen3.5-0.8B --max-len 256 --split 4,20 --output om_out/qwen3.5_split_suffix_max256.onnx
```

### ATC 编译
```bash
export INPUT_SHAPE=$(pixi run python scripts/gen_input_shape.py om_out/model.onnx)
MODEL_ONNX=om_out/model.onnx bash scripts/podman_convert.sh
```

### 验证
```bash
pixi run python scripts/validate_qwen35_kvcache_ort.py om_out/qwen3.5_kvcache_max256.onnx
pixi run python -m pytest tests/ -v
```

### Prefix Cache 验证
```bash
# 单元 + 集成测试
pixi run python -m pytest tests/test_prefix_cache_*.py -v

# 板端多轮对话验证（两轮 curl）
# 启动服务：bash /root/slm_deploy/run_kvcache_4096.sh
# 第一轮：
curl -i -X POST http://192.168.137.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-0.8B-kvcache-om","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'
# 第二轮（携带第一轮 reply）：
curl -i -X POST http://192.168.137.100:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-0.8B-kvcache-om","messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"...把第一轮回复放入..."},{"role":"user","content":"再说一遍"}],"max_tokens":32}'
# 第二轮响应头应含有 X-Prefix-Cache-Status: hit

# cache-disabled 回退
cd /root/slm_deploy && bash run_kvcache_4096.sh --cache-disabled

# 可选 cache 参数
# --cache-max-entries 8 --cache-ttl-sec 300 --cache-min-prefix-len 8
```

### 板端部署
```bash
cd /root/slm_deploy && bash run_kvcache_4096.sh  # 纯板端 KV Cache (推荐)
cd /root/slm_deploy && bash run_kvcache_4096.sh --cache-disabled  # 关闭 prefix cache
cd /root/slm_deploy && bash run_openai_split_controller_bound_2b.sh  # 参数绑定 2B
```

## 关键约束
- **ATC INPUT_SHAPE 不能内联展开**: 必须 `export INPUT_SHAPE` 后运行
- **Qwen3/Qwen3.5 tokenizer 互不兼容**: SCP 时注意不要互相覆盖
- **板端 npu-smi info 必须 OK**, 异常退出后 reboot

## 文档导航
- [快速开始](./docs/QUICKSTART.md)
- [架构设计](./docs/ARCHITECTURE.md)
- [部署详解](./docs/DEPLOYMENT.md)
- [开发指南](./docs/DEVELOPMENT.md)
- [踩坑速查](./docs/GOTCHAS.md)
