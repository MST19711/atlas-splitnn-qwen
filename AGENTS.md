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
- pixi 的环境文件（如 `pixi.lock`、`pixi.toml`、`pyproject.toml`）存放在项目目录中
- 开发板上的项目目录: `/root/slm_deploy`，开发板上没必要使用pixi
- 本项目是 SLM（小语言模型）在 Atlas 200I DK A2 上的部署项目

### pixi 使用规则（严禁违反）
1. **所有 Python 包必须通过 `pixi add` 安装**，严禁在 pixi 环境中调用 `pip install` 或 `conda install`
2. 安装 conda-forge 包: `pixi add <package>`
3. 安装 PyPI 包: `pixi add --pypi <package>`
4. 安装指定版本: `pixi add <package>@<version>` 或 `pixi add --pypi <package>==<version>`
5. 激活环境: `pixi shell`（进入交互式 shell）或 `pixi run <command>`（单次运行）
6. 如果必须从 whl 文件安装（如 torch_npu），需要将 whl 文件放入项目目录，然后在 `pixi.toml` 的 `[pypi-dependencies]` 中添加本地路径引用

## SSH 连接
- 本环境没有安装 `mcp-ssh-tmux`。
- 所有开发板 SSH/SCP 操作直接使用 `sshpass`：
  - SSH: `sshpass -p 'Mind@123' ssh -o StrictHostKeyChecking=no root@192.168.137.100 '<command>'`
  - SCP: `sshpass -p 'Mind@123' scp -o StrictHostKeyChecking=no <local> root@192.168.137.100:/root/slm_deploy/`

## 临时文件
- 所有临时文件存放在项目目录中，不要写入系统的tmp目录

## 当前项目状态（2026-06-08）

### 仓库位置与目标
- 主仓库：`/home/CX_Li/Embedded_FinalHW/DeepSeek-Atlas-Chat`
- 目标：将 SLM 部署到 Atlas 200I DK A2（Ascend310B4）。

### 🆕 新方向：Qwen3-0.6B FP16 静态图（跳过量化）
- **结论**：DeepSeek 1.5B INT8 量化方案失败（动态量化图不被 CANN 良好支持）。
- **新方案**：换用更小的 **Qwen3-0.6B（~600M 参数）**，以 FP16 静态图导出 → ATC 转 OM，完全跳过量化和自定义算子。
- **可行性**：Qwen3-0.6B FP16 仅 ~1.2GB，可装入 3.5GB NPU 内存。FP16 图算子（MatMul、LayerNorm、Softmax、GELU 等）CANN 原生支持。

### ✅ Qwen3-0.6B 部署进度（全部完成）

1. **模型下载** ✅：`models/Qwen3-0.6B/`，含 `model.safetensors`(1.5G)、`config.json`、`tokenizer.json` 等。
2. **ONNX 导出** ✅：
   - seq=1: `export_qwen3_fp16.py` → `qwen3_fp16_seq1.onnx` (1546 MB)
   - seq=32: tile 修补 (GQA Expand→Tile) → `qwen3_fp16_seq32_tile.onnx` (1546 MB)
3. **ATC 转换 OM** ✅（2026-06-08）：
   - 脚本：`podman_convert_qwen3_fp16.sh`（已加 CANN-1.84 symlink 修复）
   - **关键修复**：容器内 `ln -sf /ws/cann8_install/ascend-toolkit/8.0.RC3 /usr/local/Ascend/CANN-1.84`（TBE ccec 编译器路径硬编码问题）
   - OM：`om_out/qwen3_fp16_seq1.om` (1.5 GB) + `om_out/qwen3_fp16_seq32_tile.om` (1.5 GB)
4. **SCP 到开发板** ✅：两个 OM 均在 `/root/slm_deploy/`
5. **开发板 ACL 推理验证** ✅：
   - seq=1: 125ms/tok, 8 tok/s（无上下文，仅基准测试）
   - seq=32: 280ms/tok, 3.6 tok/s，left-padding 滑动窗口，生成连贯中文
6. **文本生成脚本** ✅：
   - `gen_text.py`：chat template + enable_thinking=False + 滑动窗口推理
   - Instruct 模型格式：`<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`

#### ONNX 修补说明
Qwen3 使用 GQA（16 Q-heads vs 8 KV-heads），ONNX 导出时 K/V 头的 Expand 目标 shape 为动态计算（Where + ConstantOfShape），ATC 无法静态推断。`patch_onnx.py` 将其替换为 Tile 算子 + 静态常量 `[1, 8, 2, 32, 128]`。

#### 解决 ATC 的关键信息
- 不需要自定义 parser！不需要 `writable_opp8`！不需要 `ASCEND_CUSTOM_OPP_PATH`！
- 如果要在 Podman 中跑，参考 `podman_convert_om.sh` 但移除 `ASCEND_OPP_PATH` 和 `ASCEND_CUSTOM_OPP_PATH` 中 `writable_opp8` 的引用。
- 简洁的容器内命令：
  ```bash
  atc --model=/workspace/qwen3_fp16_seq1.onnx --framework=5 \
      --output=/workspace/om_out/qwen3_fp16_seq1 \
      --input_format=ND \
      --input_shape="input_ids:1,1;attention_mask:1,1" \
      --soc_version=Ascend310B4 \
      --precision_mode=allow_fp32_to_fp16
  ```
- 如果在容器内遇 Python/numpy 问题，确认容器内 `/workspace/cann8_install/ascend-toolkit/latest/python/site-packages/` 中有 numpy。

### 关键文件与目录
- `deepseek_quant8.onnx` + `deepseek_quant8.onnx.data`：原始 INT8 动态量化 ONNX，图中包含 `DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(scale)`。
- `deepseek_quant8_stripped.onnx` + `.data`：移除了 `MatMulInteger` zero-point 输入的实验版本，用于绕开 `BatchMatMulV2` 把 zero-point 当 bias 的问题。
- `deepseek_quant8_stripped_transpose.onnx`：实验性插入 MatMul 输出转置的版本；已验证会破坏后续 `down_proj`，不要作为正确方案继续推进。
- `cann8_install/`：本机 x86 CANN 8.0.RC3 安装目录。
- `writable_opp8/`：可写 OPP，包含补过的 310B4 op config、自定义 ONNX parser，以及对 `DynamicQuantV2` AscendC 源码的 barrier 补丁。
- `writable_msopgen8/`：CANN 8.0 msopgen 自定义 parser 工程。
- `om_out/deepseek_quant8_stripped_seq1_allpipepatch.om`：本机已成功生成的 seq1 OM，约 2.4GB；只能证明 ATC 可生成，不代表能在 4GB 开发板实际运行。
- `logs/`：ATC 转换日志。重点看 `podman_atc_stripped_seq1_allpipepatch_20260608_003936.log` 和 `podman_atc_stripped_seq16_attrplugin_20260608_004546.log`。
- `CURRENT_CONVERSION_STATUS.md`：最近一次详细状态记录。
- `podman_convert_om.sh`：容器化 ATC 转换脚本。

### 容器转换环境
- Podman 镜像：`localhost/cann-atc-ubuntu22`
- 镜像定义：`DeepSeek-Atlas-Chat/.podman/Containerfile.cann-atc`
- 容器使用 host network，并显式清空代理环境变量，避免代理导致 apt/工具链网络异常。
- 不要在容器中直接 `source cann8_install/ascend-toolkit/set_env.sh`，该脚本内含主机绝对路径；使用 `podman_convert_om.sh` 中手动设置的 `/workspace/...` 环境变量。

常用命令：

```bash
cd /home/CX_Li/Embedded_FinalHW/DeepSeek-Atlas-Chat

# 已知可以成功生成 seq1 OM
SEQ_LEN=1 ./podman_convert_om.sh

# 当前预期失败，用于复现 seq16 shape/layout 问题
SEQ_LEN=16 ./podman_convert_om.sh
```

脚本输出：
- OM：`om_out/deepseek_quant8_stripped_seq${SEQ_LEN}_podman.om`
- 日志：`logs/podman_convert_seq${SEQ_LEN}_*.log`

### 已完成的开发与实验
- 已确认 Podman 可用，并成功构建 `localhost/cann-atc-ubuntu22`。
- 已在容器内确认 `atc`、Python/TBE 依赖可用。
- 已编译并安装自定义 ONNX parser：
  - `DynamicQuantizeLinear -> DynamicQuantV2`
  - `MatMulInteger -> BatchMatMulV2`
  - `DequantizeLinear -> AscendAntiQuant`
- 已给 `writable_opp8/built-in/op_impl/ai_core/tbe/impl/ascendc/dynamic_quant/` 中真实 `.h/.cpp` 文件打补丁：将 `pipe_barrier(PIPE_V);` 替换为 `PipeBarrier<PIPE_V>();`，解决 310B4 编译器报 `the ranges of 1st parameter must be [2, 6], [10, 10]` 的问题。
- `deepseek_quant8_stripped.onnx` 在 `SEQ_LEN=1` 下转换成功，输出 `om_out/deepseek_quant8_stripped_seq1_allpipepatch.om`。
- 已将 seq1 OM 传到开发板：`/root/slm_deploy/deepseek_quant8_stripped_seq1_allpipepatch.om`。
- 开发板 ACL 验证命令曾运行：
  `python3 board_acl_verify.py --model /root/slm_deploy/deepseek_quant8_stripped_seq1_allpipepatch.om --seq-len 1`
  结果：超过 5 分钟无输出，Python RSS 约 2.9GB，NPU memory 约 3395/3513MB，系统可用内存约 75MB；已终止验证进程，不能认为可运行。

### 当前失败原因判断
- 当前 ONNX 是 ONNX Runtime 动态量化图：
  `DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(scale)`。
- CANN 8.0.RC3/Ascend310B4 对这个图不是原生完整适配；当前 parser 是强行映射到 CANN 算子，能部分推进但出现 layout/shape 语义不一致。
- 原始 4 输入 `MatMulInteger` 直接转换失败，因为 zero-point 被 `BatchMatMulV2` 当成可选 `bias/offset_w`：

```text
dimensions a(16) and b(1536) must be same
Failed to infer bias.
```

- stripped 图在 seq16 下失败，因为 CANN 将量化 MatMul 输出推成 `[out_dim, seq]`，而 ONNX 后续 scale 分支期望 `[1, seq, out_dim]`：

```text
input1_shape: 256,16
input2_shape: 1,16,256
In op[mul], the inputs[256,16] could not be broadcast together with shapes[1,16,256].
```

- 给 MatMul 输出插入 `Transpose` 的方案不可用：它会导致后续 `down_proj` 推理失败，例如：

```text
The k-axis of a(16) and b(8960) tensors must be the same
```

- 给 parser 显式设置 `BatchMatMulV2` 的 `adj_x1=false`、`adj_x2=false`、`offset_x=0` 后，seq16 仍是同一类 broadcast 失败。

### 后续建议方向
1. 不建议继续做“只为通过 ATC 的局部 shape hack”，这很容易生成语义错误 OM。
2. 优先在当前本机容器环境中尝试重新导出或改写更适合 Ascend310B4 的量化图：
   - weight-only 量化；
   - QDQ 格式；
   - 或直接构造 CANN 原生 `WeightQuantBatchMatmulV2` / `QuantBatchMatmulV3` 可识别的图。
3. 如果继续沿当前 ONNX Runtime 动态量化图做 parser，需要系统性解决 CANN 内部量化 matmul 输出 layout 与后续 scale/bias/reshape 的语义一致性，不能只局部转置某一段。
4. 开发板只有 4GB 内存，seq1 OM 已经接近不可加载边界。不要在开发板上做长时间 ATC 转换；优先在本机 Podman/CANN 环境完成转换和结构验证，只把最终较小且可加载的 OM 放到开发板。
5. 如果要清理开发板旧进程，避免用 `pkill -f` 匹配到当前 SSH shell；优先先 `ps` 查 PID，再 `kill -9 <pid>`。

### 开发板当前状态记录
- 开发板可通过 sshpass 登录。
- `/root/slm_deploy` 下已有原始 ONNX/data、脚本和 seq1 OM。
- 最近检查后内存已恢复：
  - Mem total 约 3.4GiB，可用约 2.9GiB
  - Swap 8.0GiB
  - NPU memory 空闲后约 581/3513MB 使用
- `npu-smi info` 中 Health 显示 `Alarm`，但此前 ATC/ACL 操作仍可执行；后续如遇运行异常，应先记录 `npu-smi info` 和 `/var/log/npu/slog`。
