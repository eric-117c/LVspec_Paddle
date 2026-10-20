# Paddle 后端验证记录

交付范围：原仓库 Torch quick inference 跑通 → 迁移为 Paddle 推理 → 少量真实视频 case 验证。
不以论文速度复现、完整数据集评测或跨框架逐浮点一致作为本次验收条件。

验证日期：2026-10-16。环境为 Python 3.13、Paddle GPU 3.3.1（CUDA 12.6）、
Transformers 4.48.0、safetensors 0.8.0，设备为 NVIDIA A100 80GB。
详细机器和依赖信息保存于 `results/paddle_validation/environment_report.json`。

三步验收均已完成：Torch quick inference 5 个视频通过，Paddle 核心推理已迁移，
正式 Paddle CLI 的 3 个真实视频 case 通过。功能回归另有 29 项通过。
后续 5 视频、每方法重复 2 次的速度与回答质量评估见 [LVSpec / AR 对比](PADDLE_COMPARISON.md)。

## 原仓库 Torch quick inference

本地仓库 origin 为 `git@github.com:zju-jiyicheng/LVSpec.git`。
2026-10-15 已通过原 `run_qwenvl.sh` → `inference.py` 的 Torch 路径运行
Qwen2.5-VL-32B＋7B、VideoDetailCaption、64 帧、5 个视频，生成上限 512。
AR 和 LVSpec 各保存 5 条非空结果，任务状态为 `complete`、退出码为 0。
本次核对了状态、运行脚本快照、日志及结果文件。

证据目录：`results/qwen2_5_vl_3gpu_original_kv/`，其中包含
`run_qwenvl.snapshot.sh`、`status.json`、`ar.log`、`lvspec.log` 和两种方法的结果。
复现入口：

```bash
PATH=/home/wjy/miniconda3/envs/lvspec/bin:$PATH \
  GPU_IDS=1,2,3 METHOD=lvspec SAVE_PATH=results/torch_quick bash run_qwenvl.sh
```

## Paddle 迁移方式

参考 [X2Paddle PyTorch 转换说明](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md)
中动态图转换和权重加载的思路，已在 `paddle_backend/` 实现原生 Paddle 动态图：
视觉编码器、语言模型、KV cache、视觉剪枝、树验证、宽松接受和 PST 均由 Paddle 执行。
HF safetensors 权重按 Paddle 参数布局加载，可另存为 `.pdparams`。
当前交付是手工迁移的 Paddle 实现，没有宣称整套生成控制流通过 X2Paddle 自动 trace。

Paddle 独立环境未安装 Torch，真实模型测试也在此环境运行；
Transformers 只用于 tokenizer 和图像处理，不执行 Torch 模型。
依赖及运行入口见 [Paddle 使用说明](PADDLE.md)。

## 功能回归

```bash
CUDA_VISIBLE_DEVICES=4,6 OMP_NUM_THREADS=1 \
  /home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python -m pytest -q tests/paddle
```

结果：**29 passed**，耗时 18.80 秒。
覆盖 Qwen2.5-VL 和 LLaVA-OneVision 视频前端、四种生成模式、缓存回滚和树分支隔离、
EOS 与生成上限、权重转置及原生 checkpoint 往返、视频预处理和 CLI。
CUDA 部分覆盖 FP32/FP16/BF16、双卡分层及 KV 放置，
另用带 60 秒超时的独立进程检查 CUDA 初始化后连续加载 target/draft。
测试输出中存在 Paddle 旧版 `cuda.synchronize` 接口的弃用警告，不影响测试通过。

## 跨框架数值对齐

由 PyTorch 导出两类小模型的权重、输入和参考结果，Paddle 使用 FP32 eager 执行。
统一容差为 `atol=3e-5, rtol=3e-4`，没有为了通过测试放宽阈值。

| 模型 / 设备 | logits 最大绝对误差 | hidden states 最大绝对误差 | 全部张量通过 |
| --- | ---: | ---: | --- |
| Qwen / CPU | 8.94e-8 | 6.56e-7 | 是 |
| LLaVA / CPU | 1.19e-7 | 7.15e-7 | 是 |
| Qwen / GPU | 9.69e-8 | 6.56e-7 | 是 |
| LLaVA / GPU | 9.31e-8 | 5.96e-7 | 是 |

报告为 `results/paddle_validation/tiny_numerical_report.json` 和
`tiny_gpu_numerical_report.json`。Qwen FP32 patch projection 改用等价矩阵乘法后，
消除了本机 cuDNN Conv3D 的明显精度损失；低精度推理仍使用 Conv3D。

真实 Qwen2.5-VL-7B FP32 GPU 与 PyTorch FP32 CPU 参考比较：
视觉特征和输入 embedding 通过相同容差，logits 与 hidden states **未通过**。
logits 最大绝对误差为 2.74e-4，RMSE 为 1.83e-5；
hidden states 最大绝对误差为 1.68e-3。全部 prompt 位置的 logits argmax 一致率为 100%。
此结果只能说明该输入上的误差和预测一致性，不能代替完整数值对齐或任务质量评测。
报告为 `results/paddle_validation/qwen7b_gpu_numerical_report.json`。

## 真实 32B＋7B 三卡推理

目标为本机 Qwen2.5-VL-32B-Instruct，草稿为 Qwen2.5-VL-7B-Instruct。
FP16，目标按层放在物理 GPU 1、2，草稿放在 GPU 3。
使用真实视频 `v_wCexKIuAj3A.mp4` 的 64 帧，输入 3211 个 token，
每种方法生成 64 个 token；四种方法读取相同的预处理 NPZ。

```bash
OMP_NUM_THREADS=2 /home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python \
  tools/benchmark_paddle.py \
  --target /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --draft /home/wjy/models/Qwen2.5-VL-7B-Instruct \
  --inputs-npz results/paddle_validation/video64/inputs.npz \
  --gpu-ids 1,2,3 --target-devices 0,1 --draft-devices 2 \
  --dtype float16 --max-new-tokens 64 \
  --output results/paddle_validation/qwen32b_7b_gpu64
```

| 方法 | 解码 tokens/s | 相对 AR | 含 prefill 推理耗时（秒） | 平均接受长度 |
| --- | ---: | ---: | ---: | ---: |
| AR | 10.13 | 1.00× | 8.50 | — |
| specvlm_uniform | 6.77 | 0.67× | 12.92 | 4.00 |
| LVSpec | 11.09 | 1.09× | 9.08 | 7.11 |
| LVSpec + PST | 9.64 | 0.95× | 9.98 | 7.11 |

四模式均正常完成，生成长度与保存的 token IDs 数量一致。
严格投机解码的 64 个 token 与 AR 完全一致；两个宽松模式彼此一致，均与 AR 不同。
LVSpec 在本次解码阶段略快，计入 prefill 后仍比 AR 慢；其余投机模式本次未获得加速。
这是共享机器上的单视频、单次运行，不能据此推断数据集性能、稳定加速比或答案质量。

完整输出、token IDs、各模式指标和参数保存在
`results/paddle_validation/qwen32b_7b_gpu64/`；核对结果及速度比在该目录的
`validation.json`。输入与视频元数据保存在 `results/paddle_validation/video64/`。

## 正式 Paddle 入口的三个真实 case

通过根目录 `inference.py --backend paddle` 运行，使用 32B＋7B、三卡、FP16、
每视频 64 帧、LVSpec、生成上限 64。直接读取真实 MP4，覆盖预处理、模型推理和结果保存。
进程退出码为 0，三条结果均为非空回答，每条生成 64 个 token。

| 视频 | 输入 token 数 | 回答内容开头 | 状态 |
| --- | ---: | --- | --- |
| `v_wCexKIuAj3A.mp4` | 3211 | 早间电视节目、演播室场景 | 通过 |
| `v_FqYhC6_CMfI.mp4` | 7435 | 热带海滩上的风帆冲浪 | 通过 |
| `v_w4pO-1-FG5w.mp4` | 8107 | 理发店中男孩正在理发 | 通过 |

这里的“通过”指完整运行成功、有生成结果且符合长度限制，未使用人工或模型进行质量打分。
本次为快速测试，将生成限制为 64 token，回答可能在句中结束。

样本清单、日志、完整输出、预处理 NPZ 和验收状态分别保存在：

```text
results/paddle_validation/quick_cases/cases.json
results/paddle_validation/quick_cases/lvspec.log
results/paddle_validation/quick_cases/lvspec/results.jsonl
results/paddle_validation/quick_cases/lvspec/inputs_0.npz  # 另有 1、2
results/paddle_validation/quick_cases/validation.json
```

复现本次三个 case：

```bash
OMP_NUM_THREADS=2 /home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python inference.py \
  --backend paddle --model_type qwen2_5_vl \
  --base_model_path /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --draft_model_path /home/wjy/models/Qwen2.5-VL-7B-Instruct \
  --data_path results/paddle_validation/quick_cases/cases.json \
  --evaluation_num 3 --data_num 3 --frame_num 64 --max_new_tokens 64 \
  --gpu_ids 1,2,3 --target_devices 0,1 --draft_devices 2 \
  --dtype float16 --method lvspec --save_inputs \
  --save_path results/paddle_validation/quick_cases/lvspec
```

日常也可直接使用 `run_qwenvl_paddle.sh`，通过环境变量设置 `PYTHON`、
`GPU_IDS`、`EVAL_NUM`、`FRAME_NUM`、`MAX_NEW_TOKENS` 和 `SAVE_PATH`。

## 验证范围

本机没有 LLaVA 实权重，LLaVA 当前验证限于真实架构的小模型。
尚未运行 lmms-eval 数据集质量评测；宽松接受和 PST 的质量需要单独测量。
上述参考输出遵循 LVSpec 仓库中的顺序语言位置规则。
