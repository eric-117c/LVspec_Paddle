# Paddle 后端

全部已保存的性能、功能、数值和质量测试见 [综合测试报告](TEST_REPORT.md)。
工程改动和性能优化说明见 [交付报告](DELIVERY_REPORT.md)。

本后端用 Paddle 动态图执行 Qwen2.5-VL 和 LLaVA-OneVision 的视频推理，覆盖
`ar`、`specvlm_uniform`、`lvspec`、`lvspec_with_pst`。现有 PyTorch 文件和默认入口保留。
模型权重可直接从 Hugging Face safetensors 加载，也可预先转换成 Paddle checkpoint。
已执行的功能、数值和实模型验证见 [验证记录](PADDLE_VALIDATION.md)。
5 个真实视频上 LVSpec 相对 AR 的加速比与质量对比见 [对比报告](PADDLE_COMPARISON.md)。

## 安装

建议使用独立环境，Python 3.10 或更高版本。安装 Paddle 3.3.1 CPU 版用于验证：

```bash
python -m venv .venv-paddle
source .venv-paddle/bin/activate
python -m pip install paddlepaddle==3.3.1
python -m pip install -r requirements-paddle.txt
```

实际 GPU 推理须将 CPU 包换成与机器 CUDA 匹配的 `paddlepaddle-gpu` 3.3.x，
按 [Paddle 官方安装说明](https://www.paddlepaddle.org.cn/install/quick)选择安装命令。
不要在同一个环境同时安装 CPU 和 GPU 两个包。
预处理复用 Transformers tokenizer/image processor，通过 NumPy 传入 Paddle；
不要求安装 PyTorch、PaddleNLP、PaddleMIX 或 X2Paddle。

## 运行

```bash
# 默认使用本机 Qwen 32B + 7B 路径；变量均可覆盖
PYTHON=python GPU_IDS=0,1,2,3 METHOD=ar bash run_qwenvl_paddle.sh
PYTHON=python GPU_IDS=0,1,2,3 METHOD=lvspec bash run_qwenvl_paddle.sh

# 单视频，目标和草稿可以各自分布在指定的逻辑 GPU 上
python inference.py --backend paddle \
  --model_type qwen2_5_vl \
  --base_model_path /path/to/Qwen2.5-VL-32B-Instruct \
  --draft_model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --video /path/to/video.mp4 --prompt 'Describe this video.' \
  --gpu_ids 0,1,2,3 --target_devices 0,1,2 --draft_devices 3 \
  --frame_num 64 --max_new_tokens 512 --method lvspec_with_pst \
  --save_inputs --save_path results/paddle_qwen_pst

# LLaVA 视频推理
BASE_MODEL_PATH=/path/to/llava-onevision-qwen2-72b-ov-hf \
DRAFT_MODEL_PATH=/path/to/llava-onevision-qwen2-7b-ov-hf \
GPU_IDS=0,1,2,3 bash run_llava_paddle.sh
```

GPU 编号：`--gpu_ids` 设置进程可见的物理卡，`--target_devices` 和
`--draft_devices` 使用重编号后的逻辑卡。默认目标模型按层数连续均分到所有可见卡，
视觉塔和 embedding 放第一张卡，norm 和 lm_head 放最后一张卡；草稿默认放最后一张卡。
这是模型按层分卡，不是 tensor parallel，也不会根据空闲显存自动平衡。
应根据权重、视频长度及其他进程占用情况指定设备，给 KV 和 attention 临时内存留空间。
权重先在 CPU 上读取、转置及转换 dtype，再将各层放到指定 GPU。
目标与草稿会先完成 CPU 加载，再进行 GPU 放置；需要足够的主机内存。
共享机器的空闲显存可能在加载期间变化，启动前须确认所指定卡有足够空间。

CPU 验证使用 `--device cpu --dtype float32`。AR 不需要草稿权重。
生成当前为 greedy、batch size 1；不支持训练、量化、滑窗注意力或任意 RoPE 扩展。
支持默认/mrope/linear RoPE。LLaVA 前端支持 SigLIP + Qwen2 视频路径，
图像 AnyRes 路径尚不支持，会明确报错。

数据输入可用 `--data_path /path/to/VideoDetailCaption`，或者一个 JSON/JSONL
列表（每条包含 `video` 和 `question`，视频相对路径以列表文件目录为基准）。
`--inputs_npz` 可直接读取已预处理的数组，适合两框架使用完全相同输入进行对齐。

## 语义与性能口径

语言位置编码沿用本仓库修改后的顺序位置规则，包括剪枝后的草稿位置；
没有切换为原生 Qwen 多模态位置规则。视觉相关性也沿用当前实现，
使用 target prefill 的整个 final hidden sequence 作为参照。
视觉相似度分块取 top-k，避免一次构造完整的候选 × 长视频矩阵。

树候选按祖先关系隔离注意力，接受后将路径 KV 压缩到连续区域，丢弃其他分支。
目标和草稿均维护有容量上限的独立缓存，每层首次使用时分配；
容量按本次 prompt、生成上限及树大小计算，不按模型的最大上下文预留。
所有生成路径都严格限制 `max_new_tokens`，EOS 包含在生成 token 计数中，
`average_accept_length` 统计每轮实际接受 token 数（包含树根）。
这些边界和计数规则比旧版脚本更严格；跨框架比较时应据输出 token IDs 重新统一计数。

默认 `--attention sdpa` 使用 Paddle attention，支持 tree mask；`eager` 用于数值对齐。
`reference` 复现仓库中自定义 BF16 Q/K、FP16 score 的计算顺序，需相应设备支持；
它与默认标准 SDPA 的数值路径不同，不能据模式名称推断逐 token 一致。
原始 PyTorch 输出与 Paddle 输出不保证逐 token 相同；宽松模式还需独立评测答案质量。

输出 `results.jsonl`、`metric.jsonl` 可交给现有 `compare_speed.py` / `plot_speed.py`。
`run_config.json` 记录后端、dtype、分卡、帧数及计数口径。
解码计时排除模型加载、预处理和 target/draft prefill；保留完整浮点精度，避免短耗时被舍入为零。
比较速度应先使用相同 Paddle 输入、设备和参数运行 AR 与 LVSpec。

```bash
python compare_speed.py --baseline results/paddle_qwen_ar \
  --accelerated results/paddle_qwen_lvspec --output results/paddle_comparison.json
```

已保存输入后，可以一次加载权重，依次比较四种模式，避免重复加载大模型：

```bash
python tools/benchmark_paddle.py \
  --target /path/to/Qwen2.5-VL-32B-Instruct \
  --draft /path/to/Qwen2.5-VL-7B-Instruct \
  --gpu-ids 0,1,2 --target-devices 0,1 --draft-devices 2 \
  --inputs-npz results/paddle_qwen_pst/inputs_0.npz \
  --max-new-tokens 128 --repeats 3 --output results/paddle_benchmark
```

每种模式有单独的 `results.jsonl` 和 `metric.jsonl`，另记录生成 token IDs，
可检查严格投机解码与 AR 是否一致。省略 `--draft` 时共享目标权重，
仅适合检查程序完整性；这种配置不能代表 32B＋7B 的性能。

## 权重转换与验证

直接加载 safetensors 时严格校验名字和形状，逐张量转置 `nn.Linear` 权重，
包括正方形投影矩阵；embedding、norm、卷积不转置。支持分片和 BF16 safetensors。
LLaVA 的 SigLIP 对比学习 pooling head 不参与视频特征，加载时明确跳过。
缺少、重复、未知参数或配置不匹配均报错，不静默使用随机权重。

```bash
# 可选：转换后可从目标目录直接 load_model 或作为 --base_model_path 使用
# 转换在 CPU 上使用 FP32，需足够内存和磁盘；不会覆盖非空输出目录
python -m paddle_backend.checkpoint /path/to/hf-checkpoint /path/to/paddle-checkpoint

python -m pip install pytest
python -m pytest -q tests/paddle

# 用原 PyTorch 环境生成小尺寸、真实模型架构的权重和参考输出
/path/to/torch-env/bin/python tools/export_torch_reference.py results/torch_reference
# 在 Paddle 环境验证两类视觉特征、输入 embedding 和 logits
python tools/verify_paddle.py results/torch_reference
```

实权重验证可直接引用现有 checkpoint，不会复制整套模型：

```bash
/path/to/torch-env/bin/python tools/export_torch_reference.py results/real_reference \
  --checkpoint /path/to/Qwen2.5-VL-7B-Instruct \
  --inputs-npz results/paddle_qwen_pst/inputs_0.npz --device cuda:0
python tools/verify_paddle.py results/real_reference --models qwen \
  --output results/real_numerical_report.json
```

参考导出关闭 PyTorch 的 TF32。验证工具默认使用小模型严格容差
`atol=3e-5, rtol=3e-4`，报告全部张量的最大误差、均方根误差，以及 logits 的
argmax 一致率；即使失败也会写出报告。深层实权重模型跨框架/设备的 FP32
累积误差可能超过该容差，需结合误差报告、生成 token 对齐和任务质量评测判断，
不能将 argmax 一致率单独视为完整数值对齐。
Qwen 的 FP32 patch projection 使用与非重叠 Conv3D 等价的矩阵乘法，
避免 GPU cuDNN Conv3D 在关闭 TF32 后仍出现的精度损失；FP16/BF16 保持卷积路径。

双卡 CUDA 测试覆盖 FP32、FP16、BF16，两类视频前端、KV 分卡及四种生成模式：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -q tests/paddle/test_gpu.py
```

这条路线保留完整的动态生成控制流；X2Paddle trace/script 可作为局部模块转换工具，
没有将整套生成循环视为可一次 trace 的固定图。
参考：[X2Paddle PyTorch 转换说明](https://github.com/PaddlePaddle/X2Paddle/blob/develop/docs/inference_model_convertor/pytorch2paddle.md)。

## lmms-eval

将 `evaluation/paddle_lvspec.py` 放到 lmms-eval 的模型目录，
按所用 lmms-eval 版本的模型注册方式注册 `paddle_lvspec`；确保本项目在 `PYTHONPATH`。
使用一个进程，模型自行分卡。模型参数中的多卡设备用冒号连接，避免与模型参数分隔符冲突：

```bash
PYTHONPATH=/path/to/LVSpec CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m lmms_eval --model paddle_lvspec \
  --model_args 'pretrained=/path/to/target,draft_pretrained=/path/to/draft,model_type=qwen2_5_vl,method=lvspec,devices=0:1:2,draft_devices=3,frame_num=64' \
  --tasks videomme --batch_size 1
```

该适配器要求任务的 `doc_to_visual` 返回一个本地视频路径；不接受图片列表、
多视频拼接或采样生成。lmms-eval 及对应任务数据需要单独安装。
