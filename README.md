# LVSpec Paddle：视频大模型投机解码适配与优化

本项目将 [LVSpec](https://github.com/zju-jiyicheng/LVSpec) 的视频大模型推理流程迁移到 Paddle，
并对投机解码进行了优化和真实视频测试。迁移后的推理流程已能在独立的 Paddle 环境中运行，
测试以自回归解码（AR）为基线，比较 LVSpec 的速度和输出质量。

2026-11-01，在两张 A100 80GB 上使用 Qwen2.5-VL-32B＋7B 测试五个视频，每种方法重复两遍。
LVSpec 相对同版 Paddle AR 的解码吞吐为 **2.21×**，计入 prefill 后为 **1.67×**。

项目文档：[交付报告：改动与优化](docs/DELIVERY_REPORT.md) ·
[综合测试报告](docs/TEST_REPORT.md) · [最新优化与复测](docs/PADDLE_ROUND2_TEST.md)

## 已完成的工作

| 工作 | 实现内容 | 验证情况 |
| --- | --- | --- |
| 原项目复现 | 跑通 PyTorch quick inference，建立 AR/LVSpec 基线 | 完成三卡五视频测试 |
| 模型迁移 | 用 Paddle 实现 Qwen2.5-VL 的视觉编码器、语言模型和多模态投影，以及 LLaVA-OneVision 视频推理 | Qwen 已用 32B＋7B 实权重验证；LLaVA 完成小模型架构验证 |
| 解码迁移 | 实现 AR、严格投机、LVSpec 和 LVSpec+PST，包含视觉剪枝、候选验证与接受、缓存回滚 | 四种模式均有功能测试 |
| 权重与设备适配 | 加载 HF safetensors 时校验权重并转置 Linear，支持导出 `.pdparams`；在 CPU 加载后按层放置到 GPU | 加载、转置、缓存和双卡回归通过 |
| 推理优化 | 优化草稿链、RoPE/掩码、RMSNorm 和 CUDA Graph，提供可选投影打包 | 五视频完整解码测试达到 2.21× |
| 测试与交付 | 提供推理脚本、固定输入基准、数值对齐与质量评审工具，保存显卡监测记录和测试报告 | 最新 35 项回归通过 |

推理环境没有安装 Torch，模型前向、视觉剪枝和生成控制流均由 Paddle 执行。
Transformers 负责 tokenizer/processor，数据通过 NumPy 传入 Paddle。
模型和生成控制流采用手工迁移的 Paddle 动态图实现，权重在加载时转换布局；
完整生成循环未使用 X2Paddle 自动 trace 转换。实现代码位于本仓库的 Paddle 后端。

## 推理优化

初版 Paddle 在三卡完整测试中的解码加速约为 1.26×。Profile 显示，草稿模型连续小步生成
耗时较多，优化主要围绕这条路径展开：复用每层重复计算的结果，减少 Python 调度、
逐 token 同步和 KV 操作。

| 优化 | 具体做法 | 作用 |
| --- | --- | --- |
| RoPE 与掩码复用 | 每次前向按设备计算位置频率和 mask，供各层复用；适用时使用隐式因果注意力 | 减少重复构造 |
| RMSNorm 与 GPU mask | 低精度 GPU 使用 fused RMSNorm，在 GPU 上构建 additive mask | 减少算子调度和主机交互 |
| 草稿链快速路径 | 单链候选保留在 GPU，向量化判断接受长度，连续接受后截断缓存 | 减少逐 token 同步和分支路径拷贝 |
| 延迟末端 KV | 整条候选链接受后再补齐末端 KV | 少做未被利用的草稿前向 |
| CUDA Graph | 捕获固定步数的草稿生成块，更新 token/位置后重放 | 减少 Python 反复发起小算子的开销 |
| 可选投影打包 | 合并草稿 Q/K/V 与 gate/up 投影调用 | 减少投影调用，同时占用额外显存 |
| Graph 元数据批处理 | 合并根 token/位置更新，一次准备整块草稿的位置、RoPE 与 mask | 减少临时张量、拷贝和小算子调用 |
| 接受结果与参考块复用 | 每轮一次读取接受决策，请求内缓存固定参考隐藏状态的归一化结果 | 减少重复同步和归一化计算 |

CUDA Graph 用于满足条件的单卡 FP16/BF16 草稿，要求使用 SDPA 和 greedy chain；
其他情况走普通 eager 路径。最终测试同时开启 Graph 和投影打包，每条请求的建图开销
都计入解码时间。投影打包在单例测量中没有显示独立收益，整体加速不能归因于这一项。

各项优化的代码位置和代价见 [交付报告](docs/DELIVERY_REPORT.md)。

## 测试结果

目标模型 Qwen2.5-VL-32B 和草稿模型 7B 各占一张 A100 80GB，使用 Paddle GPU 3.3.1、
FP16、greedy、batch size 1。测试选取 VideoDetailCaption 的五个视频，每视频 64 帧，
生成上限 512 token。两种方法读取相同的 NPZ 输入，预热后交替运行，各重复两遍，共 20 次推理。

| 指标 | Paddle AR | Paddle LVSpec | 结果 |
| --- | ---: | ---: | --- |
| 解码吞吐，token/s | 13.58 | 30.04 | **2.21×** |
| 含 prefill 吞吐，token/s | 12.26 | 20.53 | **1.67×** |
| 平均生成长度，token | 512.0 | 393.2 | 回答长度不同 |
| 正确性 / 覆盖度代理分，0–5 | 3.60 / 3.60 | 3.40 / 3.40 | 质量保持仍需进一步验证 |

吞吐按总生成 token 数除以总时间计算，不包含模型加载和视频预处理。
五视频逐例解码加速分别为 **2.08×、2.04×、2.18×、2.33×、2.39×**，
两遍分别为 2.211×、2.214×。每种方法的重复输出 token IDs 一致，与上一版的全部输出也一致，
119 次显卡采样未观察到同卡其他计算进程。

上一轮完整测试为 2.195×。本轮还用同一批常驻权重做了一个长输入用例的对照，
LVSpec 吞吐提升约 0.29%。这项提升较小，也没有做新旧版五视频的随机交叉实验，
因此跨日期的整批差异不能全部归因于代码优化。当前结果确认了这五个输入上超过 2.2×，
其他输入和后续运行仍可能低于这一数值。

功能回归共 35 项通过，其中 CPU 后端/CLI/指标检查 26 项，GPU 检查 9 项。
正式 Paddle CLI 还跑通了三个真实视频的读取、预处理、推理和结果保存。

质量评分由本地模型辅助完成，仅覆盖这五个视频。生成内容仍有遗漏，评审也存在偏差，
这些分数不能换算成正式准确率，也没有证明质量无损。真实 7B 的严格跨框架数值检查
仍有未通过项。优化版三卡测试受到其他任务干扰，需要补做无干扰复测；现有数据不足以
判断 Paddle 和 PyTorch 的框架性能高低。历史测试和失败记录见 [综合测试报告](docs/TEST_REPORT.md)。

## 快速运行 Paddle

### 环境与模型

先在独立环境中安装与本机 CUDA 匹配的 Paddle GPU，再安装后端依赖。
已验证环境为 Python 3.13、Paddle GPU 3.3.1、CUDA runtime 12.6。
环境配置和参数说明见 [Paddle 使用说明](docs/PADDLE.md)。

```bash
python -m pip install -r requirements-paddle.txt
python inference.py --backend paddle --help
```

运行前需准备 Qwen2.5-VL-32B/7B 的 HF 权重和 VideoDetailCaption 数据集。
仓库不包含模型、视频和大型 NPZ 输入；测试输入的生成方法见 [测试记录说明](results/README.md)。

### 三个真实视频快速检查

下面的命令用两张 A100 80GB 分别运行目标和草稿模型，每个视频最多生成 64 token，
用于检查完整推理流程。执行前请修改路径，并确认所选 GPU 空闲。

```bash
BASE_MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct \
DRAFT_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
DATA_PATH=/path/to/VideoDetailCaption \
GPU_IDS=0,1 METHOD=lvspec EVAL_NUM=3 FRAME_NUM=64 MAX_NEW_TOKENS=64 \
SAVE_PATH=results/paddle_quick \
bash run_qwenvl_paddle.sh \
  --target_devices 0 --draft_devices 1 --pack_draft_projections --save_inputs
```

`--target_devices` 与 `--draft_devices` 指定的是可见卡重新编号后的逻辑编号。
程序将结果写入 `results.jsonl`、`metric.jsonl` 和 `run_config.json`，同时保存输入，方便后续对比。
要复现五视频、512-token 的性能测试，请使用 [复测命令](docs/PADDLE_OPTIMIZATION_TEST.md)；
上面的 64-token 检查只用于确认流程可运行，不用于复现 2.21× 指标。

### 功能回归

```bash
python -m pip install pytest
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m pytest -q tests/paddle
```

GPU 检查需要两张可用显卡。测试包含 Qwen/LLaVA 小模型、权重加载、缓存回滚和树分支，
也检查生成边界、CLI、不同精度及 CUDA Graph。

## 主要代码与文档

| 位置 | 内容 |
| --- | --- |
| [paddle_backend/](paddle_backend/) | 原生 Paddle 模型、权重加载、预处理、KV Cache、解码及优化 |
| [run_qwenvl_paddle.sh](run_qwenvl_paddle.sh) | Qwen Paddle 推理脚本 |
| [run_llava_paddle.sh](run_llava_paddle.sh) | LLaVA Paddle 视频推理脚本 |
| [tools/](tools/) | 固定输入基准、数值参考导出/校验、质量评审和显卡监测 |
| [tests/paddle/](tests/paddle/) | 后端功能与 GPU 回归 |
| [evaluation/paddle_lvspec.py](evaluation/paddle_lvspec.py) | lmms-eval 视频生成适配器，完整数据集评测尚未执行 |
| [交付报告](docs/DELIVERY_REPORT.md) | 改了什么、如何优化、效果和交付范围 |
| [综合测试报告](docs/TEST_REPORT.md) | 全部测试、数值失败项、质量差异及证据索引 |
| [测试记录](results/README.md) | 已上传的实测产物及输入生成方式 |

目前支持 greedy、batch size 1 的视频推理。Qwen 已用真实 32B＋7B 权重验证，
LLaVA 尚未进行实权重验证。本次交付不包含训练、量化和 LLaVA AnyRes 图像路径等功能。

## 原项目与致谢

LVSpec 算法、原始 PyTorch 模型代码和论文素材来自
[zju-jiyicheng/LVSpec](https://github.com/zju-jiyicheng/LVSpec)。本仓库完成了 Paddle 适配及相应的
推理优化和测试。原项目的论文介绍、作者信息和引用保留在
[原项目 README](docs/UPSTREAM_README.md)。
