# LVSpec Paddle：视频大模型投机解码适配与优化

本项目基于 [LVSpec](https://github.com/zju-jiyicheng/LVSpec)，完成视频大模型推理流程的
**原生 Paddle 适配、投机解码性能优化和真实视频验证**。目标是让原有 PyTorch 推理方法
在 Paddle 环境中独立运行，并验证 LVSpec 相对自回归解码（AR）的速度与输出质量。

最终在 **两张 A100 80GB、Qwen2.5-VL-32B＋7B、五个视频、每方法重复两遍**的测试中，
LVSpec 相对同版 Paddle AR 的解码吞吐达到 **2.19×**，计入 prefill 后的吞吐达到 **1.63×**。

**汇报文档：** [交付报告：改动与优化](docs/DELIVERY_REPORT.md) ·
[综合测试报告](docs/TEST_REPORT.md) · [完整复测记录](docs/PADDLE_OPTIMIZATION_TEST.md)

## 完成了什么

| 工作 | 实现内容 | 验证情况 |
| --- | --- | --- |
| 原项目复现 | 跑通原始 PyTorch quick inference，建立 AR/LVSpec 基线 | 三卡五视频完成 |
| 模型迁移 | Paddle 实现 Qwen2.5-VL 视觉编码器、语言模型和多模态投影；提供 LLaVA-OneVision 视频实现 | Qwen 32B＋7B 实权重验证，LLaVA 小模型架构验证 |
| 解码迁移 | AR、严格投机、LVSpec、LVSpec+PST；包含视觉剪枝、候选验证、接受及缓存回滚 | 四模式均有功能测试 |
| 权重与设备适配 | HF safetensors 严格加载及 Linear 转置，可选导出 `.pdparams`；CPU 加载后按层放置到 GPU | 加载、转置、缓存及双卡回归通过 |
| 推理优化 | 草稿链快速路径、RoPE/掩码复用、RMSNorm、CUDA Graph 和可选投影打包 | 五视频完整解码测试达到 2.19× |
| 验证与交付 | 推理入口、运行脚本、固定输入基准、数值对齐、质量评审、显卡监测及报告 | 最新 35 项回归通过 |

Paddle 环境中没有安装 Torch，模型前向、视觉剪枝和生成控制流都由 Paddle 执行。
Transformers 用于 tokenizer/processor，数据通过 NumPy 输入 Paddle。
本次采用手工迁移动态模型与生成控制流、加载转换权重的方式；完整生成循环没有通过
X2Paddle 自动 trace 转换。成果以本仓库中的 Paddle 后端交付。

## 怎么优化

初版迁移后的三卡完整测试只得到约 1.26× 解码加速。Profile 显示，草稿模型连续小步生成
是主要优化方向，因此重点减少每 token、每层重复计算、Python 调度和 KV 操作。

| 优化 | 具体做法 | 作用 |
| --- | --- | --- |
| RoPE 与掩码复用 | 每次前向按设备计算并复用位置频率和 mask，适用时使用隐式因果注意力 | 减少各层重复构造 |
| RMSNorm 与 GPU mask | 低精度 GPU 使用 fused RMSNorm；在 GPU 上构建 additive mask | 减少算子调度和主机交互 |
| 草稿链快速路径 | 单链候选保留在 GPU，向量化接受判断，连续接受后截断缓存 | 减少逐 token 同步和不必要的分支路径拷贝 |
| 延迟末端 KV | 只有整条候选链接受后才补齐末端 KV | 避免未被利用的草稿前向 |
| CUDA Graph | 捕获固定步数的草稿生成块，更新 token/位置后重放 | 减少连续小算子的 Python 发起开销 |
| 可选投影打包 | 合并草稿 Q/K/V 与 gate/up 投影调用 | 提供减少投影调用的路径，代价是额外显存 |

CUDA Graph 当前用于满足条件的单卡 FP16/BF16 草稿、SDPA 和 greedy chain；其他情况使用
普通 eager 路径。最终测试开启 Graph 和投影打包，**每条请求建图的开销也计入解码时间**。
投影打包的单例测量没有显示独立收益，不将整体加速归因于这一项。

优化过程、相关代码文件和代价见 [交付报告](docs/DELIVERY_REPORT.md)。

## 最终效果

测试配置：目标 Qwen2.5-VL-32B、草稿 7B，各用一张 A100 80GB；Paddle GPU 3.3.1、
FP16、greedy、batch size 1。VideoDetailCaption 五个视频，每视频 64 帧，生成上限 512 token，
每种方法重复两遍，共 20 次推理；使用相同 NPZ 输入，预热后交替方法顺序运行。

| 指标 | Paddle AR | Paddle LVSpec | 结果 |
| --- | ---: | ---: | --- |
| 解码吞吐，token/s | 13.49 | 29.60 | **2.19×** |
| 含 prefill 吞吐，token/s | 12.11 | 19.68 | **1.63×** |
| 平均生成长度，token | 512.0 | 393.2 | 回答长度不同 |
| 正确性 / 覆盖度代理分，0–5 | 3.60 / 3.60 | 3.40 / 3.40 | 质量保持仍需进一步验证 |

吞吐按总生成 token 数除以总时间计算，排除模型加载和视频预处理。
五视频逐例解码加速分别为 **2.07×、2.02×、2.16×、2.30×、2.36×**，
两次重复输出 token IDs 一致。185 次显卡采样未观察到同卡其他计算进程。

功能验证共 **35 项通过**：CPU 后端/CLI/指标 26 项，GPU 9 项。
另通过正式 Paddle CLI 跑通了三个真实视频的读取、预处理、推理及结果保存。

以上是限定配置下的实测结果。五视频质量评分由本地模型辅助完成，存在遗漏和评审偏差，
不能换算成正式准确率或宣称无损。真实 7B 的严格跨框架数值检查仍有未通过项。
优化版三卡测试受到其他任务干扰，尚需无干扰复测；当前数据也不能用于认定 Paddle 比
PyTorch 框架更快。历史测试及失败记录均保存在 [综合测试报告](docs/TEST_REPORT.md)。

## 快速运行 Paddle

### 环境与模型

在独立环境安装与本机 CUDA 匹配的 Paddle GPU，再安装后端依赖。
已验证环境为 Python 3.13、Paddle GPU 3.3.1、CUDA runtime 12.6。
具体环境和支持参数见 [Paddle 使用说明](docs/PADDLE.md)。

```bash
python -m pip install -r requirements-paddle.txt
python inference.py --backend paddle --help
```

准备 Qwen2.5-VL-32B/7B 的 HF 权重和 VideoDetailCaption 数据集。
模型、视频和大型 NPZ 输入不随仓库上传；生成测试输入的方法见 [测试记录说明](results/README.md)。

### 三个真实视频快速检查

下面使用两张 A100 80GB，目标与草稿各一张卡，生成上限 64 token，用于快速检查完整链路。
执行前修改路径并确认所选 GPU 空闲。

```bash
BASE_MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct \
DRAFT_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
DATA_PATH=/path/to/VideoDetailCaption \
GPU_IDS=0,1 METHOD=lvspec EVAL_NUM=3 FRAME_NUM=64 MAX_NEW_TOKENS=64 \
SAVE_PATH=results/paddle_quick \
bash run_qwenvl_paddle.sh \
  --target_devices 0 --draft_devices 1 --pack_draft_projections --save_inputs
```

`--target_devices` 与 `--draft_devices` 使用可见卡重编号后的逻辑编号。
结果写入 `results.jsonl`、`metric.jsonl` 和 `run_config.json`；保存输入便于之后做一致对比。
完整的五视频、512-token 性能基准应按 [复测命令](docs/PADDLE_OPTIMIZATION_TEST.md) 运行，
上述 64-token 快速检查不用于复现 2.19× 指标。

### 功能回归

```bash
python -m pip install pytest
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m pytest -q tests/paddle
```

GPU 检查需要两张可用显卡。测试覆盖 Qwen/LLaVA 小模型、权重加载、缓存回滚、树分支、
生成边界、CLI、不同精度和 CUDA Graph。

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

当前支持 greedy、batch size 1 的视频推理；Qwen 已做真实 32B＋7B 验证，
LLaVA 尚缺实权重验证。训练、量化、LLaVA AnyRes 图像路径等不在本次交付范围。

## 原项目与致谢

LVSpec 算法、原始 PyTorch 模型代码和论文素材来自
[zju-jiyicheng/LVSpec](https://github.com/zju-jiyicheng/LVSpec)。本仓库的工作重点是
Paddle 工程适配、推理优化与验证。上游论文介绍、作者信息和引用保留在
[原项目 README](docs/UPSTREAM_README.md)。
