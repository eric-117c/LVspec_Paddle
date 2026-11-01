# LVSpec Paddle 适配与推理优化交付报告

**2026-11-01 更新：** 新增 Graph 元数据批处理、合并主机读取和请求内参考块缓存，
五视频两遍解码达到 **2.212×**、含 prefill 吞吐 **1.674×**，35 项回归通过，
输出与上一版逐 token 一致。详情见 [第二轮优化复测](PADDLE_ROUND2_TEST.md)。
下文保留原交付与清理记录。

交付日期：2026-10-20。交付内容为当前 LVSpec 仓库中的原生 Paddle 推理后端、性能优化、
运行及验证工具、测试结果与文档。详细数据见 [综合测试报告](TEST_REPORT.md)。

## 1. 交付结果

已完成原始 PyTorch quick inference 复现、Paddle 核心推理迁移、真实视频用例验证，
并对迁移后的投机解码路径进行了优化。最终在两张 A100 80GB、Qwen2.5-VL-32B＋7B、
五个视频、每种方法重复两遍的配置下，LVSpec 相对同版 AR 的解码吞吐达到 **2.19×**。

| 交付项 | 完成情况 |
| --- | --- |
| 原仓库 PyTorch 复现 | 三卡五视频 AR/LVSpec 对比完成，解码加速 1.88× |
| Paddle 模型与生成迁移 | 原生 Paddle 前向、视觉剪枝、KV Cache、树验证、宽松接受及 PST 已实现 |
| 实际运行入口 | 根目录 `inference.py --backend paddle`、Qwen/LLaVA 脚本、独立依赖配置 |
| 性能优化 | 草稿链快速路径、RoPE/掩码复用、RMSNorm、CUDA Graph、可选投影打包 |
| 验证工具 | 固定输入基准、数值对齐、质量评审、显卡占用监测、完整性检查产物 |
| 当前实测 | 两卡五视频解码 2.19×，含 prefill 吞吐 1.63×，35 项回归检查通过 |
| 尚未完成的验证 | 三卡无干扰优化版复测、严格同条件的跨框架性能比较、完整数据集质量评测 |

本交付采用手工实现 Paddle 动态图模型与生成控制流、加载转换权重的方式。
没有将包含动态树、接受判断和缓存回滚的完整生成循环宣称为一次 X2Paddle trace 转换成果。
当前成果位于 LVSpec 仓库，不表示已合入 PaddleNLP/PaddleMIX 等上游仓库。

## 2. 改了什么

### 2.1 新增原生 Paddle 后端

| 文件 | 改动内容 | 解决的问题 |
| --- | --- | --- |
| [models.py](../paddle_backend/models.py) | Qwen2.5-VL 视觉编码器、语言层、投影及条件生成模型 | 将核心模型前向迁移为 Paddle 运算 |
| [llava.py](../paddle_backend/llava.py) | SigLIP 视觉塔、跨模态投影、Qwen2 语言模型的视频路径 | 提供 LLaVA-OneVision 视频推理实现 |
| [ops.py](../paddle_backend/ops.py) | Attention、RoPE、RMSNorm、MLP、mask 与设备辅助操作 | 统一两类模型的基础运算及设备行为 |
| [checkpoint.py](../paddle_backend/checkpoint.py) | HF safetensors 严格加载、权重布局转换、可选 `.pdparams` 导出及分卡 | 复用已有权重，避免遗漏参数或错误转置 |
| [inputs.py](../paddle_backend/inputs.py) | 视频读取、抽帧、HF processor、NumPy/Paddle 输入转换和样本读取 | 使预处理和 Paddle 模型衔接，支持保存/复用固定输入 |
| [cache.py](../paddle_backend/cache.py) | 每层有容量上限的 KV 存储、追加、截断、路径压缩 | 支持多轮投机接受与回滚，控制缓存分配 |
| [decoding.py](../paddle_backend/decoding.py) | AR、严格投机、LVSpec、LVSpec+PST，视觉剪枝、候选验证与接受 | 在 Paddle 中完成整个生成流程 |
| [draft_graph.py](../paddle_backend/draft_graph.py) | 固定形状的草稿生成块捕获和重放 | 减少连续小步草稿推理的调度成本 |
| [optimization.py](../paddle_backend/optimization.py) | 草稿 Q/K/V 与 gate/up 投影的运行时打包 | 提供可选的投影合并路径 |
| [inference.py](../paddle_backend/inference.py) | Paddle CLI、参数校验、模型放置、输入及结果保存 | 提供完整、可独立执行的后端入口 |

Paddle 推理环境未安装 Torch；模型前向、剪枝和解码均在 Paddle 执行。
Transformers 用于 tokenizer/processor，数组通过 NumPy 进入 Paddle。
数值参考导出和质量裁判使用独立 Torch 环境，不参与 Paddle 推理计时。

### 2.2 权重、缓存和多卡行为

权重加载逐项校验参数名及形状，对 Linear 权重执行布局转置，包括正方形矩阵；
embedding、norm、卷积按各自布局加载。缺失或不匹配的权重明确报错，避免随机参数静默参与推理。
目标与草稿先在 CPU 完成读取、布局及 dtype 转换，再搬移到 GPU，处理了本机 safetensors
主机存储与 CUDA 初始化顺序引发的加载问题。这一方式要求有足够主机内存。

多卡采用按层连续放置。视觉塔和 embedding 位于目标首卡，末端 norm/head 位于末卡；
KV 跟随对应层。设备 guard 使用明确的 `gpu:N` 字符串，处理了 Paddle 3.3.1 中
泛化 Place 对象可能把分配导向 GPU 0 的问题。这里没有实现 tensor parallel。

缓存按当前 prompt、生成上限和候选规模确定容量，首次使用时按层分配，原地追加，
不在每步拼接整个历史前缀。分支树接受后压缩所选路径；连续链接受后只调整有效长度。
生成严格遵守 `max_new_tokens`，EOS 计入 token 数。

FP32 数值检查中，Qwen 非重叠 patch Conv3D 改用等价矩阵乘法，避开本机 cuDNN 的精度问题；
FP16/BF16 保留卷积路径。这项修正没有使真实大模型的全部严格数值检查通过，结果单独记录。

### 2.3 运行入口与评估工具

| 文件 / 目录 | 交付内容 |
| --- | --- |
| [根入口 inference.py](../inference.py) | 在导入 Torch 前分流 `--backend paddle`，使独立 Paddle 环境可执行 |
| [run_qwenvl_paddle.sh](../run_qwenvl_paddle.sh)、[run_llava_paddle.sh](../run_llava_paddle.sh) | 视频推理脚本，模型、卡号、方法、帧数和生成长度可配置 |
| [requirements-paddle.txt](../requirements-paddle.txt) | 独立后端依赖，Paddle CPU/GPU 包按环境另行安装 |
| [benchmark_paddle.py](../tools/benchmark_paddle.py) | 一次加载模型、复用 NPZ、预热、交替方法顺序、多次重复及吞吐统计 |
| [export_torch_reference.py](../tools/export_torch_reference.py)、[verify_paddle.py](../tools/verify_paddle.py) | 导出参考权重/输入/输出，报告逐张量误差和 argmax 一致率 |
| [evaluate_paddle_comparison.py](../tools/evaluate_paddle_comparison.py) | 校验完整用例及输入哈希，计算速度、参考辅助质量分与 ROUGE-L，缓存重复评分 |
| [monitor_benchmark_gpus.py](../tools/monitor_benchmark_gpus.py) | 记录测试卡计算进程，识别运行中发生的同卡竞争 |
| [compare_speed.py](../compare_speed.py) | 按总 token/总解码时间对比 AR 与加速方法，区分吞吐比和耗时比 |
| [evaluation/paddle_lvspec.py](../evaluation/paddle_lvspec.py) | 提供 lmms-eval 视频生成适配器；尚未完成全数据集集成评测 |
| [tests/paddle](../tests/paddle) | 两类模型、缓存、分支、生成边界、加载、CLI、GPU 和指标回归 |

原 Torch 复现还涉及本机模型/数据路径、依赖及运行参数配置；这些复现准备与 Paddle 核心实现
分别说明。交付清单以以上模块为范围，不将工作区全部历史差异视为本次优化成果。

## 3. 怎么优化

### 3.1 定位瓶颈

初版 Paddle 三卡五视频完整测试中，AR 为 9.42、LVSpec 为 11.84 token/s，仅 1.26×。
后续 profile 的单样本中，草稿生成累计约 9.89 秒，而视觉相似度约 0.097 秒、
接受判断约 0.086 秒、KV 压缩约 0.354 秒。因此优先处理草稿连续小步推理，
以及每层、每 token 重复构造与主机交互。profile 含嵌套 forward/prefill，阶段时间不能直接相加。

### 3.2 优化措施与适用条件

| 优化 | 原先的开销 | 实现方式与作用 | 限制 / 代价 |
| --- | --- | --- | --- |
| 按设备复用 RoPE 与 mask | 相同前向中各层重复计算位置、三角函数及 mask | 同设备构造一次，沿层复用；普通 prefill/单 token 解码在适用时使用隐式因果路径 | 复用范围为本次前向，分卡时各设备分别维护 |
| GPU additive mask | SDPA 处理 bool mask 时反复构造主机标量 | 在 GPU 上构造所需 dtype 的 additive mask 并复用，减少主机交互，也支持 Graph 捕获 | 树和缓存前缀的可见性仍需保持正确 |
| 低精度 RMSNorm | 多个基础算子和中间结果带来调度开销 | FP16/BF16 GPU 使用 fused RMSNorm，单位 scale 归一化后另乘权重，保留原先的舍入位置 | CPU/FP32 使用原路径；核函数变化仍可能影响低精度结果 |
| greedy chain 快速路径 | 用通用分支树流程处理实际只有单链的候选 | 候选保留在 GPU，向量化接受判断；目标做连续因果验证，已接受 KV 只截断 | 仅识别出的 rank-0 单链走此路径，分支树继续通用处理 |
| 延迟最后叶子 KV | 构造候选时计算了并不总需要的末端前向 | 先构造候选；只有整链接受时补齐末端 KV，减少未被利用的草稿计算 | 必须覆盖全部接受及回滚边界 |
| 草稿 CUDA Graph | 每个 token、每层由 Python 反复发起小算子 | 捕获一整块固定步数草稿生成，更新根 token 和前缀位置后重放 | 当前要求单卡草稿、SDPA、GPU FP16/BF16、head_dim≥64 及 greedy chain |
| Graph KV 布局与写入 | 动态长度、主机索引和变化形状不利于捕获 | 使用固定容量、时间轴优先存储及 GPU 索引 `scatter_`，普通缓存持共享视图；mask 屏蔽未用位置 | 显存容量在请求开始时确定，回滚后有效前缀必须正确 |
| 可选投影打包 | Q/K/V 和 gate/up 分别调用矩阵乘法 | 运行时拼接权重，减少投影调用；buffer 不进入 checkpoint，保留原参数 | 增加草稿显存；未默认开启，单例未证明独立收益 |

CUDA Graph 只捕获适合固定形状重放的草稿块；外层接受决策、EOS 和树控制流仍然动态执行。
不符合条件时使用普通 eager 草稿路径。每条请求的捕获和准备开销计入解码时间，
不是只测已经建好的图重放。AR 同样使用公共算子优化，因此最终比值的分母也是优化后的 AR。

### 3.3 调优过程的测量

两卡、同一视频、128-token 单样本记录如下，用于说明迭代方向，不作为消融实验或最终验收：

| 阶段 | AR token/s | LVSpec token/s | LVSpec / AR |
| --- | ---: | ---: | ---: |
| 第一轮优化 | 12.88 | 19.83 | 1.54× |
| 加入投影打包 | 12.96 | 19.63 | 1.51× |
| 掩码优化 | 12.98 | 20.42 | 1.57× |
| CUDA Graph | 12.99 | 26.08 | 2.01× |

这些结果显示草稿 Graph 值得做完整验证；没有证明每一项优化都独立提高性能。
投影打包的该次结果没有提升，最终完整测试采用 Graph＋打包配置，不能将整体收益归因于打包。

## 4. 最终效果如何

### 4.1 性能

最终验收输入为五份固定视频 NPZ，每视频 64 帧，生成上限 512 token；
目标 32B 在物理 GPU 0、草稿 7B 在 GPU 4，均为 A100 80GB，FP16、greedy、batch size 1。
每方法先预热，再交替顺序运行，每视频各两遍，共 20 次生成。

| 指标 | AR | LVSpec | 结论 |
| --- | ---: | ---: | --- |
| 解码吞吐，token/s | 13.49 | 29.60 | **2.19×** |
| 含 prefill 吞吐，token/s | 12.11 | 19.68 | **1.63×** |
| 平均含 prefill 耗时，秒 | 42.28 | 19.98 | 耗时比 2.12×，包含回答更短的影响 |
| 平均生成长度，token | 512.0 | 393.2 | 输出长度不同，速度需按 token 归一化 |
| 正确性 / 覆盖度代理分，0–5 | 3.60 / 3.60 | 3.40 / 3.40 | 质量无损尚未证明 |

五视频分别达到 **2.07×、2.02×、2.16×、2.30×、2.36×**；
两遍分别聚合为 2.192×、2.198×。输入哈希与初版五视频测试一致，每种方法的重复 token IDs 一致。
185 次约每 5 秒的显卡采样未观察到同卡其他计算进程，测试结束后进程已退出。

历史完整记录中，PyTorch 三卡为 1.88×，Paddle 初版三卡为 1.26×；
优化版三卡也完成过 20 次生成，但草稿卡中途被其他任务占用，计时比为 1.94×，
已排除无干扰性能结论。随后完成上述两卡复测。
卡数、目标放置和跨框架数值路径不同，不能用这些数据宣称 Paddle 框架快于 PyTorch，
也不能把前后绝对吞吐变化全部归因于代码优化。

### 4.2 正确性和质量

最新 CPU 后端/CLI/指标回归 26 项、GPU 回归 9 项，共 **35 项通过**。
覆盖加载转置、视频前端、分卡、缓存回滚、分支隔离、EOS/生成长度、四种生成模式、
FP16/BF16 Graph 完整接受等路径。正式 Paddle CLI 的三个真实视频用例均成功完成。

小尺寸 Qwen/LLaVA 架构的 FP32 跨框架张量检查通过，logits 最大绝对误差约 1e-7。
真实 7B 在严格容差下 logits/hidden states 仍有失败项；GPU logits 最大绝对误差约 2.74e-4，
该参考输入的 argmax 一致率为 100%。这不等于真实大模型全序列严格对齐。

优化会改变部分低精度生成结果，因此变化的回答已重新评审。
五个独立视频上，正确性代理分为 AR 3.6/5、LVSpec 3.4/5；裁判共享目标权重，
存在偏差，且参考文字核查发现漏掉电子烟主题、落水、吸发装置等细节，
木桌清洁中的液体施加位置也有差异。重复生成不增加独立质量样本数。
不能把上述分数换算成正式准确率，也不能承诺 LVSpec 质量无损。

## 5. 接手与使用

运行环境、支持参数和权重转换见 [Paddle 使用说明](PADDLE.md)。
下面是使用本机默认模型和数据路径的五视频运行示例；执行前确认所选 GPU 空闲。

```bash
cd /home/wjy/workFiles/LVSpec
PYTHON=/home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python \
GPU_IDS=0,4 METHOD=lvspec EVAL_NUM=5 FRAME_NUM=64 MAX_NEW_TOKENS=512 \
SAVE_PATH="results/paddle_delivery_$(date +%Y%m%d_%H%M%S)" \
bash run_qwenvl_paddle.sh \
  --target_devices 0 --draft_devices 1 --pack_draft_projections --save_inputs
```

其中物理 GPU 0 对应逻辑 0，物理 GPU 4 对应逻辑 1。
模型和数据路径可用 `BASE_MODEL_PATH`、`DRAFT_MODEL_PATH`、`DATA_PATH` 覆盖。
严格复现 AR/LVSpec 的交替预热基准，应使用 [完整复测命令](PADDLE_OPTIMIZATION_TEST.md)，
而非将两次独立 CLI 运行直接视为同等受控的性能实验。

优化开关：CLI 用 `--disable_cuda_graph` 关闭图重放、`--pack_draft_projections` 开启打包；
基准脚本对应为 `--disable-cuda-graph`、`--pack-draft-projections`。
当前图优化满足条件时默认启用，投影打包需显式开启。

最终结果目录为 `results/paddle_optimization/two_gpu_clean_20261020_065626/`：

| 产物 | 内容 |
| --- | --- |
| [comparison.json](../results/paddle_optimization/two_gpu_clean_20261020_065626/comparison.json) | 完整性能、质量评分及理由 |
| [integrity_checks.json](../results/paddle_optimization/two_gpu_clean_20261020_065626/integrity_checks.json) | 用例完整性、输入/代码哈希、重复输出及分卡输出一致性 |
| [validation_checks.json](../results/paddle_optimization/two_gpu_clean_20261020_065626/validation_checks.json) | 35 项回归检查记录 |
| [gpu_monitor.summary.json](../results/paddle_optimization/two_gpu_clean_20261020_065626/gpu_monitor.summary.json) | 同卡竞争采样结果 |
| `ar/results.jsonl`、`lvspec/results.jsonl` | 逐次文本、生成 token IDs、时间和长度 |
| `command.json`、`run_config.json`、`source_sha256.json` | 复现命令、配置、输入哈希与代码哈希 |
| `manual_reference_audit.json`、`judge_cache_seed.json` | 参考描述核查与评分复用来源 |

## 6. 交付边界

当前完成的是 greedy、batch size 1 的视频推理；Qwen 有真实 32B＋7B 验证，
LLaVA 主要为小模型架构验证，缺少实权重测试。LLaVA AnyRes 图像路径、训练、量化、
滑窗注意力和任意 RoPE 扩展不在当前支持范围。lmms-eval 适配器已提供，完整数据集评测尚未执行。

后续验收若要求论文级质量保持或严格的框架横向比较，需要扩大独立样本，使用人工/独立裁判，
统一输入、注意力数值路径及生成计数，并补充三卡无干扰对照。
本轮可以交付并复现的性能结论是：**限定五视频两卡配置下，解码 2.19×、含 prefill 吞吐 1.63×；
质量存在差异，真实大模型严格数值对齐尚未全部通过。**

## 7. 交付后代码清理（2026-10-20）

已将根目录两份一次性三卡等待/比较脚本归档至
`results/maintenance/cleanup_20261020_144520/`；常用推理脚本、基准与评估工具继续保留。
清除了不再使用的参数设备规划函数及属性、未使用的导入和打包状态标记，
并移除 Torch AR 分支中会触发未定义名称错误的悬空语句。
同时清理 Python/pytest 缓存和 `.DS_Store`，补充忽略规则。

清理后 CPU 回归 26 项、GPU 回归 9 项均通过，共 35 项；语法和报告链接检查通过。
本轮没有重新运行大模型性能测试。历史性能结果及其测试时的源码哈希保留，
当前源码因清理产生的哈希变化不回写历史结果；清理记录与本轮验证见归档目录中的 `manifest.json`。
