# LVSpec PyTorch 复现与 Paddle 适配综合测试报告

**2026-11-01 增补：** 新版五视频两遍实测解码 **2.212×**、含 prefill 吞吐 **1.674×**，
两遍均超过 2.2×，全部输出与上一版一致。改动及旧版同权重对照见 [第二轮优化复测](PADDLE_ROUND2_TEST.md)。
下文保留截至 2026-10-20 的历史汇总，不覆盖旧数据。

工程改动、优化方法及接手入口见 [交付报告](DELIVERY_REPORT.md)。

数据截至：2026-10-20。报告依据本仓库已保存的结果、配置、生成文本和显卡监控记录编写，
本次汇总未重新执行模型推理。历史数据与优化版结果均保留，不将小样本、未完成或受干扰记录混为正式结论。

## 1. 测试结论

最初要求的三步流程已经完成：原始 PyTorch quick inference 跑通、核心推理迁移到 Paddle、
通过真实视频用例验证。后续优化版在两张 A100 80GB、五视频、每种方法重复两遍的配置下，
LVSpec 相对同版 AR 的解码吞吐达到 **2.19×**，五个视频逐例聚合均超过 2×。

| 验收项 | 结果 | 结论边界 |
| --- | --- | --- |
| PyTorch quick inference | 三卡、五视频，AR/LVSpec 各完成一次 | 原实现可运行，解码加速比 1.88× |
| Paddle 原生推理适配 | Qwen 32B＋7B 真实推理完成，四种生成模式均有测试 | 手工迁移动态控制流，未宣称整套循环通过 X2Paddle 自动转换 |
| 真实视频快速用例 | 正式 Paddle CLI 的三个视频全部通过 | 验证完整链路和非空输出，不能替代准确率评测 |
| 优化后解码加速 ≥2× | 两卡、五视频、各两遍，2.19× | 仅支持本次硬件、输入和参数范围 |
| 含 prefill 吞吐 ≥2× | 实测 1.63× | 未达到；完整端到端吞吐也未证明达到 2× |
| 准确率保持 | 正确性代理分 AR 3.6/5、LVSpec 3.4/5 | 不能宣称无损或给出正式准确率百分比 |
| 三卡优化版无干扰复测 | 三卡运行完成，但草稿卡中途被共享 | 计时受干扰，尚无有效的完整复测结论 |
| Paddle 相对 PyTorch 的框架优劣 | 尚未完成严格同条件对照 | 不能将 2.19× 与 1.88× 直接当成框架性能比 |

## 2. 环境、数据与统计口径

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB，共享服务器；每次明确指定测试卡 |
| 目标 / 草稿模型 | 本地 Qwen2.5-VL-32B-Instruct / Qwen2.5-VL-7B-Instruct |
| Paddle 环境 | Python 3.13.13、Paddle GPU 3.3.1、CUDA runtime 12.6 |
| 预处理依赖 | Transformers 4.48.0、safetensors 0.8.0；通过 NumPy 输入 Paddle |
| 主性能配置 | FP16、greedy、batch size 1、每视频 64 帧、生成上限 512 token |
| 数据集 | VideoDetailCaption，固定五个视频；三个正式 CLI case 是其中的子集 |
| LVSpec 参数 | `drop_rate=0.9`、`loose_hidden_k=7`、默认 chain |
| 优化版开关 | CUDA Graph 开启，草稿投影打包开启 |
| 设备并行方式 | 目标模型按层分卡或单卡放置；不是 tensor parallel |

Paddle 推理环境没有安装 Torch，模型前向、剪枝、KV Cache 和生成由 Paddle 执行。
质量裁判在另一个 Torch 环境运行，只读取保存的文本，不计入推理耗时。

性能统一采用总量口径：

```text
解码吞吐 = 总生成 token 数 / 总解码时间
解码加速比 = LVSpec 解码吞吐 / AR 解码吞吐
含 prefill 吞吐 = 总生成 token 数 / 总模型推理时间（prefill + decode）
```

上述时间均排除模型加载和视频预处理。优化版解码计时包含每条请求建立 CUDA Graph 的开销。
回答长度不同会影响总耗时，所以不能将未按 token 归一化的耗时比作为纯解码加速比。
Paddle EOS 计入生成长度，且严格限制生成上限；旧 Torch 结果沿用原仓库计数，
记录中存在 AR 511 token、LVSpec 个别样本 517 token 的情况，尚未统一为严格跨框架计数。

Paddle 完整比较使用相同的五份 NPZ 并校验 SHA256，先预热、交替方法顺序、每视频重复两次。
跨框架的输入预处理、注意力数值路径和 token 计数尚未全面统一。

## 3. 完整性能测试总览

吞吐单位：token/s。加速比均为该行 LVSpec 相对该行 AR，而非 Paddle 相对 PyTorch。

| 编号 | 测试 | GPU 数 | 视频数 × 每方法重复数 | AR 吞吐 | LVSpec 吞吐 | 解码加速比 | 数据用途 |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| P1 | PyTorch，原始 KV Cache | 3 | 5 × 1 | 9.41 | 17.67 | 1.877× | 原实现复现基线 |
| P2 | Paddle，优化前 | 3 | 5 × 2 | 9.42 | 11.84 | 1.256× | 有完整无同卡竞争监测记录 |
| P3 | Paddle，优化后 | 3 | 5 × 2 | 13.17 | 25.56 | 1.941× | 中途共享草稿卡，排除正式计时结论 |
| P4 | Paddle，优化后 | 2 | 5 × 2 | 13.49 | 29.60 | **2.195×** | 当前主要性能结论 |

P2 使用目标物理 GPU 2、3，草稿 GPU 4；244 次显卡采样未观察到其他计算进程。
P3 使用目标 GPU 0、4，草稿 GPU 5；GPU 5 中途出现其他用户任务，已标记受干扰。
P4 使用目标 GPU 0、草稿 GPU 4；185 次、约每 5 秒的采样未观察到同卡其他计算进程。
采样不能排除间隔中的短任务，也不能排除共享主机或互连资源的影响。

不能用 P4/P2 的吞吐商将收益全部归因于代码优化：两者的卡数和目标放置不同。
P3/P2 虽均为三卡，但 P3 受干扰，也不构成严格的优化前后性能对照。

### 3.1 最终两卡完整结果

| 指标 | AR | LVSpec | 说明 |
| --- | ---: | ---: | --- |
| 完成推理次数 | 10 | 10 | 五视频，每种方法各两次 |
| 总生成 token 数 | 5120 | 3932 | 回答长度不同 |
| 总解码时间，秒 | 379.60 | 132.83 | 解码吞吐比 2.195× |
| 解码吞吐，token/s | 13.49 | 29.60 | 已按生成 token 归一化 |
| 含 prefill 总推理时间，秒 | 422.82 | 199.80 | 排除加载和预处理 |
| 含 prefill 吞吐，token/s | 12.11 | 19.68 | 吞吐比 1.625× |
| 平均推理耗时，秒 | 42.28 | 19.98 | 耗时比 2.116×，包含回答更短的影响 |
| 平均生成长度，token | 512.0 | 393.2 | AR 全部触及长度上限 |
| 触及生成上限次数 | 10/10 | 4/10 | 回答可能在句中截断 |
| 平均接受长度 | — | 8.10 | 每轮实际接受 token 数，含树根 |

两遍分别汇总：第一遍 AR/LVSpec 为 13.47/29.53 token/s，加速 2.192×；
第二遍为 13.50/29.68 token/s，加速 2.198×。
这些是同一批视频的重复测量，不是独立数据集或统计置信区间。

### 3.2 五视频逐例结果

| 视频 / 场景 | 输入 token | 优化前，三卡加速比 | 优化后，两卡 AR 吞吐 | 优化后，两卡 LVSpec 吞吐 | 优化后加速比 | AR / LVSpec 生成长度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v_wCexKIuAj3A`，电视节目 | 3211 | 1.29× | 16.14 | 33.48 | 2.07× | 512 / 317 |
| `v_FqYhC6_CMfI`，风帆冲浪 | 7435 | 1.19× | 13.17 | 26.55 | 2.02× | 512 / 181 |
| `v_w4pO-1-FG5w`，理发 | 8107 | 1.23× | 12.78 | 27.65 | 2.16× | 512 / 512 |
| `v_4iFlM_Zu7xs`，越野摩托车 | 8107 | 1.23× | 12.74 | 29.35 | 2.30× | 512 / 444 |
| `v_4Gx_8vuxzAk`，木桌清洁 | 7531 | 1.31× | 13.15 | 31.06 | 2.36× | 512 / 512 |

表内均按每视频两次的总 token/总时间计算。前后卡数不同，只展示各自的实测收益，
不用于隔离某项优化的因果贡献。优化版每种方法同一视频的两次输出 token IDs 完全一致；
其两卡与三卡配置的全部输出 token IDs 也完全一致。

## 4. 优化过程与单样本测量

这些测量使用两卡、单视频、128-token 上限，用于寻找方向，不能代替完整测试。

| 测量阶段 | AR 吞吐 | LVSpec 吞吐 | 加速比 | 原始记录 |
| --- | ---: | ---: | ---: | --- |
| 第一轮优化 | 12.88 | 19.83 | 1.539× | `v1_measure.json` |
| 草稿投影打包 | 12.96 | 19.63 | 1.514× | `packed_measure.json` |
| 掩码优化 | 12.98 | 20.42 | 1.573× | `mask_measure.json` |
| CUDA Graph | 12.99 | 26.08 | 2.008× | `graph_measure.json` |

投影打包在该次测量中没有显示独立收益，并增加草稿显存；不能将整体收益归于投影融合。
各阶段也不是多次重复、严格隔离变量的消融实验。最终完整测试启用了打包与 CUDA Graph。

主要实现变化包括：每设备复用 RoPE/掩码、低精度 RMSNorm、链式候选快速路径、
减少逐 token 主机交互和 KV 拷贝，以及用 CUDA Graph 重放草稿生成块。
早期 profile 的草稿生成合计为 9.89 秒、视觉相似度 0.097 秒、接受判断 0.086 秒、
KV 压缩 0.354 秒，支持优先优化草稿小步生成路径。
profile 同时记录了嵌套 forward 和 prefill，不能将全部阶段时间相加。

## 5. 功能用例与模式验证

### 5.1 正式入口与四模式

正式入口 `inference.py --backend paddle` 完成三个真实视频的读取、64 帧预处理、
32B＋7B 推理和结果保存；每例生成 64 token，非空，退出码 0。
三个用例为电视节目、风帆冲浪、理发，输入长度分别为 3211、7435、8107。

早期单真实视频、三卡、每模式生成 64 token 的检查如下；这组数据只验证模式行为，
不是优化版的完整性能结果。

| 方法 | 解码 token/s | 相对 AR | 平均接受长度 | 与 AR token IDs 相同 |
| --- | ---: | ---: | ---: | --- |
| AR | 10.13 | 1.000× | — | 是 |
| 严格投机 `specvlm_uniform` | 6.77 | 0.668× | 4.00 | 是 |
| LVSpec | 11.09 | 1.095× | 7.11 | 否 |
| LVSpec + PST | 9.64 | 0.952× | 7.11 | 否 |

该组中 LVSpec 含 prefill 耗时为 9.08 秒，AR 为 8.50 秒，说明短生成下解码略快
并不等于整体推理更快。LVSpec/PST 是宽松接受，不能要求其输出天然等同 AR。

### 5.2 CPU 与自动回归

| 检查 | 结果 | 用途 |
| --- | --- | --- |
| 真实 7B CPU AR，2 帧合成短视频 | 生成 3 token，2.37 token/s | CPU 正式入口冒烟 |
| 真实 7B CPU 四模式，共享 target/draft，4 token | AR 3.043、严格 0.699、LVSpec 0.725、PST 0.720 token/s | 检查模式可运行，不代表 32B＋7B 性能 |
| 初版功能回归 | 29 passed | 历史版本检查 |
| 最新 CPU 后端/CLI/指标回归 | 26 passed | 缓存、树、位置、边界、加载与指标 |
| 最新 GPU 回归 | 9 passed | 双卡、FP32/FP16/BF16、Graph 回滚与完整接受 |

最新回归合计 **35 passed**。29 项是历史集合，与当前集合重叠，不能相加成 64 项独立测试。
Qwen 和 LLaVA 的小模型均覆盖；LLaVA 没有实权重运行记录。
另有裁判流程的单例冒烟结果，它复用前述 64-token 输出，不算额外独立质量样本。

## 6. 跨框架数值一致性

检查 FP32 小尺寸真实架构模型的 logits、hidden states、输入 embedding 和视频特征，
统一容差为 `atol=3e-5, rtol=3e-4`；没有通过放宽容差消除失败项。

| 模型 / 设备 | logits 最大绝对误差 | hidden states 最大绝对误差 | 四类张量全部通过 |
| --- | ---: | ---: | --- |
| Qwen 小模型 / CPU | 8.94e-8 | 6.56e-7 | 是 |
| Qwen 小模型 / GPU | 9.69e-8 | 6.56e-7 | 是 |
| LLaVA 小模型 / CPU | 1.19e-7 | 7.15e-7 | 是 |
| LLaVA 小模型 / GPU | 9.31e-8 | 5.96e-7 | 是 |

优化期间还保存了一份小模型 CPU 复核报告 `paddle_optimization/tiny_numerical_report.json`，
八项张量记录均通过。上述小模型通过不能替代真实大模型的检查。

真实 7B 保存的三个历史报告如下。表中是否通过以逐元素容差结果为准，不能仅凭最大绝对误差判断。

| 报告 | logits 最大绝对误差 | hidden states 最大绝对误差 | logits argmax 一致率 | 严格结果 |
| --- | ---: | ---: | ---: | --- |
| `qwen7b_numerical_report.json` | 4.57e-4 | 4.08e-3 | 100% | logits/hidden 未通过；embedding/视频特征通过 |
| `qwen7b_cpu_numerical_report.json` | 3.34e-4 | 2.95e-3 | 100% | 四类张量均未通过 |
| `qwen7b_gpu_numerical_report.json` | 2.74e-4 | 1.68e-3 | 100% | logits/hidden 未通过；embedding/视频特征通过 |

GPU 报告中 logits RMSE 为 1.83e-5。100% argmax 一致只覆盖该输入的参考位置，
不表示数值完全一致，也不保证长序列逐 token 生成一致。
当前不能宣称真实 7B/32B 全模型跨框架严格数值对齐已通过。

## 7. 回答质量与“准确率”的边界

VideoDetailCaption 是开放式描述任务，没有唯一答案可直接计算分类正确率。
现有质量检查使用人工参考描述和本地 Qwen2.5-VL-32B 文本权重，评估正确性、覆盖度，
并计算词级、无词干化的 ROUGE-L F1。评分时隐藏方法标签，每次只展示一条回答。

| 版本 | 方法 | 正确性，0–5 | 覆盖度，0–5 | ROUGE-L F1 |
| --- | --- | ---: | ---: | ---: |
| 优化前五视频 | AR | 3.40 | 3.40 | 0.1622 |
| 优化前五视频 | LVSpec | 3.40 | 3.40 | 0.1927 |
| 优化后五视频 | AR | 3.60 | 3.60 | 0.1604 |
| 优化后五视频 | LVSpec | 3.40 | 3.40 | 0.1926 |

优化后两卡与三卡输出完全一致，质量分数相同，不应重复当成两组独立质量证据。
相同回答按检查点、配置、评分规则和文本哈希复用裁判结果；两次推理重复不增加独立视频数。
对照检查中参考原文得 5/5，明显无关描述得 0/5，只证明裁判能区分这两个极端。

优化后逐视频正确性/覆盖度分数：AR 为 4、3、4、3、4；LVSpec 为 3、3、4、3、4。
不能把 3.4/5 写成“准确率 68%”，也不能把 0.2 分差当成统计显著的准确率变化。

| 视频 | 保存回答与参考描述核查 |
| --- | --- |
| 电视节目 | AR 提及吸烟但写成普通香烟；LVSpec 漏掉电子烟/吸烟主题 |
| 风帆冲浪 | 两种方法均未描述参考中的最后落水 |
| 理发 | 两种方法均未描述修剪器连接吸发装置和增高坐垫 |
| 越野摩托车 | 优化后两种回答均偏重单名骑手，未描述两名骑手先后跳跃及场内摄影者 |
| 木桌清洁 | LVSpec 将液体施加位置写成抹布，参考为桌面；两种方法均未说明最后填补裂缝 |

核查基于保存文字，没有直接观看视频；参考描述可能不完整，额外细节只能记为未核实。
裁判与目标共享权重，存在自评偏差；其理由也有误读，例如声称 AR 提及实际保存输出中
不存在的落水或酒吧切镜。因此现有结果支持粗粒度质量检查，**不支持准确率无损结论**。

## 8. 历史、失败与排除记录

| 记录 | 已保存结果 | 处理方式 |
| --- | --- | --- |
| 历史 Torch 默认输出目录 | AR 9.28、LVSpec 17.52 token/s，1.888× | 保留历史记录；配置完整核验不足，不替代 P1 |
| Torch 两卡 bounded cache | `metric.jsonl` 报 LVSpec 18.13 token/s，平均接受 7.1 | 无该目录配套 AR 对照，不构造加速比；该数是原 metric 口径 |
| Torch 两卡原始 KV Cache | 草稿缓存初始化 OOM，完成 0 例；申请 8.30 GiB，可用 4.81 GiB | 失败记录，不给吞吐或加速比；不能泛化为所有两卡配置不可运行 |
| Paddle 初版共享卡试跑 | 目标卡出现其他任务，部分结果保存 | 排除计时，后续在 GPU 2、3、4 重跑得到 P2 |
| 优化期间旧两卡整批尝试 | 用户要求暂停，仅有部分产物 | 未完成，不计入完整测试 |
| 优化版三卡 P3 | 20 次完成，但 GPU 5 被共享 | 可用于输出/质量核查，排除无干扰速度结论 |

这些记录没有被删除或用成功结果覆盖。不同 GPU 数、输入长度、生成长度和优化阶段的
小样本数据不合并成一个平均加速比。

## 9. 交付范围及后续验证

当前已具备原生 Paddle 推理链路，并在限定的五视频两卡配置下达到解码两倍目标。
仍需三卡同条件无干扰复测，才能确认原三卡配置的优化收益；仍需统一输入、
数值路径和计数的 PyTorch/Paddle 配对实验，才能评价框架相对性能。

质量方面应扩大独立视频数量，使用人工复核或独立裁判，并明确处理生成截断。
尚未完成正式 lmms-eval 全数据集质量评估；没有 LLaVA 实权重验证，也没有训练、量化或
多请求吞吐验证。当前支持范围和运行方式见 [Paddle 使用说明](PADDLE.md)。

## 10. 证据索引与复现

以下链接指向仓库中保存的结果，复现命令及原始评分理由均可追溯。

| 数据 | 入口 |
| --- | --- |
| PyTorch 原始三卡完整比较 P1 | [速度结果](../results/qwen2_5_vl_3gpu_original_kv/speed_comparison.json)、[完成状态](../results/qwen2_5_vl_3gpu_original_kv/status.json) |
| Paddle 优化前三卡 P2 | [完整比较](../results/paddle_validation/ar_vs_lvspec_5cases/comparison.json)、[显卡监测](../results/paddle_validation/ar_vs_lvspec_5cases/gpu_monitor.summary.json) |
| Paddle 优化后三卡受干扰 P3 | [完整比较](../results/paddle_optimization/three_gpu_20261020_063411/comparison.json)、[计时有效性](../results/paddle_optimization/three_gpu_20261020_063411/timing_validity.json) |
| Paddle 优化后两卡 P4 | [完整比较](../results/paddle_optimization/two_gpu_clean_20261020_065626/comparison.json)、[结果完整性](../results/paddle_optimization/two_gpu_clean_20261020_065626/integrity_checks.json)、[显卡监测](../results/paddle_optimization/two_gpu_clean_20261020_065626/gpu_monitor.summary.json) |
| 最终回归检查与质量核查 | [35 项回归](../results/paddle_optimization/two_gpu_clean_20261020_065626/validation_checks.json)、[参考描述核查](../results/paddle_optimization/two_gpu_clean_20261020_065626/manual_reference_audit.json) |
| 正式 CLI 三例 | [用例验收](../results/paddle_validation/quick_cases/validation.json) |
| 三卡四模式单例 | [模式验证](../results/paddle_validation/qwen32b_7b_gpu64/validation.json) |
| 7B CPU 四模式 | [CPU 结果](../results/paddle_validation/qwen7b_cpu_modes/summary.json) |
| 小模型 CPU / GPU 数值 | [CPU](../results/paddle_validation/tiny_numerical_report.json)、[GPU](../results/paddle_validation/tiny_gpu_numerical_report.json) |
| 真实 7B 三份数值报告 | [早期](../results/paddle_validation/qwen7b_numerical_report.json)、[CPU](../results/paddle_validation/qwen7b_cpu_numerical_report.json)、[GPU](../results/paddle_validation/qwen7b_gpu_numerical_report.json) |
| 调优单例 | [第一轮](../results/paddle_optimization/v1_measure.json)、[投影打包](../results/paddle_optimization/packed_measure.json)、[掩码](../results/paddle_optimization/mask_measure.json)、[CUDA Graph](../results/paddle_optimization/graph_measure.json) |
| 历史 Torch 与失败记录 | [默认目录比较](../results/speed_comparison.json)、[bounded cache](../results/qwen2_5_vl_lvspec_2gpu_bounded_cache/metric.jsonl)、[原始 KV 两卡 OOM](../results/speed_comparison_2gpu_original_kv.json) |

最终目录中 `ar/results.jsonl`、`lvspec/results.jsonl` 保存全部回答、token IDs 和逐次耗时；
`command.json`、`run_config.json`、`source_sha256.json` 保存命令、输入哈希和代码哈希；
`judge_cache_seed.json` 保存评分复用来源。详细复现命令见 [优化版完整复测](PADDLE_OPTIMIZATION_TEST.md)，
早期适配验收见 [后端验证记录](PADDLE_VALIDATION.md)。
