# Paddle LVSpec 相对 AR 的速度与回答质量

2026-10-20 优化后的两卡五视频复测见 [优化版完整复测](PADDLE_OPTIMIZATION_TEST.md)：
解码吞吐 2.19×、含 prefill 吞吐 1.63×；本文保留优化前的三卡历史数据。

验证日期：2026-10-16。结论：**本次配置下解码吞吐提升 1.26×，计入 prefill 后的吞吐提升 1.14×。
小样本自动评分持平，但参考答案核查发现细节遗漏，不能据此认定准确率无损。**

## 测试条件

- Qwen2.5-VL-32B 目标模型＋7B 草稿模型；Paddle GPU 3.3.1，FP16，greedy，batch size 1。
- 目标按层放在物理 GPU 2、3，草稿放在 GPU 4，均为 A100 80GB。
- VideoDetailCaption 固定种子 42 选取的前 5 个可用视频，每个 64 帧，生成上限 512 token。
- 两种方法使用完全相同的预处理 NPZ，保存 SHA256；LVSpec 使用 `drop_rate=0.9`、`loose_hidden_k=7`、默认 chain。
- 每种方法先预热，然后每视频重复 2 次，交替方法顺序，共 20 次生成。
- 两次重复中，各方法在同一视频上的 token IDs 均完全一致。质量样本量是 **5 个视频**。
- 每 5 秒记录 GPU 计算进程，244 次快照未观察到其他计算进程占用测试卡。
  首次试跑在物理 GPU 1 上遇到其他任务，已排除，其部分结果另存于 `shared_gpu_partial/`。

## 速度结果

| 指标 | AR | LVSpec | LVSpec 相对 AR |
| --- | ---: | ---: | ---: |
| 解码吞吐，tokens/s | 9.42 | 11.84 | **1.26×** |
| 含 prefill 的吞吐，tokens/s | 8.75 | 9.93 | **1.14×** |
| 平均含 prefill 的推理耗时，秒 | 58.53 | 39.59 | 耗时比 **1.48×** |
| 平均生成 token 数 | 512.0 | 393.2 | 回答长度不同 |
| 达到 512 token 上限的运行次数 | 10/10 | 4/10 | 部分回答被截断 |

解码吞吐比按 `(LVSpec 总 token / 总解码时间) / (AR 总 token / 总解码时间)` 计算。
含 prefill 的吞吐使用同样的 token 归一化口径。
1.48× 是实际回答的平均耗时比，包含 LVSpec 回答更短的影响，不能当作纯解码加速比。
这里的推理耗时包含视觉/语言 prefill 和生成，排除模型加载及视频预处理。

| 视频 | 输入 token 数 | 解码吞吐加速比 | AR / LVSpec 生成 token 数 |
| --- | ---: | ---: | ---: |
| `v_wCexKIuAj3A`：电视节目 | 3211 | 1.29× | 512 / 317 |
| `v_FqYhC6_CMfI`：风帆冲浪 | 7435 | 1.19× | 512 / 181 |
| `v_w4pO-1-FG5w`：理发 | 8107 | 1.23× | 512 / 512 |
| `v_4iFlM_Zu7xs`：越野摩托车 | 8107 | 1.23× | 512 / 444 |
| `v_4Gx_8vuxzAk`：木桌清洁 | 7531 | 1.31× | 512 / 512 |

这些是少量固定视频的实测结果，不代表全数据集、所有硬件或论文的加速比。

## 回答质量

该任务是开放式视频描述，提供人工参考描述，没有唯一答案可直接计算分类“答对率”。
本次报告参考答案辅助的事实正确性与覆盖度，另给出词级 ROUGE-L F1。

自动裁判使用本地 Qwen2.5-VL-32B 的文本权重，以未修改的 HF Qwen2 文本模型执行，
BF16、greedy。每次只看到问题、参考描述和一条回答，不知道 AR/LVSpec 标签，
也看不到另一方法的回答。相同重复输出复用同一评分，避免重复运行增加虚假的质量样本量。
裁判与目标模型共享权重，存在自评偏差；没有使用外部官方裁判。

| 质量代理指标 | AR | LVSpec | 差值 |
| --- | ---: | ---: | ---: |
| 事实正确性，0–5 | 3.40 | 3.40 | 0.00 |
| 参考事实覆盖度，0–5 | 3.40 | 3.40 | 0.00 |
| ROUGE-L F1，词级、无词干化 | 0.1622 | 0.1927 | +0.0305 |

五个视频逐例的正确性与覆盖度分数均分别为 3、3、4、3、4。
裁判对照检查中，参考原文得 5/5，明显无关描述得 0/5。
这些检查只验证评分器能区分两个极端，不说明其逐例判断可靠。
ROUGE-L 反映词序重合，**不是事实准确率**。

独立核对保存回答与参考描述时发现：

- 电视节目：AR 提到吸烟讨论，但把电子烟写成普通香烟；LVSpec 没有提及吸烟/电子烟主题。
- 风帆冲浪：两种方法保存的回答都没有提及参考描述中的最后落水。
- 理发：两种方法都没有描述吸发装置与修剪器的连接。
- 摩托车：参考描述为两名骑手先后跳跃；AR 描述多名骑手，LVSpec 主要描述一名骑手。
- 木桌清洁：参考描述为把木器抛光剂滴到桌面；LVSpec 写成倒到抹布上，AR 没有明确液体施加位置。

该核查基于数据集参考文字，没有直接观看视频。参考描述也可能不完整，
回答中未被参考提及的额外细节不能自动判为错误。
自动裁判理由中也存在误读，例如声称 AR 提到了落水或摄影者，但保存输出中未出现；
因此保留原始评分理由供复核，**没有将自动评分持平解释为“准确率完全不变”**。
本轮可确认速度提升和粗粒度评分相近，不能证明事实细节无损或论文级准确率保持率。

## 结果与复现

结果目录：`results/paddle_validation/ar_vs_lvspec_5cases/`。

- `comparison.json`：速度、质量评分、裁判提示及原始理由。
- `speed_report.json`：聚合及逐视频速度、重复输出一致性。
- `cases.json`：视频、问题、人工参考描述及输入文件。
- `ar/results.jsonl`、`lvspec/results.jsonl`：完整回答、token IDs 和每次耗时。
- `run_config.json`：模型、设备、输入 SHA256 和参数。
- `judge_cache.jsonl`：去重后的裁判原始输出；`manual_reference_audit.json`：参考答案核查。
- `gpu_monitor.jsonl`、`gpu_monitor.summary.json`：测试卡进程占用记录。

```bash
OMP_NUM_THREADS=2 /home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python \
  tools/benchmark_paddle.py \
  --target /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --draft /home/wjy/models/Qwen2.5-VL-7B-Instruct \
  --inputs-npz results/paddle_validation/ar_vs_lvspec_5cases/inputs_{0,1,2,3,4}.npz \
  --gpu-ids 2,3,4 --target-devices 0,1 --draft-devices 2 \
  --dtype float16 --max-new-tokens 512 --repeats 2 \
  --methods ar lvspec --interleave-methods \
  --output results/paddle_validation/ar_vs_lvspec_5cases

# 推理结束后，使用独立 Torch 环境进行文本评分；不计入推理耗时。
OMP_NUM_THREADS=2 /home/wjy/miniconda3/envs/lvspec/bin/python \
  tools/evaluate_paddle_comparison.py results/paddle_validation/ar_vs_lvspec_5cases \
  --cases results/paddle_validation/ar_vs_lvspec_5cases/cases.json \
  --judge-model /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --gpu-ids 3,4 --judge-device auto --judge-max-memory 34GiB
```

需要预先确认指定卡空闲。评分只读取保存的文本，推理本身仍完全在 Paddle 环境执行。
