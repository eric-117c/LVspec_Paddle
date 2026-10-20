# Paddle 优化版完整复测

全部历史与当前测试的汇总、失败项及结论边界见 [综合测试报告](TEST_REPORT.md)。

验证日期：2026-10-20。

**两卡、五视频、每方法两遍的完整测试中，LVSpec 相对同版 AR 的解码吞吐为 2.19×；
计入 prefill 后为 1.63×。质量代理评分 AR 3.6/5、LVSpec 3.4/5，不能宣称准确率无损。**

## 测试条件

- Qwen2.5-VL-32B-Instruct 目标模型，单独放在物理 GPU 0；7B 草稿模型放在物理 GPU 4。
- 两张 NVIDIA A100 80GB，Paddle GPU 3.3.1，FP16，greedy，batch size 1。
- 使用此前完整测试的五份固定 NPZ，每视频 64 帧；输入 SHA256 与旧测试全部一致。
- 生成上限 512 token；AR、LVSpec 各预热，再交替顺序执行，每视频各重复两次，共 20 次推理。
- `drop_rate=0.9`、`loose_hidden_k=7`，默认 greedy chain；启用草稿 CUDA Graph 和投影打包。
- AR 与 LVSpec 使用相同的优化版目标模型。解码时间包含每条请求建立 CUDA Graph 的开销。
- 185 次、约每 5 秒的显卡进程采样未观察到其他计算进程占用测试卡。
  采样不能排除间隔中的短任务或共享主机、互连资源的影响。
- 回归检查：CPU 后端/CLI/指标检查 26 项，GPU 检查 9 项，共 **35 passed**。
  GPU 检查包含 FP16/BF16 CUDA Graph 回滚及完整接受路径。

## 速度

| 指标 | AR | LVSpec | LVSpec / AR |
| --- | ---: | ---: | ---: |
| 解码吞吐，token/s | 13.49 | 29.60 | **2.19×** |
| 含 prefill 吞吐，token/s | 12.11 | 19.68 | **1.63×** |
| 平均含 prefill 推理耗时，秒 | 42.28 | 19.98 | 耗时比 2.12× |
| 平均生成长度，token | 512.0 | 393.2 | 回答长度不同 |
| 达到生成上限的次数 | 10/10 | 4/10 | 部分回答被截断 |

吞吐均按总生成 token 数除以总时间计算。2.12× 的耗时比包含回答更短的影响，
不能替代按 token 归一化的吞吐比。含 prefill 的推理时间仍不包含模型加载及视频预处理，
所以本轮没有证明完整端到端吞吐达到两倍。

| 视频 | 输入 token 数 | 两遍聚合解码加速比 | AR / LVSpec 生成长度 |
| --- | ---: | ---: | ---: |
| `v_wCexKIuAj3A`：电视节目 | 3211 | 2.07× | 512 / 317 |
| `v_FqYhC6_CMfI`：风帆冲浪 | 7435 | 2.02× | 512 / 181 |
| `v_w4pO-1-FG5w`：理发 | 8107 | 2.16× | 512 / 512 |
| `v_4iFlM_Zu7xs`：越野摩托车 | 8107 | 2.30× | 512 / 444 |
| `v_4Gx_8vuxzAk`：木桌清洁 | 7531 | 2.36× | 512 / 512 |

两次重复中，每种方法在同一视频上的 token IDs 均完全一致。
以上只支持这五个固定视频的结论，不代表完整数据集或其他配置。

## 回答质量

评审沿用本地 Qwen2.5-VL-32B 文本权重、BF16、greedy 的参考描述辅助评分。
每次只评一条回答，不提供方法标签。相同回答按检查点、配置、评分规则及文本哈希复用评分，
质量样本仍是五个独立视频，不是十个。对照中参考原文得 5/5、明显无关描述得 0/5。

| 质量代理指标 | AR | LVSpec |
| --- | ---: | ---: |
| 正确性，0–5 | 3.60 | 3.40 |
| 内容覆盖度，0–5 | 3.60 | 3.40 |
| 词级 ROUGE-L F1 | 0.1604 | 0.1926 |

裁判与目标共享模型权重，存在自评偏差；ROUGE-L 也不是事实准确率。
保存回答与人工参考描述的文字核查发现：LVSpec 漏掉电子烟主题，两种方法都漏掉冲浪末尾落水、
理发器连接吸发装置等细节；木桌清洁中 LVSpec 把液体施加位置写成抹布，与参考中的桌面不同。
自动裁判也有误读，例如声称 AR 提及了保存回答中并不存在的落水或酒吧切镜。
因此不能将分数相近或 ROUGE-L 更高解释成准确率无损。核查没有直接观看视频。

## 三卡运行与对比范围

本轮先完成了目标在物理 GPU 0、4、草稿在 GPU 5 的三卡 20 次推理。
GPU 5 中途出现其他用户的计算进程，监控已记录，整批解码吞吐比为 1.94×。
这批计时标记为受干扰，不能作为三卡无干扰性能结论。
随后才执行上述两卡复测，两个配置保存的全部输出 token IDs 完全一致。

三卡受干扰目录：`results/paddle_optimization/three_gpu_20261020_063411/`。
其中 `timing_validity.json`、`comparison.json` 和 `status.json` 均保留受干扰标记。

此前 PyTorch 三卡五视频为 1.88×，Paddle 优化前三卡五视频为 1.26×，
详见 [初版对比](PADDLE_COMPARISON.md)。当前两卡结果的设备放置不同，且两框架的输入/数值路径
未完成严格统一，不能据此宣称 Paddle 比 PyTorch 快。
尚需三张空闲卡完成同配置无干扰复测，才能回答三卡优化后的性能。

## 结果与复现

完整结果：`results/paddle_optimization/two_gpu_clean_20261020_065626/`。

- `comparison.json`、`speed_report.json`：聚合及逐视频指标、评分及原始理由。
- `ar/results.jsonl`、`lvspec/results.jsonl`：全部生成文本、token IDs 和耗时。
- `run_config.json`、`command.json`、`source_sha256.json`：参数、输入哈希及代码哈希。
- `integrity_checks.json`：用例完整性、输入哈希、输出长度、重复一致性和代码哈希检查。
- `gpu_monitor.jsonl`、`gpu_monitor.summary.json`：显卡采样记录。
- `validation_checks.json`、`manual_reference_audit.json`：回归结果与参考描述核查。
- `judge_cache_seed.json`：复用评分的来源；`judge_cache.jsonl`：评分缓存。

```bash
cd /home/wjy/workFiles/LVSpec
PY=/home/wjy/miniconda3/envs/lvspec-paddle-gpu/bin/python
OUT="results/paddle_optimization/retest_$(date +%Y%m%d_%H%M%S)"

# 执行前确认物理 GPU 0、4 空闲；使用新目录保存结果。
OMP_NUM_THREADS=2 "$PY" tools/benchmark_paddle.py \
  --target /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --draft /home/wjy/models/Qwen2.5-VL-7B-Instruct \
  --inputs-npz results/paddle_validation/ar_vs_lvspec_5cases/inputs_{0,1,2,3,4}.npz \
  --gpu-ids 0,4 --target-devices 0 --draft-devices 1 \
  --dtype float16 --max-new-tokens 512 --repeats 2 \
  --methods ar lvspec --interleave-methods --pack-draft-projections \
  --output "$OUT"

OMP_NUM_THREADS=2 /home/wjy/miniconda3/envs/lvspec/bin/python \
  tools/evaluate_paddle_comparison.py "$OUT" \
  --cases results/paddle_validation/ar_vs_lvspec_5cases/cases.json \
  --judge-model /home/wjy/models/Qwen2.5-VL-32B-Instruct \
  --gpu-ids 0,4 --judge-device auto --judge-max-memory 34GiB
```
