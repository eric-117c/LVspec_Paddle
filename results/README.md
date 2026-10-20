# 测试记录与输入生成

本目录发布小体积的 JSON/JSONL 测试结果、运行配置、参考描述、日志和监控记录。
完整说明见 [综合测试报告](../docs/TEST_REPORT.md) 与 [交付报告](../docs/DELIVERY_REPORT.md)。

主要结果入口：

- [原始 PyTorch 三卡比较](qwen2_5_vl_3gpu_original_kv/speed_comparison.json)
- [Paddle 优化前三卡比较](paddle_validation/ar_vs_lvspec_5cases/comparison.json)
- [Paddle 优化后三卡受干扰比较](paddle_optimization/three_gpu_20261020_063411/comparison.json)
- [Paddle 优化后两卡完整比较](paddle_optimization/two_gpu_clean_20261020_065626/comparison.json)

模型权重、原视频和 `.npz` 中间数组不随仓库上传。历史记录中的绝对路径对应测试机器，
接手时应替换为自己的模型与数据路径。`source_sha256.json` 保存实测时的源码哈希，
后续代码清理不会回写历史哈希；清理记录位于 `maintenance/` 的 manifest 文件。

## 重新生成五视频输入

先按 [Paddle 使用说明](../docs/PADDLE.md) 安装环境，准备 Qwen2.5-VL-32B/7B 权重和
VideoDetailCaption 数据集。确认指定的 A100 80GB 显卡空闲，再执行：

```bash
PYTHON=python GPU_IDS=0,1 METHOD=ar EVAL_NUM=5 FRAME_NUM=64 MAX_NEW_TOKENS=1 \
BASE_MODEL_PATH=/path/to/Qwen2.5-VL-32B-Instruct \
DRAFT_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
DATA_PATH=/path/to/VideoDetailCaption SAVE_PATH=results/local_inputs \
bash run_qwenvl_paddle.sh --target_devices 0,1 --save_inputs
```

这一步运行 AR，仅用于得到 `results/local_inputs/inputs_0.npz` 至 `inputs_4.npz`；
一次生成 1 token，不作为性能基准。样本选择、帧数、processor 配置和依赖版本会影响输入，
严格复现时应与历史配置及 SHA256 核对。

随后使用 [优化版完整复测](../docs/PADDLE_OPTIMIZATION_TEST.md) 的基准命令，
将输入参数改为 `--inputs-npz results/local_inputs/inputs_{0,1,2,3,4}.npz`，
并使用新的输出目录以及本机模型路径。参考问题和描述保存在
[cases.json](paddle_validation/ar_vs_lvspec_5cases/cases.json)，正式质量评估仍需检查样本对应关系。
