#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python}" inference.py --backend paddle \
  --model_type qwen2_5_vl \
  --base_model_path "${BASE_MODEL_PATH:-/home/wjy/models/Qwen2.5-VL-32B-Instruct}" \
  --draft_model_path "${DRAFT_MODEL_PATH:-/home/wjy/models/Qwen2.5-VL-7B-Instruct}" \
  --data_path "${DATA_PATH:-/home/wjy/datasets/VideoDetailCaption}" \
  --gpu_ids "${GPU_IDS:-0,1,2,3}" \
  --method "${METHOD:-lvspec}" --dtype "${DTYPE:-float16}" \
  --frame_num "${FRAME_NUM:-64}" --evaluation_num "${EVAL_NUM:-5}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" --drop_rate "${DROP_RATE:-0.9}" \
  --save_path "${SAVE_PATH:-results/paddle_qwen_${METHOD:-lvspec}}" "$@"
