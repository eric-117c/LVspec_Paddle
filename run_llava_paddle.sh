#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to a LLaVA-OneVision HF checkpoint}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to a compatible draft checkpoint}"
exec "${PYTHON:-python}" inference.py --backend paddle \
  --model_type llava_ov --base_model_path "$BASE_MODEL_PATH" \
  --draft_model_path "$DRAFT_MODEL_PATH" \
  --data_path "${DATA_PATH:-/home/wjy/datasets/VideoDetailCaption}" \
  --gpu_ids "${GPU_IDS:-0,1}" --method "${METHOD:-lvspec}" \
  --dtype "${DTYPE:-float16}" --frame_num "${FRAME_NUM:-64}" \
  --evaluation_num "${EVAL_NUM:-1}" --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  --drop_rate "${DROP_RATE:-0.9}" \
  --save_path "${SAVE_PATH:-results/paddle_llava_${METHOD:-lvspec}}" "$@"
