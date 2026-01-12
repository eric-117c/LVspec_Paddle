export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

MODEL_TYPE="llava_ov"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="llava-onevision-qwen2-7b-ov-hf"  # Path to base model
DRAFT_MODEL_PATH="llava-onevision-qwen2-7b-ov-hf"  # Path to draft model

# MODEL_TYPE="qwen2_5_vl"             # Model type: llava_ov or qwen2_5_vl
# BASE_MODEL_PATH="Qwen2.5-VL-7B-Instruct"  # Path to base model
# DRAFT_MODEL_PATH="Qwen2.5-VL-7B-Instruct"  # Path to draft model

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="VideoDetailCaption"  # Path to dataset

EVAL_NUM=100                     # Number of evaluation samples
MAX_NEW_TOKENS=512                # Number of new tokens to generate
DATA_NUM=100                      # Number of data samples to load
DROP_RATE=0.9                    # Pruning ratio
GPU_IDS="0,1"                   # GPU IDs to use

FRAME_NUM=256       

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_IDS python run.py \
    --model_type $MODEL_TYPE \
    --base_model_path $BASE_MODEL_PATH \
    --draft_model_path $DRAFT_MODEL_PATH \
    --data_path $DATA_PATH \
    --task $TASK \
    --frame_num $FRAME_NUM \
    --evaluation_num $EVAL_NUM \
    --max_new_tokens $MAX_NEW_TOKENS \
    --drop_rate $DROP_RATE \
    --data_num $DATA_NUM \
    --gpu_ids $GPU_IDS \
    --save_path "results/" \
    --setting "standard" \
