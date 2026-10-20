export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

MODEL_TYPE="llava_ov"             # Model type: llava_ov or qwen2_5_vl
BASE_MODEL_PATH="/ycji/models/llava-onevision-qwen2-72b-ov-hf"  # Path to base model
DRAFT_MODEL_PATH="/ycji/models/llava-onevision-qwen2-7b-ov-hf"  # Path to draft model

TASK="VideoDetailCaption"         # Task type: VideoDetailCaption, MVBench, MVLU, LongVideoBench, MMBench
DATA_PATH="/home/wjy/datasets/VideoDetailCaption"  # Path to dataset

EVAL_NUM=1                       # Number of evaluation samples
MAX_NEW_TOKENS=512               # Number of new tokens to generate
DATA_NUM=100                     # Number of data samples to load
DROP_RATE=0.9                    # Pruning ratio
GPU_IDS="0,1"                    # GPU IDs to use

METHOD="lvspec"                  # [ar, specvlm_uniform, lvspec, lvspec_with_pst]
LOOSE_HIDDEN_K=7                 # Loose parameter for relaxed verification

FRAME_NUM=64      

# Run evaluation
CUDA_VISIBLE_DEVICES=$GPU_IDS python inference.py \
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
    --method $METHOD \
    --loose_hidden_k $LOOSE_HIDDEN_K \
    --save_path "results/{$MODEL_TYPE}_{$METHOD}_drop{$DROP_RATE}_frame{$FRAME_NUM}" \
    # --setting "standard" \
