export HF_HOME="~/.cache/huggingface"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

TASK=camera_test,background_test,main_object_test,detailed_test #vdc
MODEL_PATH=/nfs/ycji/models/llava-onevision-qwen2-72b-ov-hf
DRAFT_MODEL_PATH=/nfs/ycji/models/llava-onevision-qwen2-7b-ov-hf
FRAME_NUM=64
MAX_LENGTH=512
DROP_RATE=0.9
SKIP=FALSE


WINDOW=2
LOOSE_ENTROPY=0.05
LOOSE_RANDOM=0.5
LOOSE_HIDDEN_K=3
SIMILARITY_K=10



METHOD=lvspec
TREE_CHOICE=chain
LOOSE_HIDDEN_K=7
SIMILARITY_K=10
OUTPUT_PATH=/nfs/ycji/code/lmms-eval/results/llava-onevision-72b-7b/vdc/lvspec
CUDA_VISIBLE_DEVICES=0,1 python -m lmms_eval \
    --model llava_onevision_specvlm \
    --model_args=pretrained=$MODEL_PATH,draft_pretrained=$DRAFT_MODEL_PATH,max_frames_num=$FRAME_NUM,method=$METHOD,gen_length=$MAX_LENGTH,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen,result_path=$OUTPUT_PATH,drop_rate=$DROP_RATE,window=$WINDOW,tree_choice=$TREE_CHOICE,skip=$SKIP,loose_entropy=$LOOSE_ENTROPY,loose_random=$LOOSE_RANDOM,loose_hidden_score=$LOOSE_HIDDEN,loose_hidden_k=$LOOSE_HIDDEN_K,similarity_k=$SIMILARITY_K,loose_rank=$LOOSE_RANK \
    --tasks $TASK \
    --batch_size 1 \
    --limit 30 \
    --log_samples \
    --output_path $OUTPUT_PATH

