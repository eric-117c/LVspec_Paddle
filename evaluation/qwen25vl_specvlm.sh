export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.


TASK=camera_test,background_test,main_object_test,detailed_test #vdc
MODEL_PATH=/nfs/ycji/models/Qwen2.5-VL-32B-Instruct
DRAFT_MODEL_PATH=/nfs/ycji/models/Qwen2.5-VL-7B-Instruct
FRAME_NUM=64
MAX_LENGTH=512
DROP_RATE=0.9
SKIP=FALSE



WINDOW=2
LOOSE_ENTROPY=0.05
LOOSE_RANDOM=0.5
LOOSE_HIDDEN_K=7
SIMILARITY_K=10




METHOD=lvspec
TREE_CHOICE=chain
LOOSE_HIDDEN_K=7
SIMILARITY_K=10
OUTPUT_PATH=/nfs/ycji/code/lmms-eval/results/qwen25vl-32b-7b/vdc/lvspec
CUDA_VISIBLE_DEVICES=0,1 python -m lmms_eval \
    --model qwen2_5_vl_specvlm \
    --model_args=pretrained=$MODEL_PATH,draft_pretrained=$DRAFT_MODEL_PATH,max_num_frames=$FRAME_NUM,method=$METHOD,gen_length=$MAX_LENGTH,interleave_visuals=False,device_map=auto,result_path=$OUTPUT_PATH,drop_rate=$DROP_RATE,window=$WINDOW,tree_choice=$TREE_CHOICE,skip=$SKIP,loose_entropy=$LOOSE_ENTROPY,loose_random=$LOOSE_RANDOM,loose_hidden_score=$LOOSE_HIDDEN,loose_hidden_k=$LOOSE_HIDDEN_K,similarity_k=$SIMILARITY_K \
    --tasks $TASK \
    --batch_size 1 \
    --limit 30 \
    --log_samples \
    --output_path $OUTPUT_PATH






TASK=videomme,videomme_long
FRAME_NUM=64

METHOD=lvspec
TREE_CHOICE=chain
LOOSE_HIDDEN_K=7
SIMILARITY_K=10
OUTPUT_PATH=/nfs/ycji/code/lmms-eval/results/qwen25vl-32b-7b/videomme/lvspec
CUDA_VISIBLE_DEVICES=0,1 python -m lmms_eval \
    --model qwen2_5_vl_specvlm \
    --model_args=pretrained=$MODEL_PATH,draft_pretrained=$DRAFT_MODEL_PATH,max_num_frames=$FRAME_NUM,method=$METHOD,gen_length=$MAX_LENGTH,interleave_visuals=False,device_map=auto,result_path=$OUTPUT_PATH,drop_rate=$DROP_RATE,window=$WINDOW,tree_choice=$TREE_CHOICE,skip=$SKIP,loose_entropy=$LOOSE_ENTROPY,loose_random=$LOOSE_RANDOM,loose_hidden_score=$LOOSE_HIDDEN,loose_hidden_k=$LOOSE_HIDDEN_K,similarity_k=$SIMILARITY_K \
    --tasks $TASK \
    --batch_size 1 \
    --limit 500 \
    --log_samples \
    --output_path $OUTPUT_PATH






TASK=video_dc499
FRAME_NUM=128
SKIP=TRUE

METHOD=lvspec
TREE_CHOICE=chain
LOOSE_HIDDEN_K=7
SIMILARITY_K=10
OUTPUT_PATH=/nfs/ycji/code/lmms-eval/results/qwen25vl-32b-7b/vdd/lvspec
CUDA_VISIBLE_DEVICES=0,1 python -m lmms_eval \
    --model qwen2_5_vl_specvlm \
    --model_args=pretrained=$MODEL_PATH,draft_pretrained=$DRAFT_MODEL_PATH,max_num_frames=$FRAME_NUM,method=$METHOD,gen_length=$MAX_LENGTH,interleave_visuals=False,device_map=auto,result_path=$OUTPUT_PATH,drop_rate=$DROP_RATE,window=$WINDOW,tree_choice=$TREE_CHOICE,skip=$SKIP,loose_entropy=$LOOSE_ENTROPY,loose_random=$LOOSE_RANDOM,loose_hidden_score=$LOOSE_HIDDEN,loose_hidden_k=$LOOSE_HIDDEN_K,similarity_k=$SIMILARITY_K \
    --tasks $TASK \
    --batch_size 1 \
    --limit 30 \
    --log_samples \
    --output_path $OUTPUT_PATH








TASK=moviechat_global
FRAME_NUM=128



METHOD=lvspec
TREE_CHOICE=chain
LOOSE_HIDDEN_K=7
SIMILARITY_K=10
OUTPUT_PATH=/nfs/ycji/code/lmms-eval/results/qwen25vl-32b-7b/moviechat/lvspec
CUDA_VISIBLE_DEVICES=0,1 python -m lmms_eval \
    --model qwen2_5_vl_specvlm \
    --model_args=pretrained=$MODEL_PATH,draft_pretrained=$DRAFT_MODEL_PATH,max_num_frames=$FRAME_NUM,method=$METHOD,gen_length=$MAX_LENGTH,interleave_visuals=False,device_map=auto,result_path=$OUTPUT_PATH,drop_rate=$DROP_RATE,window=$WINDOW,tree_choice=$TREE_CHOICE,skip=$SKIP,loose_entropy=$LOOSE_ENTROPY,loose_random=$LOOSE_RANDOM,loose_hidden_score=$LOOSE_HIDDEN,loose_hidden_k=$LOOSE_HIDDEN_K,similarity_k=$SIMILARITY_K \
    --tasks $TASK \
    --batch_size 1 \
    --limit 100 \
    --log_samples \
    --output_path $OUTPUT_PATH



