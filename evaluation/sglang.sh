CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /data/ycji/models/Meta-Llama-3-8B-Instruct \
    --tp 1 \
    --mem-fraction-static 0.9 \
    --port 30000 \
    --enable-deterministic-inference
    # --random-seed 1