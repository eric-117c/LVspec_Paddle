"""Exercise video decoding, the actual HF processor, and both CLI entry points."""

import json
import os
import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import paddle
from safetensors.numpy import save_file

from paddle_backend.inputs import load_processor, prepare_frames, read_video
from paddle_backend.models import Qwen2_5_VLForConditionalGeneration


def test_video_processor_and_cli(tmp_path):
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    from models.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    vocab = {character: i for i, character in enumerate(bytes_to_unicode().values())}
    specials = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>"]
    vocab.update({token: len(vocab) + i for i, token in enumerate(specials)})
    (checkpoint / "vocab.json").write_text(json.dumps(vocab))
    (checkpoint / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = Qwen2TokenizerFast(vocab_file=str(checkpoint / "vocab.json"), merges_file=str(checkpoint / "merges.txt"), eos_token="<|im_end|>", pad_token="<|im_end|>")
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    image_processor = Qwen2VLImageProcessor(min_pixels=16, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2)
    template = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{{ messages[0]['content'][1]['text'] }}<|im_end|>\n<|im_start|>assistant\n"
    processor = Qwen2_5_VLProcessor(image_processor, tokenizer, chat_template=template)
    processor.save_pretrained(checkpoint)
    config = dict(model_type="qwen2_5_vl", vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=1024, rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
                  video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                  image_token_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                  eos_token_id=tokenizer.eos_token_id,
                  vision_config=dict(depth=2, hidden_size=16, intermediate_size=24, num_heads=2,
                                     patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                                     out_hidden_size=32, window_size=8, fullatt_block_indexes=[1]))
    (checkpoint / "config.json").write_text(json.dumps(config))
    model = Qwen2_5_VLForConditionalGeneration(config)
    linear_keys = {f"{name}.weight" for name, layer in model.named_sublayers() if isinstance(layer, paddle.nn.Linear)}
    arrays = {name: value.numpy().T.copy() if name in linear_keys else value.numpy() for name, value in model.named_parameters()}
    save_file(arrays, checkpoint / "model.safetensors")
    video = tmp_path / "video.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(4):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 30 + i * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    loaded_processor = load_processor(checkpoint, "qwen2_5_vl", max_pixels=64)
    frames, fps = read_video(video, 2)
    prepared = prepare_frames(loaded_processor, frames, "Describe.", "qwen2_5_vl", fps)
    assert prepared["video_grid_thw"].shape == (1, 3)
    root = Path(__file__).resolve().parents[2]
    env = os.environ | {"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    for method, entry in (("ar", [str(root / "inference.py"), "--backend", "paddle"]),
                          ("lvspec_with_pst", ["-m", "paddle_backend.inference"])):
        output = tmp_path / method
        command = [sys.executable, *entry, "--model_type", "qwen2_5_vl", "--base_model_path", str(checkpoint),
                   "--draft_model_path", str(checkpoint), "--video", str(video), "--prompt", "Describe.",
                   "--device", "cpu", "--dtype", "float32", "--max_pixels", "64", "--frame_num", "2",
                   "--max_new_tokens", "3", "--method", method, "--save_inputs", "--save_path", str(output)]
        completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        row = json.loads((output / "results.jsonl").read_text())
        assert row["backend"] == "paddle" and 1 <= row["generate_len"] <= 3
        assert row["decoding_time"] > 0
        assert (output / "inputs_0.npz").exists()
        assert json.loads((output / "metric.jsonl").read_text())["samples"] == 1
        assert json.loads((output / "run_config.json").read_text())["target_devices_resolved"] == ["cpu"]
    benchmark = tmp_path / "benchmark"
    command = [sys.executable, str(root / "tools/benchmark_paddle.py"), "--target", str(checkpoint),
               "--inputs-npz", str(tmp_path / "ar/inputs_0.npz"), "--target-devices", "cpu",
               "--dtype", "float32", "--methods", "ar", "lvspec", "--repeats", "2",
               "--interleave-methods", "--max-new-tokens", "3", "--output", str(benchmark)]
    completed = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for method in ("ar", "lvspec"):
        rows = [json.loads(line) for line in (benchmark / method / "results.jsonl").read_text().splitlines()]
        assert [row["sample_index"] for row in rows] == [0, 1]
        assert [row["repeat"] for row in rows] == [0, 1]
        assert all(row["generate_len"] == len(row["generated_ids"]) for row in rows)
    order = [json.loads(line)["method"] for line in completed.stdout.splitlines() if line.startswith('{"sample_index"')]
    assert order == ["ar", "lvspec", "lvspec", "ar"]
    assert len(json.loads((benchmark / "run_config.json").read_text())["input_sha256"]) == 1
