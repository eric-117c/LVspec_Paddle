"""Optional real CUDA checks, including transfer of KV storage between layers.

CUDA_VISIBLE_DEVICES=3,4 python -m pytest -q tests/paddle/test_gpu.py
"""

import numpy as np
import paddle
import pytest
import json
import subprocess
import sys
from pathlib import Path

from safetensors.numpy import save_file

from paddle_backend.checkpoint import place_model
from paddle_backend.decoding import generate
from paddle_backend.llava import LlavaOnevisionForConditionalGeneration
from paddle_backend.models import Qwen2_5_VLForConditionalGeneration
from paddle_backend.ops import device_of
from test_backend import llava_config, qwen_config

pytestmark = pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="Requires paddlepaddle-gpu")


def test_checkpoint_pair_reload_after_cuda_initialization(tmp_path):
    if paddle.device.cuda.device_count() < 2:
        pytest.skip("Requires two visible GPUs")
    model = Qwen2_5_VLForConditionalGeneration(qwen_config()).eval()
    linear_keys = {f"{name}.weight" for name, layer in model.named_sublayers()
                   if isinstance(layer, paddle.nn.Linear)}
    arrays = {name: value.numpy().T.copy() if name in linear_keys else value.numpy()
              for name, value in model.named_parameters()}
    save_file(arrays, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(qwen_config()))
    # A subprocess timeout turns a loader deadlock into an actionable failure.
    script = """
import sys
import numpy as np
import paddle
from paddle_backend.checkpoint import load_model, load_model_pair
from paddle_backend.ops import device_of
root = sys.argv[1]
first = load_model(root, dtype='float32', devices=['gpu:0', 'gpu:1'])
target, draft = load_model_pair(root, root, dtype='float32',
    target_devices=['gpu:0', 'gpu:1'], draft_devices=['gpu:1'])
with paddle.device.device_guard('gpu:0'):
    inputs = paddle.to_tensor([[1, 2, 3]], dtype='int64')
with paddle.no_grad():
    expected = first(input_ids=inputs).logits.numpy()
    for model in (target, draft):
        output = model(input_ids=inputs).logits
        assert device_of(output) == 'gpu:1'
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-5, rtol=1e-5)
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True,
                   timeout=60, cwd=Path(__file__).resolve().parents[2])


@pytest.mark.parametrize("family", ["qwen", "llava"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@paddle.no_grad()
def test_two_gpu_video_decoding_and_cache(family, dtype):
    if paddle.device.cuda.device_count() < 2:
        pytest.skip("Requires two visible GPUs")
    previous_device, previous_dtype = paddle.get_device(), paddle.get_default_dtype()
    try:
        paddle.set_device("gpu:0")
        paddle.set_default_dtype(dtype)
        paddle.seed(456)
        if family == "qwen":
            model = Qwen2_5_VLForConditionalGeneration(qwen_config()).eval()
            inputs = {"input_ids": paddle.to_tensor([[1, 61, 61, 61, 61, 2]], dtype="int64"),
                      "pixel_values_videos": paddle.randn([16, 24], dtype="float32"),
                      "video_grid_thw": paddle.to_tensor([[1, 4, 4]], dtype="int64")}
        else:
            model = LlavaOnevisionForConditionalGeneration(llava_config()).eval()
            inputs = {"input_ids": paddle.to_tensor([[1] + [61] * 9 + [2]], dtype="int64"),
                      "pixel_values_videos": paddle.randn([1, 2, 3, 8, 8], dtype="float32")}
        reference = model(**inputs).logits.astype("float32").numpy()
        place_model(model, ["gpu:0", "gpu:1"])
        cache = model.new_cache(32)
        output = model(**inputs, cache=cache)
        with paddle.device.device_guard(device_of(output.logits)):
            actual = output.logits.astype("float32").numpy()
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
        assert cache.layers[0].key.place.gpu_device_id() == 0
        assert cache.layers[-1].key.place.gpu_device_id() == 1
        ar = generate(inputs, model, method="ar", max_new_tokens=5, eos_token_id=[])
        for method in ("specvlm_uniform", "lvspec", "lvspec_with_pst"):
            result = generate(inputs, model, model, method=method, max_new_tokens=5, drop_rate=1,
                              tree_choices=[[0], [1], [0, 0], [1, 0]], eos_token_id=[])
            assert result["generate_len"] == 5
            if method == "specvlm_uniform":
                assert result["output_ids"].tolist() == ar["output_ids"].tolist()
    finally:
        paddle.set_default_dtype(previous_dtype)
        paddle.set_device(previous_device)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@paddle.no_grad()
def test_graph_drafting_rollbacks_and_fully_accepted_chain(dtype):
    from paddle_backend.decoding import _draft_chain
    from paddle_backend.draft_graph import DraftGraph
    previous_device, previous_dtype = paddle.get_device(), paddle.get_default_dtype()
    try:
        paddle.set_device("gpu:0")
        paddle.set_default_dtype(dtype)
        paddle.seed(2026)
        config = qwen_config()
        config.update(hidden_size=512, num_attention_heads=4, num_key_value_heads=1,
                      intermediate_size=768, rope_scaling={"type": "mrope", "mrope_section": [16, 24, 24]})
        config["vision_config"]["out_hidden_size"] = 512
        model = Qwen2_5_VLForConditionalGeneration(config).eval()
        normal, captured = model.new_cache(48), model.new_cache(48)
        for cache in (normal, captured):
            model(input_ids=paddle.to_tensor([[1, 2, 3]]), cache=cache)
        graph = DraftGraph(model, captured, 17, 5)
        for root, prefix in ((17, 3), (31, 6), (21, 11)):
            normal.truncate(prefix)
            captured.truncate(prefix)
            expected = _draft_chain(model, normal, root, 5)
            actual = graph.replay(root, prefix)
            assert actual.tolist() == expected.tolist()
            tolerance = 2e-2 if dtype == "bfloat16" else 2e-3
            for left, right in zip(normal.layers, captured.layers):
                np.testing.assert_allclose(left.key[:, :, :normal.length].astype("float32").numpy(),
                    right.key[:, :, :normal.length].astype("float32").numpy(), atol=tolerance, rtol=tolerance)
            # Exercise filling the deferred leaf through the normal decoder.
            for cache in (normal, captured):
                model(input_ids=actual[:, -1:], cache=cache)
    finally:
        paddle.set_default_dtype(previous_dtype)
        paddle.set_device(previous_device)
