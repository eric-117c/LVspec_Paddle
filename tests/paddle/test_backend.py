"""CPU regression checks for actual Paddle kernels and decoding state."""

import json

import numpy as np
import paddle
import pytest
from safetensors.numpy import save_file

from paddle_backend.cache import KVCache
from paddle_backend.checkpoint import convert_checkpoint, load_model
from paddle_backend.decoding import acceptance_lengths, drop_visual_tokens_uniform, generate, generate_tree_buffers, visual_similarity_hidden_state
from paddle_backend.llava import LlavaOnevisionForConditionalGeneration
from paddle_backend.models import Qwen2_5_VLForConditionalGeneration


def qwen_config():
    return dict(model_type="qwen2_5_vl", vocab_size=64, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
                video_token_id=61, image_token_id=60, eos_token_id=63,
                vision_config=dict(depth=2, hidden_size=16, intermediate_size=24, num_heads=2,
                                   patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                                   out_hidden_size=32, window_size=8, fullatt_block_indexes=[1]))


def llava_config():
    text = qwen_config().copy()
    for key in ("vision_config", "video_token_id", "image_token_id"):
        text.pop(key)
    text.update(model_type="qwen2", rope_scaling=None)
    return dict(model_type="llava_onevision", text_config=text, video_token_index=61,
                image_token_index=60, eos_token_id=63, vision_feature_layer=-1,
                vision_feature_select_strategy="full", projector_hidden_act="gelu",
                vision_config=dict(model_type="siglip_vision_model", image_size=8, patch_size=2,
                                   hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                   num_attention_heads=2))


@pytest.fixture(params=["qwen", "llava"])
def model(request):
    paddle.seed(123)
    if request.param == "qwen":
        return Qwen2_5_VLForConditionalGeneration(qwen_config(), "eager").eval()
    return LlavaOnevisionForConditionalGeneration(llava_config(), "eager").eval()


def test_cache_compaction_and_rollback():
    cache = KVCache(12)
    x = paddle.arange(8, dtype="float32").reshape([1, 1, 4, 2])
    cache.append(x, x + 10)
    cache.compact([3, 2], 2)
    np.testing.assert_array_equal(cache.key[0, 0, :4].numpy(), [[0, 1], [2, 3], [6, 7], [4, 5]])
    np.testing.assert_array_equal(cache.value[0, 0, :4].numpy(), [[10, 11], [12, 13], [16, 17], [14, 15]])
    cache.truncate(2)
    cache.append(x[:, :, :1], x[:, :, :1])
    assert cache.length == 3
    with pytest.raises(ValueError):
        cache.truncate(4)
    with pytest.raises(ValueError):
        cache.compact([9], 2)


def test_uniform_pruning_does_not_mutate_inputs():
    inputs = {"input_ids": paddle.to_tensor([[1, 61, 61, 61, 61, 2]]), "attention_mask": paddle.ones([1, 6])}
    output = drop_visual_tokens_uniform(inputs, 0.5, 61)
    assert output["input_ids"].tolist() == [[1, 61, 61, 2]]
    assert output["selected_indices"].tolist() == [0, 3]
    assert inputs["input_ids"].shape[-1] == 6
    assert drop_visual_tokens_uniform(inputs, 1, 61)["input_ids"].tolist() == [[1, 2]]
    with pytest.raises(ValueError):
        drop_visual_tokens_uniform(inputs, 1.1, 61)


def test_tree_ancestors_and_retrieval():
    tree = generate_tree_buffers([[0], [1], [0, 0], [1, 0]])
    assert tree["retrieve_indices"] == [[0, 1, 3], [0, 2, 4]]
    assert tree["mask"][3].tolist() == [True, True, False, True, False]
    with pytest.raises(ValueError):
        generate_tree_buffers([[0, 0]])


def test_loose_and_position_shift_acceptance():
    exact = paddle.to_tensor([[True, False, False, True]])
    scores = paddle.to_tensor([[0.9, 0.1, 0.8, 0.7]])
    assert acceptance_lengths(exact).tolist() == [1]
    assert acceptance_lengths(exact, scores, k=1).tolist() == [2]
    candidates = paddle.to_tensor([[1, 2, 3, 4, 5]])
    predicted = paddle.to_tensor([[2, 9, 5, 5]])
    assert acceptance_lengths(exact, scores, 1, candidates, predicted, pst=True).tolist() == [4]


def test_chunked_visual_similarity_matches_dense():
    from paddle_backend.decoding import _normalized_visual_chunks
    paddle.seed(45)
    visual, tree = paddle.randn([1, 19, 32]), paddle.randn([2, 5, 32])
    a = visual_similarity_hidden_state(visual, tree, chunk_size=4)
    b = visual_similarity_hidden_state(visual, tree, chunk_size=100)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)
    reference = _normalized_visual_chunks(visual, tree, chunk_size=4)
    for candidates in (tree, paddle.randn([1, 3, 32])):
        expected = visual_similarity_hidden_state(visual, candidates, chunk_size=4)
        cached = visual_similarity_hidden_state(visual, candidates, normalized_reference=reference)
        np.testing.assert_array_equal(cached.numpy(), expected.numpy())


@paddle.no_grad()
def test_cached_forward_matches_full_sequence(model):
    ids = paddle.to_tensor([[1, 2, 3, 4, 5]])
    full = model(input_ids=ids).logits.numpy()
    cache = model.new_cache(16)
    model(input_ids=ids[:, :3], cache=cache)
    tail = model(input_ids=ids[:, 3:], cache=cache).logits.numpy()
    np.testing.assert_allclose(tail, full[:, 3:], rtol=2e-5, atol=2e-5)
    assert cache.length == 5


@paddle.no_grad()
def test_tree_attention_matches_independent_branches(model):
    prompt = paddle.to_tensor([[1, 2, 3]])
    cache = model.new_cache(16)
    model(input_ids=prompt, cache=cache)
    tree = generate_tree_buffers([[0], [1], [0, 0], [1, 0]])
    tokens = [4, 5, 6, 7, 8]
    output = model(input_ids=paddle.to_tensor([tokens]), cache=cache,
                   position_ids=paddle.to_tensor([3 + d for d in tree["positions"]]),
                   tree_mask=paddle.to_tensor(tree["mask"]))
    for path in tree["retrieve_indices"]:
        branch = paddle.to_tensor([[1, 2, 3] + [tokens[i] for i in path]])
        expected = model(input_ids=branch).logits[0, 3:]
        np.testing.assert_allclose(output.logits[0, path].numpy(), expected.numpy(), rtol=2e-5, atol=2e-5)


@paddle.no_grad()
@pytest.mark.parametrize("tree", [[[0], [0, 0], [0, 0, 0]], [[0], [1], [0, 0], [1, 0]]])
def test_strict_speculative_matches_ar_and_caps_length(model, tree):
    inputs = {"input_ids": paddle.to_tensor([[1, 2, 3]])}
    ar = generate(inputs, model, method="ar", max_new_tokens=7, eos_token_id=[])
    sd = generate(inputs, model, model, method="specvlm_uniform", max_new_tokens=7, tree_choices=tree, eos_token_id=[])
    assert sd["output_ids"].tolist() == ar["output_ids"].tolist()
    assert sd["generate_len"] == 7
    assert sd["output_ids"].shape[-1] == 10


@paddle.no_grad()
def test_eos_and_empty_generation(model):
    inputs = {"input_ids": paddle.to_tensor([[1, 2]])}
    first = generate(inputs, model, method="ar", max_new_tokens=1, eos_token_id=[])["output_ids"][0, -1].item()
    for method in ("ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"):
        result = generate(inputs, model, model, method=method, max_new_tokens=7, eos_token_id=int(first), tree_choices=[[0], [0, 0]])
        assert result["generate_len"] == 1
        assert result["output_ids"][0, -1].item() == first
    assert generate(inputs, model, method="ar", max_new_tokens=0)["generate_len"] == 0


@paddle.no_grad()
def test_strict_verification_corrects_a_different_draft(model):
    paddle.seed(987)
    cls = Qwen2_5_VLForConditionalGeneration if model.model_type == "qwen2_5_vl" else LlavaOnevisionForConditionalGeneration
    draft = cls(model.raw_config, "eager").eval()
    inputs = {"input_ids": paddle.to_tensor([[1, 2, 3]])}
    ar = generate(inputs, model, method="ar", max_new_tokens=12, eos_token_id=[])
    sd = generate(inputs, model, draft, method="specvlm_uniform", max_new_tokens=12, eos_token_id=[],
                  tree_choices=[[0], [1], [2], [0, 0], [0, 0, 0], [1, 0]])
    assert sd["output_ids"].tolist() == ar["output_ids"].tolist()


@paddle.no_grad()
def test_chain_fastpath_preserves_tokens_and_acceptance(model):
    paddle.seed(489)
    cls = type(model)
    other = cls(model.raw_config, "eager").eval()
    inputs = {"input_ids": paddle.to_tensor([[1, 2, 3]])}
    tree = [[0] * depth for depth in range(1, 5)]
    for draft in (model, other):
        for method in ("specvlm_uniform", "lvspec", "lvspec_with_pst"):
            options = dict(method=method, max_new_tokens=23, eos_token_id=[], tree_choices=tree, loose_hidden_k=2)
            expected = generate(inputs, model, draft, use_chain_fastpath=False, **options)
            actual = generate(inputs, model, draft, use_chain_fastpath=True, **options)
            assert actual["output_ids"].tolist() == expected["output_ids"].tolist()
            assert actual["mean_accept_length"] == expected["mean_accept_length"]


@paddle.no_grad()
def test_video_frontend_pruning_and_generation(model):
    if model.model_type == "qwen2_5_vl":
        inputs = {"input_ids": paddle.to_tensor([[1, 61, 61, 61, 61, 2]]),
                  "pixel_values_videos": paddle.randn([16, 24]), "video_grid_thw": paddle.to_tensor([[1, 4, 4]])}
        count = 4
    else:
        inputs = {"input_ids": paddle.to_tensor([[1] + [61] * 9 + [2]]),
                  "pixel_values_videos": paddle.randn([1, 2, 3, 8, 8])}
        count = 9
    full = model(**inputs)
    assert full.video_hidden_states.shape[0] == count
    pruned = drop_visual_tokens_uniform(inputs, 0.5, 61)
    output = model(**pruned)
    np.testing.assert_allclose(output.video_hidden_states.numpy(),
                               full.video_hidden_states.numpy()[pruned["selected_indices"].numpy()], atol=1e-6)
    for method in ("ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"):
        result = generate(inputs, model, model, method=method, max_new_tokens=5, drop_rate=0.5,
                          tree_choices=[[0], [1], [0, 0], [1, 0]], eos_token_id=[])
        assert result["generate_len"] == 5


def test_hf_loading_transposes_square_linears_and_roundtrips(tmp_path):
    paddle.seed(11)
    source = Qwen2_5_VLForConditionalGeneration(qwen_config(), "eager").eval()
    arrays = {}
    linear_keys = {f"{name}.weight" for name, layer in source.named_sublayers() if isinstance(layer, paddle.nn.Linear)}
    for name, value in source.named_parameters():
        a = value.numpy()
        arrays[name] = a.T.copy() if name in linear_keys else a
    root = tmp_path / "hf"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(qwen_config()))
    save_file(arrays, root / "model.safetensors")
    loaded = load_model(root, dtype="float32", devices=["cpu"], attention_mode="eager")
    ids = paddle.to_tensor([[1, 2, 3]])
    np.testing.assert_array_equal(source(input_ids=ids).logits.numpy(), loaded(input_ids=ids).logits.numpy())
    converted = tmp_path / "pd"
    convert_checkpoint(root, converted)
    restored = load_model(converted, dtype="float32", devices=["cpu"], attention_mode="eager")
    np.testing.assert_array_equal(source(input_ids=ids).logits.numpy(), restored(input_ids=ids).logits.numpy())
    arrays.pop("model.layers.0.self_attn.q_proj.weight")
    save_file(arrays, root / "model.safetensors")
    with pytest.raises(ValueError, match="Missing checkpoint"):
        load_model(root, dtype="float32", devices=["cpu"])


@paddle.no_grad()
def test_qwen_short_vision_config_uses_hf_defaults():
    # The actual 32B config omits depth, num_heads, patch_size and window fields.
    config = qwen_config()
    config["vision_config"] = {"hidden_size": 128, "intermediate_size": 96,
                               "out_hidden_size": 32, "spatial_patch_size": 14, "in_chans": 3}
    model = Qwen2_5_VLForConditionalGeneration(config, "eager").eval()
    output = model(input_ids=paddle.to_tensor([[1, 61, 61, 61, 61, 2]]),
                   pixel_values_videos=paddle.randn([16, 1176]), video_grid_thw=paddle.to_tensor([[1, 4, 4]]))
    assert output.video_hidden_states.shape == [4, 32]
    assert bool(paddle.all(paddle.isfinite(output.logits)).item())
