"""Inference-only Qwen2.5-VL and LLaVA-OneVision models.

Architecture and checkpoint names follow the Apache-2.0 Qwen/Hugging Face
implementations vendored in ../models. Sequential language positions deliberately
match LVSpec's modified forwards, including after visual token pruning.
"""

from types import SimpleNamespace

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from .cache import ModelCache
from .ops import MLP, RMSNorm, additive_mask, attention, causal_mask, device_of, move, on_device, rotary, rotary_frequencies, rotate_half


def namespace(values, **defaults):
    defaults.update(values)
    return SimpleNamespace(**defaults)


def text_config(config):
    return namespace(config, hidden_act="silu", rms_norm_eps=1e-6, rope_theta=1000000.0,
                     rope_scaling=None, tie_word_embeddings=False, use_sliding_window=False,
                     attention_bias=True)


class TextAttention(nn.Layer):
    def __init__(self, config, mode):
        super().__init__()
        self.config, self.mode = config, mode
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        if config.hidden_size % self.heads or self.heads % self.kv_heads:
            raise ValueError("Invalid attention head or GQA dimensions")
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias_attr=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias_attr=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias_attr=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias_attr=False)

    def forward(self, x, positions, mask, cache, is_causal=False, frequencies=None):
        batch, length, _ = x.shape
        def heads(y, n):
            return y.reshape([batch, length, n, self.head_dim]).transpose([0, 2, 1, 3])
        packed = getattr(self, "_packed_qkv", None)
        if packed is None:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            q, k, v = paddle.split(F.linear(x, packed, getattr(self, "_packed_qkv_bias", None)),
                [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim], axis=-1)
        q, k, v = heads(q, self.heads), heads(k, self.kv_heads), heads(v, self.kv_heads)
        scaling = self.config.rope_scaling or {}
        q, k = rotary(q, k, positions, self.config.rope_theta, scaling.get("mrope_section"), scaling, frequencies)
        if cache is not None:
            k, v = cache.append(k, v)
        y = attention(q, k, v, mask, self.mode, is_causal=is_causal)
        return self.o_proj(y.transpose([0, 2, 1, 3]).reshape([batch, length, -1]))


class DecoderLayer(nn.Layer):
    def __init__(self, config, mode):
        super().__init__()
        self.self_attn = TextAttention(config, mode)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, positions, mask, cache, is_causal=False, frequencies=None):
        x = x + self.self_attn(self.input_layernorm(x), positions, mask, cache, is_causal=is_causal, frequencies=frequencies)
        return x + self.mlp(self.post_attention_layernorm(x))


class TextModel(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        if config.use_sliding_window:
            raise ValueError("Sliding-window checkpoints are not supported by the LVSpec tree mask")
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.LayerList([DecoderLayer(config, mode) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @on_device("embed_tokens.weight")
    def forward(self, input_ids=None, inputs_embeds=None, cache=None, position_ids=None, tree_mask=None, attention_mask=None):
        x = self.embed_tokens(move(input_ids, self.embed_tokens.weight)) if inputs_embeds is None else inputs_embeds
        if x.shape[0] != 1:
            raise ValueError("LVSpec currently supports batch_size=1")
        prefix = cache.length if cache is not None else 0
        length = x.shape[1]
        if attention_mask is not None and bool(paddle.all(attention_mask.astype("bool")).item()):
            attention_mask = None
        # Standard prefill can use fused causal attention without a quadratic
        # dense mask. One-token decode can attend to the entire cached prefix.
        implicit_mask = tree_mask is None and attention_mask is None and (prefix == 0 or length == 1)
        is_causal = implicit_mask and prefix == 0 and length > 1
        metadata = {}
        for i, layer in enumerate(self.layers):
            param = layer.input_layernorm.weight
            x = move(x, param)
            device = device_of(param)
            with paddle.device.device_guard(device):
                if device not in metadata:
                    pos = paddle.arange(prefix, prefix + length, dtype="int64") if position_ids is None else move(position_ids, param)
                    mask = None if implicit_mask else causal_mask(length, prefix, move(tree_mask, param), move(attention_mask, param))
                    if mask is not None and layer.self_attn.mode == "sdpa":
                        mask = additive_mask(mask, x.dtype)
                    scaling = self.config.rope_scaling or {}
                    frequencies = rotary_frequencies(pos, self.config.hidden_size // self.config.num_attention_heads,
                        self.config.rope_theta, scaling.get("mrope_section"), scaling, x.dtype)
                    metadata[device] = pos, mask, frequencies
                pos, mask, frequencies = metadata[device]
                x = layer(x, pos, mask, cache.layers[i] if cache is not None else None,
                          is_causal=is_causal, frequencies=frequencies)
        return self.norm(move(x, self.norm.weight))


class CausalLM(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.config = config
        self.model = TextModel(config, mode)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

    def forward(self, **kwargs):
        hidden = self.model(**kwargs)
        with paddle.device.device_guard(device_of(self.lm_head.weight)):
            return SimpleNamespace(logits=self.lm_head(move(hidden, self.lm_head.weight)), hidden_states=hidden)


class VisionAttention(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.mode = mode
        self.num_heads = config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, boundaries, cos, sin):
        length = x.shape[0]
        q, k, v = paddle.unstack(self.qkv(x).reshape([length, 3, self.num_heads, -1]), axis=1)
        q_dtype, k_dtype = q.dtype, k.dtype
        qf, kf = q.astype("float32"), k.astype("float32")
        q = (qf * cos.unsqueeze(1) + rotate_half(qf) * sin.unsqueeze(1)).astype(q_dtype)
        k = (kf * cos.unsqueeze(1) + rotate_half(kf) * sin.unsqueeze(1)).astype(k_dtype)
        chunks = []
        # Each frame/window is independent. Avoid allocating a full video x video mask.
        for start, end in zip(boundaries, boundaries[1:]):
            if start == end:
                continue
            heads = [y[start:end].transpose([1, 0, 2]).unsqueeze(0) for y in (q, k, v)]
            out = attention(*heads, mode=self.mode)
            chunks.append(out[0].transpose([1, 0, 2]).reshape([end - start, -1]))
        return self.proj(paddle.concat(chunks, axis=0))


class VisionBlock(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)
        self.attn = VisionAttention(config, mode)
        self.mlp = MLP(config, bias=True)

    def forward(self, x, boundaries, cos, sin):
        x = x + self.attn(self.norm1(x), boundaries, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        kernel = [config.temporal_patch_size, config.patch_size, config.patch_size]
        self.proj = nn.Conv3D(config.in_channels, config.hidden_size, kernel, stride=kernel, bias_attr=False)

    @on_device("proj.weight")
    def forward(self, x):
        c = self.config
        x = move(x, self.proj.weight).astype(self.proj.weight.dtype)
        if x.dtype != paddle.float32:
            return self.proj(x.reshape([-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size])).reshape([-1, c.hidden_size])
        # Each input row is exactly one nonoverlapping convolution patch.
        # The equivalent matrix product avoids cuDNN's reduced-precision
        # Conv3D path for FP32 inputs, even when TF32 flags are disabled.
        width = c.in_channels * c.temporal_patch_size * c.patch_size**2
        return paddle.matmul(x.reshape([-1, width]), self.proj.weight.reshape([c.hidden_size, width]),
                             transpose_y=True)


class PatchMerger(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.width = config.hidden_size * config.spatial_merge_size**2
        self.ln_q = RMSNorm(config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(self.width, self.width), nn.GELU(approximate=False), nn.Linear(self.width, config.out_hidden_size))

    @on_device("ln_q.weight")
    def forward(self, x):
        return self.mlp(self.ln_q(x).reshape([-1, self.width]))


class QwenVision(nn.Layer):
    def __init__(self, config, mode="sdpa"):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.LayerList([VisionBlock(config, mode) for _ in range(config.depth)])
        self.merger = PatchMerger(config)

    def geometry(self, grid):
        c = self.config
        merge, unit = c.spatial_merge_size, c.spatial_merge_size**2
        window = c.window_size // merge // c.patch_size
        if window < 1:
            raise ValueError("Vision window_size must cover at least one merged patch")
        positions, indices, full_lengths, window_lengths = [], [], [], []
        offset = 0
        for t, h, w in grid:
            t, h, w = int(t), int(h), int(w)
            if min(t, h, w) <= 0 or h % merge or w % merge:
                raise ValueError("Invalid visual grid or spatial merge dimensions")
            hh, ww = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            pos = np.stack([hh, ww], axis=-1).reshape(h // merge, merge, w // merge, merge, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2)
            positions.append(np.tile(pos, (t, 1)))
            gh, gw = h // merge, w // merge
            idx = np.arange(t * gh * gw).reshape(t, gh, gw)
            ph, pw = window - gh % window, window - gw % window
            idx = np.pad(idx, ((0, 0), (0, ph), (0, pw)), constant_values=-100)
            idx = idx.reshape(t, (gh + ph) // window, window, (gw + pw) // window, window).transpose(0, 1, 3, 2, 4).reshape(-1, window * window)
            counts = (idx != -100).sum(-1)
            window_lengths.extend((counts[counts > 0] * unit).tolist())
            indices.extend((idx[idx != -100] + offset).tolist())
            offset += t * gh * gw
            full_lengths.extend([h * w] * t)
        return np.concatenate(positions), indices, np.cumsum([0] + full_lengths).tolist(), np.cumsum([0] + window_lengths).tolist()

    @on_device("patch_embed.proj.weight")
    def forward(self, pixels, grid_thw):
        x = self.patch_embed(pixels)
        with paddle.device.device_guard(device_of(x)):
            positions, order, full, windows = self.geometry(grid_thw.numpy())
            if x.shape[0] != len(positions):
                raise ValueError("Visual pixels and grid sizes do not match")
            dim = self.config.hidden_size // self.config.num_heads // 2
            inv = 1.0 / (10000.0 ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
            freq = paddle.to_tensor(positions, dtype="float32").unsqueeze(-1) * inv
            freq = freq.reshape([x.shape[0], -1])
            unit = self.config.spatial_merge_size**2
            idx = paddle.to_tensor(order, dtype="int64")
            x = paddle.index_select(x.reshape([-1, unit, x.shape[-1]]), idx).reshape([x.shape[0], -1])
            freq = paddle.index_select(freq.reshape([-1, unit, freq.shape[-1]]), idx).reshape([x.shape[0], -1])
            emb = paddle.concat([freq, freq], axis=-1)
            cos, sin = paddle.cos(emb), paddle.sin(emb)
        for i, block in enumerate(self.blocks):
            param = block.norm1.weight
            x, cos, sin = move(x, param), move(cos, param), move(sin, param)
            with paddle.device.device_guard(device_of(param)):
                x = block(x, full if i in self.config.fullatt_block_indexes else windows, cos, sin)
        x = self.merger(move(x, self.merger.ln_q.weight))
        reverse = paddle.to_tensor(np.argsort(order), dtype="int64", place=x.place)
        return paddle.index_select(x, reverse)


class Qwen2_5_VLForConditionalGeneration(nn.Layer):
    model_type = "qwen2_5_vl"

    def __init__(self, config, attention_mode="sdpa"):
        super().__init__()
        self.raw_config = config
        self.config = text_config(config.get("text_config", config))
        vision = namespace(config["vision_config"], depth=32, hidden_size=3584, intermediate_size=3420,
                           num_heads=16, patch_size=14, in_channels=3, hidden_act="silu", spatial_merge_size=2,
                           temporal_patch_size=2, window_size=112, out_hidden_size=self.config.hidden_size,
                           fullatt_block_indexes=[7, 15, 23, 31])
        self.visual = QwenVision(vision, "eager" if attention_mode == "eager" else "sdpa")
        self.model = TextModel(self.config, attention_mode)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias_attr=False)
        self.video_token_id = config.get("video_token_id", 151656)
        self.image_token_id = config.get("image_token_id", 151655)

    @property
    def text_model(self):
        return self.model

    def new_cache(self, capacity):
        return ModelCache(self.config.num_hidden_layers, capacity)

    @on_device("model.embed_tokens.weight")
    def forward(self, input_ids, cache=None, pixel_values=None, pixel_values_videos=None,
                image_grid_thw=None, video_grid_thw=None, selected_indices=None,
                position_ids=None, tree_mask=None, attention_mask=None, logits_to_keep=0, **kwargs):
        if kwargs.keys() - {"second_per_grid_ts", "image_sizes_videos"}:
            raise TypeError(f"Unsupported model inputs: {sorted(kwargs)}")
        ids = move(input_ids, self.model.embed_tokens.weight)
        x = self.model.embed_tokens(ids)
        video = None
        for pixels, grid, token in ((pixel_values, image_grid_thw, self.image_token_id),
                                    (pixel_values_videos, video_grid_thw, self.video_token_id)):
            if pixels is None:
                continue
            features = self.visual(pixels, grid)
            if selected_indices is not None and token == self.video_token_id:
                features = paddle.index_select(features, move(selected_indices, features))
            features = move(features, x).astype(x.dtype)
            locations = paddle.nonzero(ids[0] == token).flatten()
            if len(locations) != features.shape[0]:
                raise ValueError(f"Visual token/feature mismatch: {len(locations)} != {features.shape[0]}")
            x[0, locations] = features
            if token == self.video_token_id:
                video = features
        hidden = self.model(inputs_embeds=x, cache=cache, position_ids=position_ids, tree_mask=tree_mask, attention_mask=attention_mask)
        with paddle.device.device_guard(device_of(self.lm_head.weight)):
            head_input = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
            logits = self.lm_head(move(head_input, self.lm_head.weight))
        return SimpleNamespace(logits=logits, hidden_states=hidden, video_hidden_states=video, output_embeddings=x)
