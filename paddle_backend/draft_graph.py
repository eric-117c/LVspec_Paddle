"""CUDA graph for fixed-width greedy drafting with device-side KV positions."""

import numpy as np
import paddle
from paddle.device.cuda.graphs import CUDAGraph

from .ops import additive_mask, attention, device_of, rotary, rotary_frequencies


class DraftGraph:
    """Capture a whole draft block, without Python work between its tokens.

    Prompt KV remains in the ordinary ModelCache allocations. Dynamic GPU
    indices update only the current position; masked, zero-filled tail storage
    keeps attention shapes fixed across replays and speculative rollbacks.
    """

    def __init__(self, model, cache, root, count):
        self.model, self.cache, self.count = model, cache, count
        self.device = device_of(model.text_model.embed_tokens.weight)
        self.capacity = cache.layers[0].capacity
        with paddle.device.device_guard(self.device):
            # Keep both graph inputs in one persistent allocation. Updating it
            # from NumPy avoids two temporary CUDA tensors and device copies.
            self.inputs = paddle.to_tensor([root, cache.length], dtype="int64")
            self.token = self.inputs[:1].reshape([1, 1])
            self.position = self.inputs[1:]
            self.offsets = paddle.arange(count - 1, dtype="int64")
            self.key_positions = paddle.arange(self.capacity, dtype="int64")
            config = model.config
            scaling = config.rope_scaling or {}
            self.cos, self.sin = rotary_frequencies(self.key_positions,
                config.hidden_size // config.num_attention_heads, config.rope_theta,
                scaling.get("mrope_section"), scaling, model.text_model.embed_tokens.weight.dtype)
            self.cos, self.sin = self.cos[0, 0], self.sin[0, 0]
            self.keys, self.values = [], []
            for layer in cache.layers:
                layer.key[:, :, layer.length:, :] = 0
                layer.value[:, :, layer.length:, :] = 0
                key = layer.key.transpose([2, 0, 1, 3]).contiguous()
                value = layer.value.transpose([2, 0, 1, 3]).contiguous()
                self.keys.append(key)
                self.values.append(value)
                layer.key = key.transpose([1, 2, 0, 3])
                layer.value = value.transpose([1, 2, 0, 3])
            # Warm up kernels and initialize runtime-only norm buffers before capture.
            self._block()
            paddle.device.synchronize(self.device)
            self.graph = CUDAGraph(place=paddle.CUDAPlace(int(self.device.split(":")[1])))
            self.graph.capture_begin()
            self.output = self._block()
            self.graph.capture_end()

    def _step(self, token, position, frequencies, mask):
        model = self.model
        x = model.text_model.embed_tokens(token)
        for block, cache, key_storage, value_storage in zip(
                model.text_model.layers, self.cache.layers, self.keys, self.values):
            residual = x
            y = block.input_layernorm(x)
            attn = block.self_attn
            packed = getattr(attn, "_packed_qkv", None)
            if packed is None:
                q, k, v = attn.q_proj(y), attn.k_proj(y), attn.v_proj(y)
            else:
                q, k, v = paddle.split(paddle.nn.functional.linear(y, packed,
                    getattr(attn, "_packed_qkv_bias", None)),
                    [attn.heads * attn.head_dim, attn.kv_heads * attn.head_dim, attn.kv_heads * attn.head_dim], axis=-1)
            q = q.reshape([1, attn.heads, 1, attn.head_dim])
            k = k.reshape([1, attn.kv_heads, 1, attn.head_dim])
            v = v.reshape([1, attn.kv_heads, 1, attn.head_dim])
            q, k = rotary(q, k, position, model.config.rope_theta, frequencies=frequencies)
            key_storage.scatter_(position, k.transpose([2, 0, 1, 3]))
            value_storage.scatter_(position, v.transpose([2, 0, 1, 3]))
            y = attention(q, cache.key, cache.value, mask, mode=attn.mode)
            x = residual + attn.o_proj(y.reshape([1, 1, -1]))
            x = x + block.mlp(block.post_attention_layernorm(x))
        hidden = model.text_model.norm(x)
        head = model.lm_head if model.model_type == "qwen2_5_vl" else model.language_model.lm_head
        return paddle.topk(head(hidden)[0, -1], 1)[1].reshape([1, 1])

    def _block(self):
        token = self.token
        tokens = [token]
        # All positions are known before drafting; prepare their metadata in
        # one batch rather than launching the same small kernels per token.
        positions = self.position + self.offsets
        frequencies = tuple(paddle.index_select(table, positions).unsqueeze(0).unsqueeze(0)
                            for table in (self.cos, self.sin))
        masks = additive_mask(self.key_positions.unsqueeze(0) <= positions.unsqueeze(1),
                              self.model.text_model.embed_tokens.weight.dtype)
        for offset in range(self.count - 1):
            token = self._step(token, positions[offset:offset + 1],
                tuple(table[:, :, offset:offset + 1] for table in frequencies),
                masks[offset:offset + 1].reshape([1, 1, 1, self.capacity]))
            tokens.append(token)
        return paddle.concat(tokens, axis=1)

    def replay(self, root, prefix):
        with paddle.device.device_guard(self.device):
            self.inputs.set_value(np.asarray([root, prefix], dtype=np.int64))
            self.graph.replay()
            for layer in self.cache.layers:
                layer.length = prefix + self.count - 1
            return self.output
