"""Attention and rotary kernels shared by the two model families."""

import math
from functools import wraps

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.incubate.nn.functional import fused_rms_norm

USE_FUSED_RMS_NORM = True


def device_of(tensor):
    """Normalize Tensor.place to a device string for Paddle's device_guard.

    In Paddle 3.3.1, passing a generic libpaddle.Place (as returned by
    Tensor.place) to device_guard can reset allocations to GPU 0. A string
    resolves to the concrete CUDAPlace and preserves the requested GPU ID.
    """
    place = tensor.place
    if place.is_gpu_place():
        return f"gpu:{place.gpu_device_id()}"
    if place.is_cpu_place():
        return "cpu"
    raise ValueError(f"Unsupported tensor device: {place}")


def on_device(path, index=0, keyword=None):
    """Execute tensor operations on the device of an argument's parameter."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            value = kwargs[keyword] if keyword and keyword in kwargs else args[index]
            for part in path.split("."):
                value = value[part] if isinstance(value, dict) else getattr(value, part)
            with paddle.device.device_guard(device_of(value)):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def move(x, parameter):
    return x.to(parameter.place) if x is not None and x.place != parameter.place else x


def activation(name, x):
    if name in ("gelu_pytorch_tanh", "gelu_new", "gelu_fast"):
        return F.gelu(x, approximate=True)
    if name == "gelu":
        return F.gelu(x, approximate=False)
    if name in ("silu", "swish"):
        return F.silu(x)
    raise ValueError(f"Unsupported activation: {name}")


class RMSNorm(nn.Layer):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = self.create_parameter([width], default_initializer=nn.initializer.Constant(1))
        self.eps = eps

    @on_device("weight")
    def forward(self, x):
        if USE_FUSED_RMS_NORM and x.place.is_gpu_place() and x.dtype in (paddle.float16, paddle.bfloat16):
            # Keep the original two rounding points: normalize -> cast to the
            # input dtype -> multiply by weight. A weighted fused RMSNorm would
            # fuse the last multiplication and change low-precision semantics.
            unit = getattr(self, "_unit_scale", None)
            if unit is None or unit.place != self.weight.place or unit.dtype != self.weight.dtype:
                unit = paddle.ones_like(self.weight)
                self.register_buffer("_unit_scale", unit, persistable=False)
            return fused_rms_norm(x, unit, None, self.eps, begin_norm_axis=x.ndim - 1)[0] * self.weight
        y = x.astype("float32")
        y = y * paddle.rsqrt(paddle.mean(y * y, axis=-1, keepdim=True) + self.eps)
        return y.astype(x.dtype) * self.weight


class MLP(nn.Layer):
    def __init__(self, config, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias_attr=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=bias)
        self.act = config.hidden_act

    @on_device("gate_proj.weight")
    def forward(self, x):
        packed = getattr(self, "_packed_gate_up", None)
        if packed is None:
            gate, up = self.gate_proj(x), self.up_proj(x)
        else:
            gate, up = paddle.split(F.linear(x, packed, getattr(self, "_packed_gate_up_bias", None)), 2, axis=-1)
        return self.down_proj(activation(self.act, gate) * up)


def rotate_half(x):
    a, b = paddle.chunk(x, 2, axis=-1)
    return paddle.concat([-b, a], axis=-1)


def rotary_frequencies(positions, dim, theta, sections=None, scaling=None, dtype="float32"):
    scaling = scaling or {}
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if kind not in ("default", "mrope", "linear"):
        raise ValueError(f"Unsupported rope scaling: {kind}")
    with paddle.device.device_guard(device_of(positions)):
        inv = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype="float32") / dim))
        if kind == "linear":
            inv = inv / scaling["factor"]
        pos = positions.astype("float32")
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        if pos.ndim == 2:
            pos = pos.unsqueeze(0).expand([3, -1, -1])
        freqs = pos.unsqueeze(-1) * inv
        emb = paddle.concat([freqs, freqs], axis=-1)
        cos, sin = paddle.cos(emb), paddle.sin(emb)
        if sections:
            if sum(sections) * 2 != dim:
                raise ValueError("mrope_section does not match attention head dimension")
            cos = paddle.concat([part[i % 3] for i, part in enumerate(paddle.split(cos, sections * 2, axis=-1))], axis=-1)
            sin = paddle.concat([part[i % 3] for i, part in enumerate(paddle.split(sin, sections * 2, axis=-1))], axis=-1)
        else:
            cos, sin = cos[0], sin[0]
        return cos.unsqueeze(1).astype(dtype), sin.unsqueeze(1).astype(dtype)


def rotary(q, k, positions, theta, sections=None, scaling=None, frequencies=None):
    cos, sin = frequencies if frequencies is not None else rotary_frequencies(
        move(positions, q), q.shape[-1], theta, sections, scaling, q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def attention(q, k, v, mask=None, mode="sdpa", is_causal=False):
    """Inputs use [batch, heads, sequence, dim]; bool masks mean allowed."""
    if mode == "sdpa":
        if mask is not None and mask.dtype == paddle.bool:
            mask = additive_mask(mask, q.dtype)
        out = F.scaled_dot_product_attention(
            q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]),
            v.transpose([0, 2, 1, 3]), attn_mask=mask,
            dropout_p=0.0, is_causal=is_causal, training=False,
        )
        return out.transpose([0, 2, 1, 3])
    groups = q.shape[1] // k.shape[1]
    if groups != 1:
        k = paddle.repeat_interleave(k, groups, axis=1)
        v = paddle.repeat_interleave(v, groups, axis=1)
    if mode not in ("eager", "reference"):
        raise ValueError(f"Unknown attention mode: {mode}")
    if is_causal:
        mask = paddle.arange(k.shape[-2]).unsqueeze(0) <= paddle.arange(q.shape[-2]).unsqueeze(1)
    # reference reproduces this repository's BF16 Q/K -> FP16 scores path.
    score_q = q.astype("bfloat16") if mode == "reference" else q
    score_k = k.astype("bfloat16") if mode == "reference" else k
    scores = paddle.matmul(score_q, score_k, transpose_y=True) / math.sqrt(q.shape[-1])
    scores = scores.astype("float16" if mode == "reference" else "float32")
    if mask is not None:
        scores = paddle.where(mask, scores, paddle.full_like(scores, float("-inf")))
    probs = F.softmax(scores, axis=-1).astype(v.dtype)
    return paddle.matmul(probs, v)


def additive_mask(mask, dtype):
    """Avoid SDPA's host-created scalar tensors for every boolean-mask call."""
    return paddle.where(mask, paddle.zeros_like(mask, dtype=dtype),
                        paddle.full_like(mask, float("-inf"), dtype=dtype))


def causal_mask(query_length, prefix_length, tree_mask=None, padding_mask=None):
    total = prefix_length + query_length
    allowed = paddle.arange(total).unsqueeze(0) <= (prefix_length + paddle.arange(query_length)).unsqueeze(1)
    if tree_mask is not None:
        if list(tree_mask.shape) == [query_length, total]:
            allowed = tree_mask.astype("bool")
        elif list(tree_mask.shape) == [query_length, query_length]:
            allowed[:, prefix_length:] = tree_mask
        else:
            raise ValueError("tree_mask shape must match the complete speculative suffix")
    allowed = allowed.unsqueeze(0).unsqueeze(0)
    if padding_mask is not None:
        if padding_mask.shape[-1] != total:
            raise ValueError("attention_mask must cover cached prefix and current input")
        allowed = allowed & padding_mask.astype("bool").unsqueeze(1).unsqueeze(1)
    return allowed
