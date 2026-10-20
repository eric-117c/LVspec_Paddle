"""Runtime-only projection packing; original checkpoint parameters stay intact."""

import paddle

from .ops import device_of


@paddle.no_grad()
def pack_draft_projections(model):
    """Merge Q/K/V and gate/up GEMMs for the sequential draft model.

    Packed buffers are excluded from checkpoints and follow Layer.to(). The
    original parameters remain available for conversion and reference checks.
    """
    for layer in model.text_model.layers:
        attention, mlp = layer.self_attn, layer.mlp
        with paddle.device.device_guard(device_of(attention.q_proj.weight)):
            attention.register_buffer("_packed_qkv", paddle.concat(
                [attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight], axis=1), persistable=False)
            if attention.q_proj.bias is not None:
                attention.register_buffer("_packed_qkv_bias", paddle.concat(
                    [attention.q_proj.bias, attention.k_proj.bias, attention.v_proj.bias]), persistable=False)
        with paddle.device.device_guard(device_of(mlp.gate_proj.weight)):
            mlp.register_buffer("_packed_gate_up", paddle.concat(
                [mlp.gate_proj.weight, mlp.up_proj.weight], axis=1), persistable=False)
            if mlp.gate_proj.bias is not None:
                mlp.register_buffer("_packed_gate_up_bias", paddle.concat(
                    [mlp.gate_proj.bias, mlp.up_proj.bias]), persistable=False)
    return model
