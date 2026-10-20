"""Strict HF safetensors loading and optional Paddle checkpoint conversion."""

import json
from pathlib import Path

import paddle
from paddle import nn
from safetensors import safe_open

from .llava import LlavaOnevisionForConditionalGeneration
from .models import Qwen2_5_VLForConditionalGeneration
from .ops import device_of


def resolve_path(path):
    local = Path(path).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(path, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]))


def place_model(model, devices):
    """Explicit contiguous layer sharding; devices are logical CUDA IDs."""
    if not devices or any(d != "cpu" and not d.startswith("gpu:") for d in devices):
        raise ValueError("devices must contain cpu or gpu:<logical index>")
    if "cpu" in devices and len(devices) != 1:
        raise ValueError("Mixing CPU offload and GPU sharding is not supported")
    text = model.text_model
    text.embed_tokens.to(device=devices[0])
    for i, layer in enumerate(text.layers):
        layer.to(device=devices[min(i * len(devices) // len(text.layers), len(devices) - 1)])
    text.norm.to(device=devices[-1])
    if model.model_type == "qwen2_5_vl":
        model.visual.to(device=devices[0])
        model.lm_head.to(device=devices[-1])
    else:
        model.vision_tower.to(device=devices[0])
        model.multi_modal_projector.to(device=devices[0])
        model.language_model.lm_head.to(device=devices[-1])
        # The newline vector is transferred alongside features in the forward.
    model.devices = tuple(devices)


def load_hf_weights(model, path):
    path = Path(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        if any(Path(shard).name != shard for shard in shards):
            raise ValueError("Checkpoint shards must be files within the checkpoint directory")
    elif (path / "model.safetensors").exists():
        weight_map, shards = None, ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No HF safetensors checkpoint found in {path}")
    parameters = dict(model.named_parameters())
    linear_keys = {f"{name}.weight" for name, layer in model.named_sublayers() if isinstance(layer, nn.Linear)}
    loaded, ignored = set(), []
    for shard in shards:
        print(f"Loading {shard}", flush=True)
        with paddle.device.device_guard("cpu"), safe_open(path / shard, framework="paddle", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if weight_map is not None and weight_map.get(key) != shard:
                    raise ValueError(f"Checkpoint index mismatch for {key}")
                if key in loaded:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                if key not in parameters:
                    # SigLIP's contrastive pooling head is never used for video features.
                    if key.startswith("vision_tower.vision_model.head.") or key.endswith("rotary_emb.inv_freq"):
                        ignored.append(key)
                        continue
                    raise ValueError(f"Unexpected checkpoint tensor: {key}")
                parameter = parameters[key]
                device = device_of(parameter) if parameter._is_initialized() else "cpu"
                # Paddle MmapStorage is host storage. Reading a CPU safetensor
                # under a nonzero CUDA device can block in get_slice().
                with paddle.device.device_guard("cpu"):
                    tensor = checkpoint.get_tensor(key)
                with paddle.device.device_guard(device):
                    tensor = tensor.to(device)
                    if key in linear_keys:
                        tensor = tensor.transpose([1, 0])
                    if list(tensor.shape) != list(parameter.shape):
                        raise ValueError(f"Checkpoint shape mismatch for {key}: {tensor.shape} != {parameter.shape}")
                    parameter.set_value(tensor.astype(parameter.dtype))
                loaded.add(key)
                del tensor
    missing = set(parameters) - loaded
    if model.config.tie_word_embeddings:
        prefix = "" if model.model_type == "qwen2_5_vl" else "language_model."
        head, embed = prefix + "lm_head.weight", prefix + "model.embed_tokens.weight"
        if head in missing and embed in loaded:
            device = device_of(parameters[head]) if parameters[head]._is_initialized() else "cpu"
            with paddle.device.device_guard(device):
                parameters[head].set_value(parameters[embed].to(device).transpose([1, 0]))
            missing.remove(head)
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
    return {"loaded": len(loaded), "ignored": ignored}


def _prepare_model(path, model_type, dtype, devices, attention_mode):
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    root = resolve_path(path)
    config = json.loads((root / "config.json").read_text())
    detected = {"qwen2_5_vl": "qwen2_5_vl", "llava_onevision": "llava_ov"}.get(config.get("model_type"))
    model_type = model_type or detected
    if model_type != detected or detected is None:
        raise ValueError(f"Unsupported/mismatched model_type: {config.get('model_type')} / {model_type}")
    cls = Qwen2_5_VLForConditionalGeneration if model_type == "qwen2_5_vl" else LlavaOnevisionForConditionalGeneration
    devices = devices or ["gpu:0" if paddle.is_compiled_with_cuda() else "cpu"]
    if not devices or any(device != "cpu" and not device.startswith("gpu:") for device in devices):
        raise ValueError("devices must contain cpu or gpu:<logical index>")
    if "cpu" in devices and devices != ["cpu"]:
        raise ValueError("Mixing CPU offload and GPU sharding is not supported")
    if devices == ["cpu"] and dtype != "float32":
        raise ValueError("CPU inference requires --dtype float32")
    if devices != ["cpu"] and not paddle.is_compiled_with_cuda():
        raise RuntimeError("GPU inference requires paddlepaddle-gpu")
    previous_dtype = paddle.get_default_dtype()
    previous_device = paddle.get_device()
    try:
        # Define shapes without filling random 7B/32B parameters. Load weights
        # on the host before moving any layers to CUDA.
        paddle.set_device("cpu")
        paddle.set_default_dtype(dtype)
        with paddle.LazyGuard():
            model = cls(config, attention_mode)
        model.devices = tuple(devices)
        model.checkpoint_path = str(root)
        return model
    finally:
        paddle.set_default_dtype(previous_dtype)
        paddle.set_device(previous_device)


def _load_prepared(model, place=True):
    root = Path(model.checkpoint_path)
    native = root / "model_state.pdparams"
    if native.exists():
        manifest = json.loads((root / "paddle_backend.json").read_text())
        if manifest.get("format_version") != 1 or manifest.get("model_type") != model.model_type:
            raise ValueError("Incompatible Paddle checkpoint manifest")
        with paddle.device.device_guard("cpu"):
            state = paddle.load(str(native), return_numpy=True)
        expected = model.state_dict()
        if set(state) != set(expected):
            raise ValueError("Paddle checkpoint keys do not match model")
        for name, value in state.items():
            if list(value.shape) != list(expected[name].shape):
                raise ValueError(f"Paddle checkpoint shape mismatch: {name}")
        for name, parameter in model.named_parameters():
            with paddle.device.device_guard("cpu"):
                parameter.set_value(paddle.to_tensor(state[name]).astype(parameter.dtype))
    else:
        model.loading_info = load_hf_weights(model, root)
    if place:
        place_model(model, list(model.devices))
    return model.eval()


def load_model(path, model_type=None, dtype="float16", devices=None, attention_mode="sdpa"):
    return _load_prepared(_prepare_model(path, model_type, dtype, devices, attention_mode))


def load_model_pair(target_path, draft_path, model_type=None, dtype="float16", target_devices=None,
                    draft_devices=None, attention_mode="sdpa"):
    """Load both checkpoints on the host before placing their CUDA layers."""
    target = _prepare_model(target_path, model_type, dtype, target_devices, attention_mode)
    draft = _prepare_model(draft_path, model_type, dtype, draft_devices, attention_mode)
    _load_prepared(target, place=False)
    _load_prepared(draft, place=False)
    place_model(target, list(target.devices))
    place_model(draft, list(draft.devices))
    return target, draft


def convert_checkpoint(source, destination, model_type=None, dtype="float32"):
    """Convert locally, without executing PyTorch or tracing the generation loop."""
    import shutil
    source = resolve_path(source)
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Conversion destination must be empty")
    model = load_model(source, model_type, dtype=dtype, devices=["cpu"])
    destination.mkdir(parents=True, exist_ok=True)
    paddle.save(model.state_dict(), str(destination / "model_state.pdparams"))
    for pattern in ("*.json", "*.txt", "*.model", "*.jinja"):
        for path in source.glob(pattern):
            if path.name != "model.safetensors.index.json":
                shutil.copy2(path, destination / path.name)
    (destination / "paddle_backend.json").write_text(json.dumps(
        {"format_version": 1, "model_type": model.model_type, "dtype": dtype,
         "source": str(source), "linear_layout": "in_out"}, indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert HF LVSpec model weights to Paddle layout")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--model-type", choices=["qwen2_5_vl", "llava_ov"])
    args = parser.parse_args()
    convert_checkpoint(args.source, args.destination, args.model_type)
