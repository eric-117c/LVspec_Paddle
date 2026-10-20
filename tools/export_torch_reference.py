"""Export tiny reference checkpoints/outputs with the existing PyTorch backend.

Run in the original LVSpec environment, then use tools/verify_paddle.py in the
Paddle environment. The two frameworks need not be installed together.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", help="Optional real HF checkpoint; skips tiny fixture creation")
    p.add_argument("--inputs-npz", help="Exact preprocessed inputs for --checkpoint")
    p.add_argument("--model-type", choices=["qwen2_5_vl", "llava_ov"], default="qwen2_5_vl")
    args = p.parse_args()
    import numpy as np
    import torch
    # Full FP32 reference: cuDNN otherwise permits TF32 for the real vision tower.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from safetensors.torch import save_file
    from models.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    from models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from models.modeling_llava_onevision_tree import LlavaOnevisionForConditionalGeneration
    from transformers import LlavaOnevisionConfig

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        if not args.inputs_npz:
            p.error("--inputs-npz is required with --checkpoint")
        cls = Qwen2_5_VLForConditionalGeneration if args.model_type == "qwen2_5_vl" else LlavaOnevisionForConditionalGeneration
        model = cls.from_pretrained(args.checkpoint, torch_dtype=torch.float32,
                                   device_map=args.device, attn_implementation="eager", low_cpu_mem_usage=True).eval()
        with np.load(args.inputs_npz, allow_pickle=False) as arrays:
            numpy_inputs = dict(arrays)
        inputs = {key: torch.from_numpy(value).to(args.device) for key, value in numpy_inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        directory = root / ("qwen" if args.model_type == "qwen2_5_vl" else "llava")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source.json").write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve())}))
        np.savez(directory / "inputs.npz", **numpy_inputs)
        values = dict(logits=output.logits.cpu().numpy(), hidden_states=output.hidden_states.cpu().numpy(),
                      input_embeddings=output.output_embeddings.cpu().numpy())
        if output.video_hidden_states is not None:
            values["video_features"] = output.video_hidden_states.cpu().numpy()
        np.savez(directory / "reference.npz", **values)
        print(f"Exported real checkpoint reference: {directory}")
        return
    qwen = dict(model_type="qwen2_5_vl", vocab_size=64, hidden_size=32, intermediate_size=48,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
                video_token_id=61, image_token_id=60, eos_token_id=63,
                vision_config=dict(depth=2, hidden_size=16, intermediate_size=24, num_heads=2,
                                   patch_size=2, temporal_patch_size=2, spatial_merge_size=2,
                                   out_hidden_size=32, window_size=8, fullatt_block_indexes=[1]))
    text = {key: value for key, value in qwen.items() if key not in ("vision_config", "video_token_id", "image_token_id")}
    text.update(model_type="qwen2", rope_scaling=None)
    llava = dict(model_type="llava_onevision", text_config=text, video_token_index=61,
                 image_token_index=60, eos_token_id=63, vision_feature_layer=-1,
                 vision_feature_select_strategy="full", projector_hidden_act="gelu",
                 vision_config=dict(model_type="siglip_vision_model", image_size=8, patch_size=2,
                                    hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                    num_attention_heads=2))
    for name, raw, config_cls, model_cls in (("qwen", qwen, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration),
                                             ("llava", llava, LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration)):
        torch.manual_seed(123)
        config = config_cls(**raw)
        config._attn_implementation = "eager"
        config.vision_config._attn_implementation = "eager"
        if name == "llava":
            config.text_config._attn_implementation = "eager"
        model = model_cls(config).to(args.device).eval()
        directory = root / name
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, directory / "model.safetensors")
        if name == "qwen":
            inputs = dict(input_ids=torch.tensor([[1, 61, 61, 61, 61, 2]], device=args.device),
                          pixel_values_videos=torch.randn(16, 24, device=args.device),
                          video_grid_thw=torch.tensor([[1, 4, 4]], device=args.device))
        else:
            inputs = dict(input_ids=torch.tensor([[1] + [61] * 9 + [2]], device=args.device),
                          pixel_values_videos=torch.randn(1, 2, 3, 8, 8, device=args.device))
        with torch.no_grad():
            output = model(**inputs)
        np.savez(directory / "inputs.npz", **{k: v.cpu().numpy() for k, v in inputs.items()})
        np.savez(directory / "reference.npz", logits=output.logits.cpu().numpy(),
                 video_features=output.video_hidden_states.cpu().numpy(),
                 hidden_states=output.hidden_states.cpu().numpy(),
                 input_embeddings=output.output_embeddings.cpu().numpy())
        print(f"Exported {name}: {directory}")


if __name__ == "__main__":
    main()
