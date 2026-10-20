"""Command-line video inference using native Paddle LVSpec models."""

import argparse
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["paddle"], default="paddle")
    p.add_argument("--model_type", choices=["qwen2_5_vl", "llava_ov"], default="qwen2_5_vl")
    p.add_argument("--base_model_path", required=True)
    p.add_argument("--draft_model_path")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data_path", help="Dataset directory or JSON/JSONL video manifest")
    source.add_argument("--video", help="One local video")
    source.add_argument("--inputs_npz", help="Already preprocessed arrays, shared with the PyTorch reference")
    p.add_argument("--prompt", default="Please provide a detailed description of the video.")
    p.add_argument("--task", default="VideoDetailCaption")
    p.add_argument("--frame_num", type=int, default=64)
    p.add_argument("--max_pixels", type=int, default=448 * 448)
    p.add_argument("--evaluation_num", type=int, default=1)
    p.add_argument("--data_num", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--drop_rate", type=float, default=0.9)
    p.add_argument("--loose_hidden_k", type=int, default=7)
    p.add_argument("--method", choices=["ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"], default="lvspec")
    p.add_argument("--tree_choice", choices=["chain", "chain_15", "chain2_8", "chain2_10", "mc_sim_7b_63"])
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--gpu_ids", default="0", help="Physical GPU IDs exposed to this process")
    p.add_argument("--target_devices", help="Comma-separated logical GPU IDs used by the target (default: all visible)")
    p.add_argument("--draft_devices", help="Comma-separated logical GPU IDs used by the draft (default: last visible GPU)")
    p.add_argument("--attention", choices=["sdpa", "eager", "reference"], default="sdpa")
    p.add_argument("--save_path", default="results/paddle_lvspec")
    p.add_argument("--save_inputs", action="store_true", help="Save each prepared NumPy input for exact reference comparisons")
    p.add_argument("--disable_cuda_graph", action="store_true", help="Use ordinary eager greedy drafting")
    p.add_argument("--pack_draft_projections", action="store_true", help="Optional QKV/MLP packing; uses extra draft GPU memory")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.evaluation_num < 1 or args.data_num < 1:
        raise ValueError("evaluation_num and data_num must be positive")
    if args.method != "ar" and not args.draft_model_path:
        raise ValueError("--draft_model_path is required for speculative decoding")
    # This module intentionally delays the first Paddle import until visibility is set.
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    import paddle
    from .checkpoint import load_model, load_model_pair, resolve_path
    from .decoding import generate
    from .inputs import load_processor, prepare_frames, read_samples, read_video, to_paddle
    if args.device == "gpu" and not paddle.is_compiled_with_cuda():
        raise RuntimeError("Install paddlepaddle-gpu for GPU inference, or use --device cpu --dtype float32")
    if args.device == "cpu":
        target_devices = draft_devices = ["cpu"]
    else:
        count = paddle.device.cuda.device_count()
        if count == 0:
            raise RuntimeError("No visible CUDA GPUs")
        def devices(value, default):
            indices = [int(i) for i in value.split(",")] if value else default
            if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= count for i in indices):
                raise ValueError("Invalid logical GPU device list")
            return [f"gpu:{i}" for i in indices]
        target_devices = devices(args.target_devices, list(range(count)))
        draft_devices = devices(args.draft_devices, [count - 1])
    if args.method != "ar":
        target, draft = load_model_pair(args.base_model_path, args.draft_model_path, args.model_type, args.dtype,
                                        target_devices, draft_devices, args.attention)
    else:
        target = load_model(args.base_model_path, args.model_type, args.dtype, target_devices, args.attention)
        draft = None
    processor = load_processor(str(resolve_path(args.base_model_path)), args.model_type, args.max_pixels)
    if draft is not None:
        if args.pack_draft_projections:
            from .optimization import pack_draft_projections
            pack_draft_projections(draft)
        draft_processor = load_processor(str(resolve_path(args.draft_model_path)), args.model_type, args.max_pixels)
        if processor.tokenizer.get_vocab() != draft_processor.tokenizer.get_vocab():
            raise ValueError("Target and draft tokenizer vocabularies differ")
        if target.video_token_id != draft.video_token_id:
            raise ValueError("Target and draft video token IDs differ")
        del draft_processor
    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"target_devices_resolved": target_devices, "draft_devices_resolved": draft_devices,
                           "paddle_version": paddle.__version__, "token_accounting": "generated suffix only; includes EOS; capped at max_new_tokens",
                           "position_encoding": "LVSpec sequential positions", "timing_scope": "decoding only; excludes model loading, preprocessing and prefill"}
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if args.inputs_npz:
        samples = [{"inputs_npz": args.inputs_npz}]
    elif args.video:
        samples = [{"video_path": args.video, "question": args.prompt}]
    else:
        samples = read_samples(args.data_path, args.task, args.data_num)
    tree = None
    if args.tree_choice:
        from tree_choices import choices
        tree = getattr(choices, args.tree_choice)
    records = []
    # Write each sample immediately so long runs retain completed results.
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as result_file:
        for index, sample in enumerate(samples):
            if index >= args.evaluation_num:
                break
            if "inputs_npz" in sample:
                with np.load(sample["inputs_npz"], allow_pickle=False) as arrays:
                    inputs = dict(arrays)
            else:
                frames, fps = read_video(sample["video_path"], args.frame_num)
                inputs = prepare_frames(processor, frames, sample["question"], args.model_type, fps)
            if args.save_inputs:
                np.savez(output_dir / f"inputs_{index}.npz", **inputs)
            result = generate(to_paddle(inputs, target_devices[0]), target, draft,
                              method=args.method, max_new_tokens=args.max_new_tokens, drop_rate=args.drop_rate,
                              loose_hidden_k=args.loose_hidden_k, tree_choices=tree,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              use_cuda_graph=not args.disable_cuda_graph)
            row = {"sample_index": index, "method": args.method, "backend": "paddle",
                   "decoding_time": result["decoding_time"], "inference_time": result["inference_time"],
                   "generate_len": result["generate_len"],
                   "output": processor.batch_decode(result["output_ids"].numpy(), skip_special_tokens=True)[0]}
            if args.method != "ar":
                row["average_accept_length"] = result["mean_accept_length"]
            result_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            result_file.flush()
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            records.append(row)
    if not records:
        raise ValueError("No valid video samples were found")
    tokens = sum(r["generate_len"] for r in records)
    seconds = sum(r["decoding_time"] for r in records)
    metric = {"method": args.method, "backend": "paddle", "samples": len(records),
              "avg_generate_len": tokens / len(records), "avg_decoding_time": seconds / len(records),
              "avg_tokens_per_second": tokens / seconds if seconds else None}
    if args.method != "ar":
        metric["avg_accept_length"] = sum(r["average_accept_length"] for r in records) / len(records)
    (output_dir / "metric.jsonl").write_text(json.dumps(metric) + "\n")


if __name__ == "__main__":
    main()
