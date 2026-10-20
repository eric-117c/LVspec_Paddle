"""Run all LVSpec modes on identical saved inputs without reloading model weights."""

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", help="Defaults to sharing target weights for a smoke test")
    parser.add_argument("--inputs-npz", nargs="+", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--target-devices", default="0", help="Comma-separated logical GPU IDs, or cpu")
    parser.add_argument("--draft-devices", default="0")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--interleave-methods", action="store_true",
                        help="Alternate method order per input/repeat to reduce ordering bias")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--pack-draft-projections", action="store_true")
    parser.add_argument("--drop-rate", type=float, default=0.9)
    parser.add_argument("--loose-hidden-k", type=int, default=7)
    parser.add_argument("--methods", nargs="+", default=["ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"],
                        choices=["ar", "specvlm_uniform", "lvspec", "lvspec_with_pst"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    import numpy as np
    import paddle
    from paddle_backend.checkpoint import load_model, load_model_pair
    from paddle_backend.decoding import generate
    from paddle_backend.inputs import load_processor, to_paddle
    def devices(value):
        return ["cpu"] if value == "cpu" else [f"gpu:{int(i)}" for i in value.split(",")]
    if args.draft:
        target, draft = load_model_pair(args.target, args.draft, dtype=args.dtype,
                                        target_devices=devices(args.target_devices), draft_devices=devices(args.draft_devices))
    else:
        target = load_model(args.target, dtype=args.dtype, devices=devices(args.target_devices))
        draft = target
    if args.pack_draft_projections:
        if draft is target:
            parser.error("Projection packing requires a separate --draft checkpoint")
        from paddle_backend.optimization import pack_draft_projections
        pack_draft_projections(draft)
    processor = load_processor(target.checkpoint_path, target.model_type)
    if args.draft:
        draft_processor = load_processor(draft.checkpoint_path, draft.model_type)
        if processor.tokenizer.get_vocab() != draft_processor.tokenizer.get_vocab():
            raise ValueError("Target and draft tokenizers differ")
    inputs_list = []
    for path in args.inputs_npz:
        with np.load(path, allow_pickle=False) as arrays:
            inputs_list.append(to_paddle(dict(arrays), devices(args.target_devices)[0]))
    # Warm up target and draft kernels before collecting either method's timings.
    for method in args.methods:
        generate(inputs_list[0], target, draft, method=method, max_new_tokens=2,
                 drop_rate=args.drop_rate, loose_hidden_k=args.loose_hidden_k,
                 use_cuda_graph=not args.disable_cuda_graph)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_config.json").write_text(json.dumps(vars(args) | {"paddle_version": paddle.__version__,
        "shared_target_draft": not bool(args.draft), "timing_scope": "decoding only; excludes prefill and preprocessing",
        "input_sha256": [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.inputs_npz]}, indent=2) + "\n")
    summaries = []
    records_by_method = {method: [] for method in args.methods}
    for method in args.methods:
        directory = root / method
        directory.mkdir(exist_ok=True)
    if args.interleave_methods:
        schedule = [(method, repeat, index) for repeat in range(args.repeats)
                    for index in range(len(inputs_list))
                    for method in (args.methods if (repeat + index) % 2 == 0 else args.methods[::-1])]
    else:
        schedule = [(method, repeat, index) for method in args.methods
                    for repeat in range(args.repeats) for index in range(len(inputs_list))]
    with ExitStack() as stack:
        files = {method: stack.enter_context((root / method / "results.jsonl").open("w")) for method in args.methods}
        for method, repeat, index in schedule:
            inputs = inputs_list[index]
            result = generate(inputs, target, draft, method, max_new_tokens=args.max_new_tokens,
                              drop_rate=args.drop_rate, loose_hidden_k=args.loose_hidden_k,
                              eos_token_id=processor.tokenizer.eos_token_id,
                              use_cuda_graph=not args.disable_cuda_graph)
            suffix = result["output_ids"][:, inputs["input_ids"].shape[-1]:].numpy()
            row = dict(sample_index=repeat * len(inputs_list) + index, input_index=index, repeat=repeat,
                       method=method, backend="paddle",
                       decoding_time=result["decoding_time"], inference_time=result["inference_time"],
                       generate_len=result["generate_len"], generated_ids=suffix[0].tolist(),
                       output=processor.batch_decode(suffix, skip_special_tokens=True)[0])
            if method != "ar":
                row["average_accept_length"] = result["mean_accept_length"]
            files[method].write(json.dumps(row, ensure_ascii=False) + "\n")
            files[method].flush()
            records_by_method[method].append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for method, records in records_by_method.items():
        directory = root / method
        seconds = sum(r["decoding_time"] for r in records)
        tokens = sum(r["generate_len"] for r in records)
        metric = dict(method=method, samples=len(records), avg_generate_len=tokens / len(records),
                      avg_decoding_time=seconds / len(records), avg_tokens_per_second=tokens / seconds if seconds else None)
        if method != "ar":
            metric["avg_accept_length"] = sum(r["average_accept_length"] for r in records) / len(records)
        (directory / "metric.jsonl").write_text(json.dumps(metric) + "\n")
        summaries.append(metric)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
