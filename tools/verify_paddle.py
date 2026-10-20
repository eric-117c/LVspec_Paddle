"""Compare native Paddle model outputs with export_torch_reference.py artifacts."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_dir")
    p.add_argument("--device", default="cpu")
    p.add_argument("--atol", type=float, default=3e-5)
    p.add_argument("--rtol", type=float, default=3e-4)
    p.add_argument("--models", nargs="+", choices=["qwen", "llava"], default=["qwen", "llava"])
    p.add_argument("--output", help="Save a machine-readable error report")
    args = p.parse_args()
    import numpy as np
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.set_flags({"FLAGS_cudnn_allow_tf32": False, "FLAGS_cublas_allow_tf32": False})
    from paddle_backend.checkpoint import load_model
    from paddle_backend.inputs import to_paddle
    reports = []
    failures = []
    for name in args.models:
        directory = Path(args.reference_dir) / name
        source = directory / "source.json"
        checkpoint = json.loads(source.read_text())["checkpoint"] if source.exists() else directory
        model = load_model(checkpoint, dtype="float32", devices=[args.device], attention_mode="eager")
        with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
            inputs = to_paddle(dict(arrays), args.device)
        with paddle.no_grad():
            output = model(**inputs)
        actual = dict(logits=output.logits.numpy(), hidden_states=output.hidden_states.numpy(),
                      input_embeddings=output.output_embeddings.numpy())
        if output.video_hidden_states is not None:
            actual["video_features"] = output.video_hidden_states.numpy()
        with np.load(directory / "reference.npz", allow_pickle=False) as expected:
            for key, value in actual.items():
                error = float(np.max(np.abs(value - expected[key])))
                passed = bool(np.allclose(value, expected[key], atol=args.atol, rtol=args.rtol))
                item = dict(model=name, tensor=key, max_absolute_error=error,
                            root_mean_square_error=float(np.sqrt(np.mean((value - expected[key])**2))),
                            passed=passed)
                if key == "logits":
                    item["argmax_agreement"] = float(np.mean(value.argmax(-1) == expected[key].argmax(-1)))
                reports.append(item)
                if not passed:
                    failures.append(f"{name}/{key}")
    report = json.dumps(reports, indent=2) + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).write_text(report)
    if failures:
        raise AssertionError(f"Outside atol={args.atol}, rtol={args.rtol}: {', '.join(failures)}")


if __name__ == "__main__":
    main()
