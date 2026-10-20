"""Compare saved AR and accelerated results using aggregate decoding throughput."""
import argparse
import json
from pathlib import Path


def read_results(directory):
    path = Path(directory) / "results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No sample records in {path}")
    indices = [row["sample_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate sample indices in {path}")
    if any(row["decoding_time"] <= 0 or row["generate_len"] <= 0 for row in rows):
        raise ValueError(f"Non-positive time or token count in {path}")
    return rows


def summarize(rows):
    seconds = sum(row["decoding_time"] for row in rows)
    tokens = sum(row["generate_len"] for row in rows)
    return {
        "samples": len(rows),
        "total_generated_tokens": tokens,
        "total_decoding_seconds": seconds,
        "tokens_per_second": tokens / seconds,
        "avg_decoding_seconds": seconds / len(rows),
    }


def compare(baseline, accelerated):
    if {row["sample_index"] for row in baseline} != {row["sample_index"] for row in accelerated}:
        raise ValueError("Sample indices differ; run both methods on the same samples")
    if any(row["method"] != "ar" for row in baseline):
        raise ValueError("The full baseline must use method=ar")
    if any(row["method"] == "ar" for row in accelerated):
        raise ValueError("Accelerated records must use a speculative decoding method")
    full, fast = summarize(baseline), summarize(accelerated)
    fast_by_index = {row["sample_index"]: row for row in accelerated}
    per_sample = []
    for row in baseline:
        other = fast_by_index[row["sample_index"]]
        full_tps = row["generate_len"] / row["decoding_time"]
        fast_tps = other["generate_len"] / other["decoding_time"]
        per_sample.append({
            "sample_index": row["sample_index"],
            "full_tokens": row["generate_len"],
            "accelerated_tokens": other["generate_len"],
            "full_tokens_per_second": full_tps,
            "accelerated_tokens_per_second": fast_tps,
            "throughput_speedup": fast_tps / full_tps,
        })
    return {
        "scope": "decoding only; excludes prefill, video preprocessing and model loading",
        "full_ar": full,
        "accelerated": fast,
        "decoding_throughput_speedup": fast["tokens_per_second"] / full["tokens_per_second"],
        "decoding_time_ratio": full["total_decoding_seconds"] / fast["total_decoding_seconds"],
        "per_sample": per_sample,
        "note": "Uses repository generate_len accounting. Outputs may have different lengths; throughput speedup normalizes by generated tokens. Matching sample indices alone does not validate identical inputs or hardware.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="AR result directory")
    parser.add_argument("--accelerated", required=True, help="Accelerated result directory")
    parser.add_argument("--output", help="Optional comparison JSON path")
    args = parser.parse_args()
    report = compare(read_results(args.baseline), read_results(args.accelerated))
    report["result_directories"] = {
        "full_ar": str(Path(args.baseline).resolve()),
        "accelerated": str(Path(args.accelerated).resolve()),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)
