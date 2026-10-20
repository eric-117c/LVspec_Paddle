"""Protect paired timing and judge score accounting from misleading metrics."""

import pytest

from tools.evaluate_paddle_comparison import parse_review, rouge_l_f1, speed_report


def test_throughput_and_latency_are_distinct_for_different_output_lengths():
    def row(tokens, decoding, inference):
        return dict(sample_index=0, input_index=0, repeat=0, generate_len=tokens,
                    decoding_time=decoding, inference_time=inference, generated_ids=[1] * tokens)
    report = speed_report({"ar": [row(100, 10, 12)], "lvspec": [row(50, 2, 4)]})
    assert report["decoding_throughput_speedup"] == 2.5
    assert report["inference_wall_time_speedup"] == 3
    assert report["inference_throughput_speedup"] == 1.5
    with pytest.raises(ValueError, match="same paired"):
        speed_report({"ar": [row(100, 10, 12)], "lvspec": []})


def test_reference_overlap_and_judge_failures_are_explicit():
    assert rouge_l_f1("A man sails.", "A man sails.") == 1
    assert rouge_l_f1("A man sails.", "A boy walks.") == pytest.approx(1 / 3)
    assert rouge_l_f1("", "A man sails.") == 0
    result = parse_review('```json\n{"correctness": 4, "coverage": 3, "reason": "Missing event", "unverified_claims": []}\n```')
    assert result["correctness"] == 4
    with pytest.raises(ValueError):
        parse_review('{"correctness": 6, "coverage": 3, "reason": "", "unverified_claims": []}')
    with pytest.raises(ValueError):
        parse_review('{"correctness": true, "coverage": 3, "reason": "", "unverified_claims": []}')
