"""
Guards metric semantics and the committed results — the regression tests that
catch a silently-degraded model or a silently-changed metric definition.

test_committed_results_are_sane() is the actual regression gate: if a future
eval run lands below the floor, or a stage ordering inverts, it fails.
"""
import json
from pathlib import Path

import pytest

from src.eval_utils import compute_accuracy, compute_win_rate, is_correct

RESULTS_DIR = Path("results")
# Floors, not exact values — reruns vary slightly. Set below the numbers in
# results/eval_20260805_210727.json (base .522 / sft .580 / dpo .592 on the
# official 1147-example FinQA test set).
ACCURACY_FLOOR = {"base": 0.45, "sft": 0.52, "dpo": 0.53}


def test_tolerance_is_relative_not_absolute():
    """1% of 4.2e9 is 42 million. Callers must not assume absolute slack."""
    assert is_correct(4_200_000_000.0, "4200000000", 0.01)
    assert is_correct(4_220_000_000.0, "4200000000", 0.01)      # +0.48%
    assert not is_correct(4_300_000_000.0, "4200000000", 0.01)  # +2.4%


def test_zero_ground_truth_uses_absolute_tolerance():
    """Relative error is undefined at 0 — must not divide by zero."""
    assert is_correct(0.0, "0", 0.01)
    assert not is_correct(5.0, "0", 0.01)


def test_accuracy_counts_unparseable_as_wrong():
    examples = [{"answer": "100"}, {"answer": "200"}]
    predictions = ["Final Answer: 100", "I cannot determine this"]
    assert compute_accuracy(examples, predictions, 0.01) == 0.5


def test_win_rate_ignores_agreements():
    """Win rate counts ONLY contested pairs.

    This is why the published 0.630 rests on 27 examples out of 1147 — the
    models agreed on the other 1120. Anyone reading the number must know the
    denominator is small.
    """
    examples = [{"answer": "100"}, {"answer": "200"}, {"answer": "300"}]
    sft = ["Final Answer: 100", "Final Answer: 999", "Final Answer: 999"]
    dpo = ["Final Answer: 100", "Final Answer: 200", "Final Answer: 888"]
    # ex0: both right (skip). ex1: dpo wins. ex2: both wrong (skip). -> 1/1
    assert compute_win_rate(sft, dpo, examples, 0.01) == 1.0


def test_win_rate_returns_half_when_nothing_contested():
    examples = [{"answer": "100"}]
    preds = ["Final Answer: 100"]
    assert compute_win_rate(preds, preds, examples, 0.01) == 0.5


def _latest_results():
    files = sorted(RESULTS_DIR.glob("eval_2026*.json"))
    return json.loads(files[-1].read_text()) if files else None


def test_committed_results_are_sane():
    """Regression gate on the newest committed eval run."""
    res = _latest_results()
    if res is None:
        pytest.skip("no results JSON committed")
    if "base_accuracy" not in res:
        pytest.skip("not a 3-stage eval file")

    for stage, floor in ACCURACY_FLOOR.items():
        acc = res[f"{stage}_accuracy"]
        assert 0.0 <= acc <= 1.0, f"{stage} accuracy out of range: {acc}"
        assert acc >= floor, f"{stage} accuracy {acc:.3f} regressed below {floor}"

    assert res["sft_accuracy"] > res["base_accuracy"], "SFT must beat base"
    assert res["sft_perplexity"] < res["base_perplexity"], "SFT must lower perplexity"


def test_base_accuracy_is_not_a_parser_artifact():
    """A near-zero base score means the parser broke, not that the model can't reason.

    Qwen2.5-7B-Instruct scores ~52% zero-shot on FinQA. If a future run reports
    base near 0, extract_numeric_answer's fallback chain has regressed.
    """
    res = _latest_results()
    if res is None or "base_accuracy" not in res:
        pytest.skip("no 3-stage results committed")
    assert res["base_accuracy"] > 0.10, (
        f"base accuracy {res['base_accuracy']:.4f} is implausibly low — "
        "check extract_numeric_answer's fallback path"
    )
