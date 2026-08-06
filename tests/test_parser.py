"""
Guards extract_numeric_answer — the single highest-risk function in the repo.

Two production bugs came from it, both silent:
  1. Strict `Final Answer:`-only matching scored the BASE model at 0.3%. That
     measured format compliance, not reasoning, and inflated the reported
     base->DPO gain from ~+7pp to a meaningless +58pp.
  2. The same strictness made ground-truth parsing return None in
     build_preference_pairs, so `is_correct` was permanently False and EVERY
     sampled run got labelled "rejected" regardless of correctness. DPO trained
     on pairs with no correctness signal.

Neither raised an exception. Both only showed up as suspicious metrics.
"""
import pytest

from src.data_utils import extract_numeric_answer


@pytest.mark.parametrize("text,expected", [
    ("Step 1: 5829 - 5735 = 94\nFinal Answer: 94", 94.0),
    ("Final Answer: 127.40", 127.40),
    ("Final Answer: -12", -12.0),
    ("Final Answer: 14.1%", 14.1),
    ("Final Answer: 1,234.5", 1234.5),          # regression: used to return 1.0
    ("Final Answer: $2,500", 2500.0),
])
def test_trained_format(text, expected):
    assert extract_numeric_answer(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("The revenue grew. The answer is 1,234.", 1234.0),
    ("Answer: $2,500", 2500.0),
    ("So 500 / 4 = 125", 125.0),                 # last-number fallback
])
def test_untrained_phrasing_still_parses(text, expected):
    """Base-model phrasing must score on its math, not its formatting."""
    assert extract_numeric_answer(text) == expected


def test_accounting_negatives():
    """Financial text writes negatives in parentheses."""
    assert extract_numeric_answer("net change was (1,234)") == -1234.0


def test_bare_ground_truth_parses():
    """ex["answer"] has no "Final Answer:" prefix.

    This is the exact case that broke DPO pair mining. If it returns None,
    build_preference_pairs silently stops checking correctness.
    """
    assert extract_numeric_answer("127.40") == 127.40
    assert extract_numeric_answer("94") == 94.0


def test_last_final_answer_wins():
    """Multi-step chains can mention the phrase more than once."""
    assert extract_numeric_answer("Final Answer: 10\nFinal Answer: 20") == 20.0


def test_no_number_returns_none():
    assert extract_numeric_answer("no numbers at all here") is None


def test_strict_mode_measures_format_compliance():
    """fallback=False is how you'd measure 'did the model learn the format'."""
    assert extract_numeric_answer("The answer is 1234", fallback=False) is None
    assert extract_numeric_answer("Final Answer: 42", fallback=False) == 42.0
