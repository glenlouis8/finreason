"""
Guards the data contract and the DPO pair miner.

Every example must be {"messages": [system, user, assistant], "answer": str},
and the assistant turn must end in a parseable `Final Answer:` line — SFT
teaches that format, and eval depends on it.

Tests that need data/ are skipped when it's absent (it's gitignored), so this
suite still runs on a clean clone. Run `python scripts/prepare_data.py` first
for full coverage.
"""
from pathlib import Path

import pytest

from src.data_utils import (
    SYSTEM_PROMPT, build_preference_pairs, extract_numeric_answer, load_jsonl,
)

SPLITS = {
    "data/sft_train.jsonl": 6251,   # FinQA official counts — a mismatch means
    "data/sft_dev.jsonl": 883,      # the split logic regressed to a homemade
    "data/sft_test.jsonl": 1147,    # holdout, which breaks paper comparability
}


@pytest.mark.parametrize("path,expected_n", SPLITS.items())
def test_split_sizes_match_official_finqa(path, expected_n):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} not built — run scripts/prepare_data.py")
    assert len(load_jsonl(path)) == expected_n


@pytest.mark.parametrize("path", SPLITS)
def test_example_shape_and_parseable_answer(path):
    p = Path(path)
    if not p.exists():
        pytest.skip(f"{path} not built — run scripts/prepare_data.py")

    for ex in load_jsonl(path)[:50]:
        assert set(ex) == {"messages", "answer"}
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert ex["messages"][0]["content"] == SYSTEM_PROMPT
        assert "Final Answer:" in ex["messages"][2]["content"]
        assert extract_numeric_answer(ex["messages"][2]["content"]) is not None


def _example(answer: str = "100"):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": f"Step 1: ...\nFinal Answer: {answer}"},
        ],
        "answer": answer,
    }


def test_pair_miner_rejects_the_wrong_run_not_the_correct_one():
    """The regression that made DPO training a no-op.

    With a broken parser this still produced a pair, but `rejected` was the
    FIRST run — the correct one — so the preference signal was arbitrary.
    """
    outputs = [{"runs": [{"text": "Final Answer: 100"},      # correct
                         {"text": "Final Answer: 250"}]}]    # wrong
    pairs = build_preference_pairs([_example()], outputs)

    assert len(pairs) == 1
    assert "250" in pairs[0]["rejected"][0]["content"]
    assert "100" in pairs[0]["chosen"][0]["content"]


def test_pair_miner_emits_nothing_when_every_run_is_correct():
    """No wrong answer means no preference signal — must not fabricate one."""
    outputs = [{"runs": [{"text": "Final Answer: 100"},
                         {"text": "Final Answer: 100.5"}]}]  # within 1% tolerance
    assert build_preference_pairs([_example()], outputs) == []


def test_pair_record_shape():
    outputs = [{"runs": [{"text": "Final Answer: 999"}]}]
    pair = build_preference_pairs([_example()], outputs)[0]

    assert set(pair) == {"prompt", "chosen", "rejected"}
    assert [m["role"] for m in pair["prompt"]] == ["system", "user"]
