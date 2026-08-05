import json
import re
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a financial analyst. Given a table from a SEC filing and a question, "
    "reason step-by-step and provide the final numeric answer."
)


def format_table(table: list[list[str]]) -> str:
    if not table:
        return ""
    rows = [" | ".join(str(cell) for cell in row) for row in table]
    return "\n".join(rows)


def format_sft_example(example: dict) -> dict:
    qa = example["qa"]
    question = qa["question"]
    answer = str(qa["answer"])

    table_str = format_table(example.get("table_ori", []))
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))
    context = f"{pre_text}\n\n{post_text}".strip()

    steps = qa.get("steps", [])
    if steps:
        reasoning = "\n".join(
            f"Step {i+1}: {s.get('arg1', '')} {s.get('op', '')} {s.get('arg2', '')} = {s.get('res', '')}"
            for i, s in enumerate(steps)
        )
        full_answer = f"{reasoning}\nFinal Answer: {answer}"
    else:
        full_answer = f"Final Answer: {answer}"

    user_content = f"Context:\n{context}\n\nTable:\n{table_str}\n\nQuestion: {question}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": full_answer},
        ],
        "answer": answer,
    }


FINQA_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev":   "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test":  "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def _download_finqa(split: str) -> list[dict]:
    url = FINQA_URLS[split]
    print(f"Downloading FinQA {split} from {url}...")
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())


def load_finqa_sft(seed: int = 42):
    """FinQA's own 3-way split: train 6251 / dev 883 / test 1147.

    Train is shuffled (seed-locked) so batch order is stable across runs; dev and
    test keep their published order. No homemade holdout — dev is the official
    validation file, so test numbers stay comparable to the FinQA paper.
    """
    import random

    train_formatted = [format_sft_example(ex) for ex in _download_finqa("train")]
    dev_formatted = [format_sft_example(ex) for ex in _download_finqa("dev")]
    test_formatted = [format_sft_example(ex) for ex in _download_finqa("test")]

    random.seed(seed)
    random.shuffle(train_formatted)

    return train_formatted, dev_formatted, test_formatted


# Matches 1234 / 1,234.5 / -12 / $1,234 / (1,234) / 14.1% — financial text is messy.
_NUM = r"\(?\s*[-+]?\s*\$?\s*\d[\d,]*\.?\d*\s*\)?\s*%?"


def _to_float(raw: str) -> float | None:
    s = raw.strip()
    negative = s.startswith("(") and s.endswith(")")   # accounting notation for negatives
    s = s.strip("()").replace("$", "").replace("%", "").replace(",", "").replace(" ", "")
    try:
        val = float(s)
    except ValueError:
        return None
    return -val if negative else val


def extract_numeric_answer(text: str, fallback: bool = True) -> float | None:
    """Pull the predicted number out of a model response.

    Tries the trained format first, then degrades. The fallback exists so the BASE
    model gets a fair score: it was never told to emit a "Final Answer:" line, so a
    strict-only parser scores it near 0% for formatting, not for reasoning — which
    inflates the base->SFT gain into a meaningless number.

    Pass fallback=False when you specifically want to measure format compliance.
    """
    matches = re.findall(rf"Final Answer:\s*({_NUM})", text)
    if matches:
        return _to_float(matches[-1])

    if not fallback:
        return None

    # "the answer is 1,234" / "answer: 1234" — common base-model phrasing
    matches = re.findall(rf"answers?\s*(?:is|:|=)\s*({_NUM})", text, flags=re.IGNORECASE)
    if matches:
        return _to_float(matches[-1])

    # Last resort: last number in the response. Reasoning chains end on their result.
    matches = re.findall(_NUM, text)
    if matches:
        return _to_float(matches[-1])

    return None


def build_preference_pairs(
    examples: list[dict],
    model_outputs: list[dict],
) -> list[dict]:
    # chosen = ground-truth reasoning (always good); rejected = any NOT-correct
    # sampled run. Numerically-wrong and malformed runs both count as rejected —
    # either is worse than the ground-truth chain, which is all DPO needs.
    #
    # NOTE: ex["answer"] is a bare string ("127.40"), no "Final Answer:" prefix.
    # extract_numeric_answer's fallback path is what parses it. Without the fallback
    # this returned None, is_correct was permanently False, and every run got
    # rejected regardless of correctness — the pairs carried no signal.
    pairs = []
    for ex, output in zip(examples, model_outputs):
        correct_answer = extract_numeric_answer(str(ex.get("answer", ex.get("qa", {}).get("answer", ""))))
        chosen_text = ex["messages"][2]["content"]
        rejected_text = None

        for run in output["runs"]:
            run_text = run["text"].strip()
            if not run_text or run_text == chosen_text.strip():
                continue
            predicted = extract_numeric_answer(run_text)
            is_correct = (
                correct_answer is not None
                and predicted is not None
                and abs(predicted - correct_answer) / (abs(correct_answer) + 1e-9) < 0.01
            )
            if not is_correct:
                rejected_text = run["text"]
                break

        if chosen_text and rejected_text:
            user_content = ex["messages"][1]["content"]
            pairs.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "chosen": [{"role": "assistant", "content": chosen_text}],
                "rejected": [{"role": "assistant", "content": rejected_text}],
            })

    return pairs


def build_synthetic_preference_pairs(examples: list[dict], seed: int = 42) -> list[dict]:
    import random
    rng = random.Random(seed)

    corrupt_factors = [1.5, 2.0, -1.0, 0.5, 3.0, 0.1, 10.0]

    pairs = []
    for ex in examples:
        raw = str(ex.get("answer", "")).replace("%", "").replace(",", "").strip()
        try:
            correct_answer = float(raw)
        except ValueError:
            continue

        chosen_text = ex["messages"][2]["content"]

        factor = rng.choice(corrupt_factors)
        wrong_answer = correct_answer * factor
        if abs(wrong_answer - correct_answer) < 1e-6:
            wrong_answer = correct_answer + 100.0

        if abs(wrong_answer) < 1e-9:
            wrong_answer = 999.0

        wrong_str = f"{wrong_answer:.4f}".rstrip("0").rstrip(".")
        rejected_text = re.sub(
            r"(Final Answer:\s*)[-+]?[\d,]*\.?\d+%?",
            rf"\g<1>{wrong_str}",
            chosen_text,
        )

        if rejected_text == chosen_text:
            continue

        user_content = ex["messages"][1]["content"]
        pairs.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
        })

    return pairs


def save_jsonl(data: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]
