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
    "test":  "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def _download_finqa(split: str) -> list[dict]:
    url = FINQA_URLS[split]
    print(f"Downloading FinQA {split} from {url}...")
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())


def load_finqa_sft(eval_split_ratio: float = 0.05, seed: int = 42):
    import random

    train_raw = _download_finqa("train")
    test_raw = _download_finqa("test")

    train_formatted = [format_sft_example(ex) for ex in train_raw]
    test_formatted = [format_sft_example(ex) for ex in test_raw]

    random.seed(seed)
    random.shuffle(train_formatted)
    split = int(len(train_formatted) * (1 - eval_split_ratio))

    return train_formatted[:split], train_formatted[split:], test_formatted


def extract_numeric_answer(text: str) -> float | None:
    matches = re.findall(r"Final Answer:\s*([-+]?\d*\.?\d+%?)", text)
    if not matches:
        return None
    val = matches[-1].replace("%", "").replace(",", "").strip()
    try:
        return float(val)
    except ValueError:
        return None


def build_preference_pairs(
    examples: list[dict],
    model_outputs: list[dict],
) -> list[dict]:
    pairs = []
    for ex, output in zip(examples, model_outputs):
        correct_answer = extract_numeric_answer(str(ex.get("answer", ex.get("qa", {}).get("answer", ""))))
        chosen_text = None
        rejected_text = None

        for run in output["runs"]:
            predicted = extract_numeric_answer(run["text"])
            if correct_answer is not None and predicted is not None:
                is_correct = abs(predicted - correct_answer) / (abs(correct_answer) + 1e-9) < 0.01
                if is_correct and chosen_text is None:
                    chosen_text = run["text"]
                elif not is_correct and rejected_text is None:
                    rejected_text = run["text"]

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
