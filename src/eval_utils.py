import json
import math
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data_utils import extract_numeric_answer


def is_correct(predicted: float | None, ground_truth: str, tolerance: float = 0.01) -> bool:
    if predicted is None:
        return False
    gt = extract_numeric_answer(f"Final Answer: {ground_truth}")
    if gt is None:
        return predicted == 0.0
    if abs(gt) < 1e-9:
        return abs(predicted) < tolerance
    return abs(predicted - gt) / abs(gt) < tolerance


def compute_accuracy(examples: list[dict], predictions: list[str], tolerance: float = 0.01) -> float:
    correct = 0
    for ex, pred_text in zip(examples, predictions):
        predicted = extract_numeric_answer(pred_text)
        if is_correct(predicted, str(ex["answer"]), tolerance):
            correct += 1
    return correct / len(examples) if examples else 0.0


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    examples: list[dict],
    max_length: int = 2048,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for ex in examples:
        messages = ex["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.shape[1]
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")


def compute_win_rate(
    sft_predictions: list[str],
    dpo_predictions: list[str],
    examples: list[dict],
    tolerance: float = 0.01,
) -> float:
    wins = 0
    contested = 0
    for ex, sft_pred, dpo_pred in zip(examples, sft_predictions, dpo_predictions):
        sft_correct = is_correct(extract_numeric_answer(sft_pred), str(ex["answer"]), tolerance)
        dpo_correct = is_correct(extract_numeric_answer(dpo_pred), str(ex["answer"]), tolerance)
        if sft_correct != dpo_correct:
            contested += 1
            if dpo_correct:
                wins += 1
    return wins / contested if contested > 0 else 0.5


def save_results(results: dict, results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(results_dir) / f"eval_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return str(path)
