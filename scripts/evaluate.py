"""
Stage 4: 3-stage evaluation — base → SFT → DPO.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/eval.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from tqdm import tqdm

from src.model_utils import load_base_only, load_model_for_inference, generate
from src.data_utils import load_jsonl
from src.eval_utils import compute_accuracy, compute_perplexity, compute_win_rate, save_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    return parser.parse_args()


def run_predictions(model, tokenizer, examples: list[dict], max_new_tokens: int) -> list[str]:
    predictions = []
    for ex in tqdm(examples, desc="Generating"):
        messages = ex["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pred = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        predictions.append(pred)
    return predictions


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    test_examples = load_jsonl(cfg["eval"]["test_path"])
    max_new_tokens = cfg["model"]["max_new_tokens"]
    tolerance = cfg["eval"]["numeric_tolerance"]

    results = {}

    # Base model
    print("\n=== Evaluating base model ===")
    model, tokenizer = load_base_only(cfg["model"]["base"], cfg)
    base_preds = run_predictions(model, tokenizer, test_examples, max_new_tokens)
    base_perplexity = compute_perplexity(model, tokenizer, test_examples[:200])
    results["base_accuracy"] = compute_accuracy(test_examples, base_preds, tolerance)
    results["base_perplexity"] = base_perplexity
    print(f"Base accuracy: {results['base_accuracy']:.3f} | Perplexity: {base_perplexity:.2f}")
    del model

    # SFT model
    print("\n=== Evaluating SFT model ===")
    model, tokenizer = load_model_for_inference(cfg["model"]["base"], cfg["model"]["sft_checkpoint"], cfg)
    sft_preds = run_predictions(model, tokenizer, test_examples, max_new_tokens)
    sft_perplexity = compute_perplexity(model, tokenizer, test_examples[:200])
    results["sft_accuracy"] = compute_accuracy(test_examples, sft_preds, tolerance)
    results["sft_perplexity"] = sft_perplexity
    print(f"SFT accuracy: {results['sft_accuracy']:.3f} | Perplexity: {sft_perplexity:.2f}")
    del model

    # DPO model
    print("\n=== Evaluating DPO model ===")
    model, tokenizer = load_model_for_inference(cfg["model"]["base"], cfg["model"]["dpo_checkpoint"], cfg)
    dpo_preds = run_predictions(model, tokenizer, test_examples, max_new_tokens)
    dpo_perplexity = compute_perplexity(model, tokenizer, test_examples[:200])
    results["dpo_accuracy"] = compute_accuracy(test_examples, dpo_preds, tolerance)
    results["dpo_perplexity"] = dpo_perplexity
    results["dpo_win_rate"] = compute_win_rate(sft_preds, dpo_preds, test_examples, tolerance)
    print(f"DPO accuracy: {results['dpo_accuracy']:.3f} | Perplexity: {dpo_perplexity:.2f}")
    print(f"DPO win rate vs SFT: {results['dpo_win_rate']:.3f}")
    del model

    print("\n=== Final Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")

    save_results(results, cfg["eval"]["results_dir"])


if __name__ == "__main__":
    main()
