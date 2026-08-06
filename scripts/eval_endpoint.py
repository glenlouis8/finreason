"""
Evaluate a live vLLM endpoint on the FinQA test set.

Answers the question offline eval can't: what did AWQ 4-bit quantization
actually cost in accuracy? Same test set, same tolerance, same parser as
scripts/evaluate.py — so the number is directly comparable to the fp16
DPO accuracy in results/eval_*.json.

Fast because it goes through vLLM's continuous batching instead of
transformers' generate loop: ~1 minute for 1147 examples vs hours.

    pip install openai
    python scripts/eval_endpoint.py --model glen-louis/finreason-qwen2.5-7b-awq
    python scripts/eval_endpoint.py --concurrency 64 --limit 200   # quick check
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tqdm import tqdm

from src.data_utils import load_jsonl, extract_numeric_answer
from src.eval_utils import is_correct, save_results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="glen-louis/finreason-qwen2.5-7b-awq")
    p.add_argument("--test-path", default="data/sft_test.jsonl")
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--tolerance", type=float, default=0.01)
    # Qwen ships generation_config.json with repetition_penalty=1.05 (a chat
    # default). vLLM honours it; scripts/evaluate.py never did. Penalising
    # repeated tokens is wrong here — correct FinQA answers restate numbers from
    # the table ("... = 94\nFinal Answer: 94"). Default 1.0 = off, matching the
    # offline eval so the two numbers are comparable. Pass 1.05 to reproduce the
    # server default.
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=None, help="only first N examples")
    p.add_argument("--results-dir", default="results/")
    p.add_argument("--save-predictions", default="results/endpoint_predictions.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    examples = load_jsonl(args.test_path)
    if args.limit:
        examples = examples[: args.limit]
    print(f"Evaluating {args.model} on {len(examples)} examples "
          f"(concurrency={args.concurrency}, "
          f"repetition_penalty={args.repetition_penalty})")

    def run_one(ex: dict) -> str:
        # temperature=0 to match how evaluate.py generated (greedy). The model's
        # generation_config.json sets temperature=0.7, so NOT passing this would
        # sample and make the comparison to offline eval invalid.
        resp = client.chat.completions.create(
            model=args.model,
            messages=ex["messages"][:-1],
            temperature=0.0,
            max_tokens=args.max_tokens,
            # repetition_penalty isn't in the OpenAI schema — vLLM reads it from
            # extra_body. Sent explicitly so generation_config.json can't override.
            extra_body={"repetition_penalty": args.repetition_penalty},
        )
        return resp.choices[0].message.content or ""

    start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        predictions = list(tqdm(
            pool.map(run_one, examples), total=len(examples), desc="Generating"
        ))
    elapsed = time.time() - start

    correct = 0
    rows = []
    for ex, pred in zip(examples, predictions):
        parsed = extract_numeric_answer(pred)
        ok = is_correct(parsed, str(ex["answer"]), args.tolerance)
        correct += ok
        rows.append({"answer": ex["answer"], "parsed": parsed,
                     "correct": ok, "prediction": pred})

    accuracy = correct / len(examples) if examples else 0.0

    # Predictions are saved so the run can be re-scored later without paying for
    # GPU time again — the thing the earlier eval runs never did.
    pred_path = Path(args.save_predictions)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nAccuracy: {accuracy:.4f}  ({correct}/{len(examples)})")
    print(f"Elapsed: {elapsed:.1f}s  ({len(examples)/elapsed:.2f} req/s)")
    print(f"Predictions -> {pred_path}")

    save_results({
        "endpoint_model": args.model,
        "endpoint_accuracy": accuracy,
        "endpoint_correct": correct,
        "endpoint_n": len(examples),
        "endpoint_elapsed_s": elapsed,
        "endpoint_req_per_s": len(examples) / elapsed,
        "numeric_tolerance": args.tolerance,
        "repetition_penalty": args.repetition_penalty,
    }, args.results_dir)


if __name__ == "__main__":
    main()
