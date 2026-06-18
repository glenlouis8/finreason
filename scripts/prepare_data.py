"""
Stage 0: Download FinQA and prepare SFT + DPO data files.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --dpo  # also generate DPO pairs (needs SFT checkpoint)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
from pathlib import Path

from src.data_utils import (
    load_finqa_sft, save_jsonl, load_jsonl, SYSTEM_PROMPT,
    build_preference_pairs, build_synthetic_preference_pairs,
)
from src.model_utils import load_model_for_inference, generate_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--dpo", action="store_true", help="Generate DPO preference pairs")
    parser.add_argument("--dpo_config", default="configs/dpo.yaml")
    parser.add_argument("--n_runs", type=int, default=3, help="Runs per question for DPO pair mining")
    parser.add_argument("--max_examples", type=int, default=2000, help="Max train examples to use for DPO pairs")
    parser.add_argument("--batch_size", type=int, default=8, help="Generation batch size")
    parser.add_argument("--synthetic", action="store_true", help="Build DPO pairs from ground truth (no model inference)")
    return parser.parse_args()


def prepare_sft(cfg: dict):
    print("Loading and formatting FinQA...")
    train, eval_, test = load_finqa_sft(
        eval_split_ratio=0.05,
        seed=cfg["training"]["seed"],
    )

    save_jsonl(train, "data/sft_train.jsonl")
    save_jsonl(eval_, "data/sft_eval.jsonl")
    save_jsonl(test, "data/sft_test.jsonl")

    print(f"SFT data saved — train: {len(train)}, eval: {len(eval_)}, test: {len(test)}")


def prepare_dpo_synthetic(cfg: dict, max_examples: int = 2000):
    train_examples = load_jsonl("data/sft_train.jsonl")[:max_examples]
    print(f"Building synthetic DPO pairs from {len(train_examples)} examples...")
    pairs = build_synthetic_preference_pairs(train_examples, seed=cfg["training"]["seed"])
    save_jsonl(pairs, "data/dpo_pairs.jsonl")
    print(f"Synthetic DPO pairs saved — {len(pairs)} pairs")


def prepare_dpo(cfg: dict, dpo_cfg: dict, n_runs: int, max_examples: int = 2000, batch_size: int = 8, checkpoint_every: int = 500):
    import json

    sft_checkpoint = dpo_cfg["training"]["sft_checkpoint"]
    if not Path(sft_checkpoint).exists():
        raise FileNotFoundError(f"SFT checkpoint not found at {sft_checkpoint}. Run train_sft.py first.")

    pairs_path = Path("data/dpo_pairs.jsonl")
    progress_path = Path("data/dpo_progress.json")

    start_idx = 0
    existing_pairs = []
    if progress_path.exists() and pairs_path.exists():
        progress = json.loads(progress_path.read_text())
        start_idx = progress["completed"]
        existing_pairs = load_jsonl(str(pairs_path))
        print(f"Resuming from example {start_idx} ({len(existing_pairs)} pairs so far)")

    print(f"Loading SFT model from {sft_checkpoint}...")
    model, tokenizer = load_model_for_inference(
        cfg["model"]["name"], sft_checkpoint, cfg
    )

    train_examples = load_jsonl("data/sft_train.jsonl")[:max_examples]
    remaining = train_examples[start_idx:]
    print(f"Generating {n_runs} runs per question for {len(remaining)} remaining examples (batch_size={batch_size})...")

    batch_outputs = []
    batch_examples = []

    for i, ex in enumerate(remaining):
        global_i = start_idx + i
        if i % 100 == 0:
            print(f"  {global_i}/{len(train_examples)}")

        messages = ex["messages"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        prompts_for_runs = [prompt] * n_runs
        texts = []
        for b in range(0, n_runs, batch_size):
            batch = prompts_for_runs[b:b + batch_size]
            texts.extend(generate_batch(model, tokenizer, batch, max_new_tokens=256, do_sample=True, temperature=1.1))

        batch_outputs.append({"runs": [{"text": t} for t in texts]})
        batch_examples.append(ex)

        if (i + 1) % checkpoint_every == 0:
            new_pairs = build_preference_pairs(batch_examples, batch_outputs)
            existing_pairs.extend(new_pairs)
            save_jsonl(existing_pairs, str(pairs_path))
            progress_path.write_text(json.dumps({"completed": global_i + 1}))
            print(f"  Checkpoint saved — {global_i + 1} done, {len(existing_pairs)} pairs total")
            batch_outputs = []
            batch_examples = []

    if batch_examples:
        new_pairs = build_preference_pairs(batch_examples, batch_outputs)
        existing_pairs.extend(new_pairs)

    save_jsonl(existing_pairs, str(pairs_path))
    progress_path.write_text(json.dumps({"completed": len(train_examples)}))
    print(f"DPO pairs saved — {len(existing_pairs)} pairs from {len(train_examples)} examples")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    Path("data").mkdir(exist_ok=True)

    prepare_sft(cfg)

    if args.dpo:
        if args.synthetic:
            prepare_dpo_synthetic(cfg, args.max_examples)
        else:
            with open(args.dpo_config) as f:
                dpo_cfg = yaml.safe_load(f)
            prepare_dpo(cfg, dpo_cfg, args.n_runs, args.max_examples, args.batch_size)


if __name__ == "__main__":
    main()
