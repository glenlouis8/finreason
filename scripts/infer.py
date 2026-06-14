"""
Inference CLI — ask the fine-tuned model a financial question.

Usage:
    python scripts/infer.py --question "What was the % change in revenue?"
    python scripts/infer.py --question "..." --table "Year | Revenue\n2019 | 12.4B\n2018 | 11.2B"
    python scripts/infer.py --interactive
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml

from src.model_utils import load_model_for_inference, generate
from src.data_utils import SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--question", default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def build_prompt(tokenizer, question: str, table: str | None) -> str:
    user_content = f"Table:\n{table}\n\nQuestion: {question}" if table else f"Question: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint = args.checkpoint or cfg["model"]["dpo_checkpoint"]
    print(f"Loading model from {checkpoint}...")
    model, tokenizer = load_model_for_inference(cfg["model"]["base"], checkpoint, cfg)

    if args.interactive:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            question = input("Question: ").strip()
            if question.lower() == "quit":
                break
            table = input("Table (optional, press Enter to skip): ").strip() or None
            prompt = build_prompt(tokenizer, question, table)
            answer = generate(model, tokenizer, prompt, max_new_tokens=cfg["model"]["max_new_tokens"])
            print(f"\nAnswer:\n{answer}\n")
    else:
        if not args.question:
            raise ValueError("Provide --question or use --interactive")
        prompt = build_prompt(tokenizer, args.question, args.table)
        answer = generate(model, tokenizer, prompt, max_new_tokens=cfg["model"]["max_new_tokens"])
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
