"""
Stage 5: Push final model + model card to HuggingFace Hub.

Usage:
    python scripts/push_to_hub.py --repo your-username/finreason-qwen2.5-7b-dpo
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import yaml
from pathlib import Path
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoTokenizer

from src.model_utils import load_base_only, get_bnb_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--sft_config", default="configs/sft.yaml")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/finreason-dpo")
    parser.add_argument("--results", default=None, help="Path to eval results JSON")
    return parser.parse_args()


def find_latest_results(results_dir: str) -> dict:
    results_path = sorted(Path(results_dir).glob("eval_*.json"))
    if not results_path:
        raise FileNotFoundError(f"No results found in {results_dir}. Run evaluate.py first.")
    return json.loads(results_path[-1].read_text())


def build_model_card(repo: str, results: dict, base_model: str) -> str:
    return f"""---
language: en
license: apache-2.0
base_model: {base_model}
tags:
  - finance
  - fine-tuned
  - qlora
  - dpo
  - finqa
datasets:
  - dreamerdeo/finqa
---

# FinReason — {repo.split('/')[-1]}

Financial reasoning LLM fine-tuned on [FinQA](https://github.com/czyssrs/FinQA) (SEC earnings reports).
SFT + DPO alignment pipeline on top of `{base_model}`.

## Results (3-stage eval on FinQA test set)

| Stage | Accuracy | Perplexity |
|-------|----------|------------|
| Base  | {results.get('base_accuracy', 0):.1%} | {results.get('base_perplexity', 0):.1f} |
| SFT   | {results.get('sft_accuracy', 0):.1%} | {results.get('sft_perplexity', 0):.1f} |
| DPO   | {results.get('dpo_accuracy', 0):.1%} | {results.get('dpo_perplexity', 0):.1f} |

**DPO win rate vs SFT:** {results.get('dpo_win_rate', 0):.1%}

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo}")
tokenizer = AutoTokenizer.from_pretrained("{repo}")

messages = [
    {{"role": "system", "content": "You are a financial analyst. Reason step-by-step and provide the final numeric answer."}},
    {{"role": "user", "content": "Table:\\nYear | Revenue\\n2019 | 12.4B\\n2018 | 11.2B\\n\\nQuestion: What was the % change in revenue?"}},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training

- **Dataset:** FinQA (~7.6k train, ~1k test)
- **SFT:** QLoRA NF4 4-bit, r=16 alpha=32, 3 epochs, lr=2e-4
- **DPO:** beta=0.1, 1 epoch, lr=5e-5
- **Preference pairs:** mined from SFT model outputs (correct vs wrong chain-of-thought)
- **Infrastructure:** AWS SageMaker ml.g5.2xlarge
- **Experiment tracking:** Weights & Biases
"""


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.sft_config) as f:
        sft_cfg = yaml.safe_load(f)

    results_path = args.results or None
    if results_path:
        results = json.loads(Path(results_path).read_text())
    else:
        results = find_latest_results(cfg["eval"]["results_dir"])

    print(f"Pushing to {args.repo}...")

    base_model_name = sft_cfg["model"]["name"]
    bnb_config = get_bnb_config(sft_cfg)
    base_model, _ = load_base_only(base_model_name, sft_cfg)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, cfg["model"]["dpo_checkpoint"])

    model.push_to_hub(args.repo)
    tokenizer.push_to_hub(args.repo)

    model_card = build_model_card(args.repo, results, base_model_name)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"Done. Model live at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
