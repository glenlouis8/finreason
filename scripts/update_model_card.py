"""
Regenerate and upload the HuggingFace model card — text only, no weights.

push_to_hub.py loads the base model and adapter to push weights, so it needs a GPU
and local checkpoints. Fixing a stale card shouldn't require either.

    export HF_TOKEN=hf_...
    python scripts/update_model_card.py --repo glen-louis/finreason-qwen2.5-7b-dpo
    python scripts/update_model_card.py --repo ... --dry-run    # print, don't upload
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import os
import re

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--sft-config", default="configs/sft.yaml")
    p.add_argument("--dpo-config", default="configs/dpo.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the card, don't upload")
    return p.parse_args()


def load_card_builders():
    """Pull the pure functions out of push_to_hub.py without importing torch/peft.

    push_to_hub imports the training stack at module scope; this script is meant to
    run on a laptop, so the two text-only functions are exec'd in isolation rather
    than duplicated (duplicating them is how the card drifted from the configs).
    """
    src = Path(__file__).with_name("push_to_hub.py").read_text()
    ns = {"Path": Path, "json": json}
    for fn in ("find_latest_results", "build_model_card"):
        m = re.search(rf"^def {fn}\(.*?(?=^def |^if __name__)", src, re.S | re.M)
        if not m:
            raise RuntimeError(f"could not locate {fn}() in push_to_hub.py")
        exec(m.group(0), ns)
    return ns["find_latest_results"], ns["build_model_card"]


def main():
    args = parse_args()
    find_latest_results, build_model_card = load_card_builders()

    cfg = yaml.safe_load(open(args.config))
    sft_cfg = yaml.safe_load(open(args.sft_config))
    dpo_cfg = yaml.safe_load(open(args.dpo_config))

    results = find_latest_results(cfg["eval"]["results_dir"])
    card = build_model_card(args.repo, results, sft_cfg["model"]["name"], sft_cfg, dpo_cfg)

    if args.dry_run:
        print(card)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN (https://huggingface.co/settings/tokens, write scope)")

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        token=token,
        commit_message="update model card: official FinQA test-set results",
    )
    print(f"card updated -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
