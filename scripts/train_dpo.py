"""
Stage 2: DPO alignment on SFT checkpoint.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --config configs/dpo.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import wandb
from datasets import Dataset
from trl import DPOTrainer, DPOConfig

from src.model_utils import load_model_for_inference, load_tokenizer, get_bnb_config
from src.data_utils import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--sft_config", default="configs/sft.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.sft_config) as f:
        sft_cfg = yaml.safe_load(f)

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    sft_checkpoint = cfg["training"]["sft_checkpoint"]
    model_name = sft_cfg["model"]["name"]

    print(f"Loading SFT model from {sft_checkpoint}...")
    model, tokenizer = load_model_for_inference(model_name, sft_checkpoint, sft_cfg)
    model.enable_input_require_grads()
    model.train()

    pairs_raw = load_jsonl(cfg["data"]["pairs_path"])

    dataset = Dataset.from_dict({
        "prompt":   [ex["prompt"] for ex in pairs_raw],
        "chosen":   [ex["chosen"] for ex in pairs_raw],
        "rejected": [ex["rejected"] for ex in pairs_raw],
    })

    t = cfg["training"]
    dpo_config = DPOConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        optim=t["optim"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        seed=t["seed"],
        beta=t["beta"],
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        report_to="wandb",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"DPO checkpoint saved to {t['output_dir']}")

    wandb.finish()


if __name__ == "__main__":
    main()
