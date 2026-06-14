"""
Stage 1: SFT fine-tuning on FinQA with QLoRA.

Usage:
    python scripts/train_sft.py
    python scripts/train_sft.py --config configs/sft.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yaml
import wandb
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

from src.model_utils import load_model_for_training, load_tokenizer
from src.data_utils import load_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft.yaml")
    return parser.parse_args()


def format_for_trainer(examples: list[dict], tokenizer) -> list[str]:
    texts = []
    for ex in examples:
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return texts


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    model, tokenizer = load_model_for_training(cfg)

    train_raw = load_jsonl(cfg["data"]["train_path"])
    eval_raw = load_jsonl(cfg["data"]["eval_path"])

    train_texts = format_for_trainer(train_raw, tokenizer)
    eval_texts = format_for_trainer(eval_raw, tokenizer)

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    t = cfg["training"]
    sft_config = SFTConfig(
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
        dataset_text_field="text",
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"SFT checkpoint saved to {t['output_dir']}")

    wandb.finish()


if __name__ == "__main__":
    main()
