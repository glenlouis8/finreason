# FinReason — Financial QA Fine-Tuning with DPO Alignment

Fine-tuning Qwen2.5-7B-Instruct on FinQA (SEC filings) using QLoRA SFT + DPO alignment.
Portfolio project targeting ML Engineer roles.

**Model on HuggingFace:** [glen-louis/finreason-qwen2.5-7b-dpo](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-dpo)

---

## Results

| Stage | Accuracy | Perplexity |
|-------|----------|------------|
| Base (Qwen2.5-7B-Instruct) | 0.3% | 6.46 |
| SFT (QLoRA fine-tuned) | 56.5% | **1.71** |
| DPO (aligned) | **58.5%** | 1.72 |

- **Perplexity drop:** 6.46 → 1.71 (base → SFT)
- **Accuracy gain:** 0.3% → 58.5% (+58.2pp, base → DPO)
- **DPO win rate vs SFT:** 0.625 (DPO preferred on 62.5% of contested pairs)

> **Stale — rerun pending.** These numbers came from a run that held out 5% of FinQA train
> (313 examples) as its eval set; the official `dev`/`test` files were never used. The pipeline
> now trains on all 6,251 train rows, validates on dev (883), and reports on test (1,147).
> Table gets replaced once SFT is retrained and `evaluate.py` reruns on test.

Numeric tolerance ±1%, seed=42.

---

## Dataset

[FinQA](https://github.com/czyssrs/FinQA) — multi-step numerical reasoning over SEC earnings reports.
Questions require arithmetic over financial tables (revenue growth, margins, YoY changes).

Official 3-way split, used as published:

| Split | N | Use |
|-------|---|-----|
| train | 6,251 | SFT + DPO pair mining |
| dev | 883 | per-epoch validation, best-checkpoint selection |
| test | 1,147 | reported metrics only |

---

## Pipeline

```
prepare_data.py   → download FinQA, format for SFT, build DPO preference pairs
train_sft.py      → QLoRA SFT on Qwen2.5-7B-Instruct
train_dpo.py      → DPO alignment on SFT checkpoint
evaluate.py       → 3-stage eval: base → SFT → DPO
push_to_hub.py    → publish to HuggingFace with model card
```

---

## Stack

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **QLoRA:** NF4 4-bit, double quant, LoRA r=16 α=32, all 7 projections
- **Training:** TRL SFTTrainer + DPOTrainer, paged_adamw_8bit, cosine LR
- **Tracking:** Weights & Biases
- **Compute:** Google Colab Pro A100 (40GB)
- **Serving:** HuggingFace Hub

---

## Repo Layout

```
configs/    # YAML hyperparams
src/        # data_utils, model_utils, eval_utils
scripts/    # one entry point per pipeline stage
results/    # versioned eval JSON
notebooks/  # colab_sft.ipynb, colab_dpo.ipynb
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py

# SFT
python scripts/train_sft.py --config configs/sft.yaml

# DPO
python scripts/prepare_data.py --dpo --synthetic
python scripts/train_dpo.py --config configs/dpo.yaml

# Eval
python scripts/evaluate.py --config configs/eval.yaml
```

---

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "glen-louis/finreason-qwen2.5-7b-dpo")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

Or use the inference CLI:

```bash
python scripts/infer.py --question "What was the revenue growth from 2021 to 2022?"
```
