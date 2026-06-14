# FinReason — Financial QA Fine-Tuning with DPO Alignment

Fine-tuning Qwen2.5-7B-Instruct on FinQA (SEC filings) using QLoRA SFT + DPO alignment.
Portfolio project targeting ML Engineer roles.

**Model on HuggingFace:** [glen-louis/finreason-qwen2.5-7b-dpo](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-dpo)

---

## Results

| Stage | Accuracy | Perplexity |
|-------|----------|------------|
| Base (Qwen2.5-7B-Instruct) | 0.3% | 6.46 |
| SFT (QLoRA fine-tuned) | **56.5%** | **1.71** |
| DPO (aligned) | 56.5% | 1.71 |

- **Perplexity drop:** 6.46 → 1.71 (base → SFT)
- **Accuracy gain:** 0.3% → 56.5% (+56.2pp)
- **DPO win rate vs SFT:** 0.500 (neutral — SFT already learned correct reasoning patterns)

Eval on FinQA test set (313 examples, seed=42). Numeric tolerance ±1%.

---

## Dataset

[FinQA](https://github.com/czyssrs/FinQA) — multi-step numerical reasoning over SEC earnings reports.
~8k train / ~1k test. Questions require arithmetic over financial tables (revenue growth, margins, YoY changes).

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
