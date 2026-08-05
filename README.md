# FinReason — Financial QA Fine-Tuning with DPO Alignment

Fine-tuning Qwen2.5-7B-Instruct on FinQA (SEC filings) using QLoRA SFT + DPO alignment.
Portfolio project targeting ML Engineer roles.

**Model on HuggingFace:** [glen-louis/finreason-qwen2.5-7b-dpo](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-dpo)

---

## Results

Official FinQA test set, 1,147 examples. Numeric exact match, ±1% relative tolerance.

| Stage | Accuracy | Correct | Perplexity |
|-------|----------|---------|------------|
| Base (Qwen2.5-7B-Instruct, zero-shot) | 52.2% | 599/1147 | 6.60 |
| SFT (QLoRA fine-tuned) | 58.0% | 665/1147 | **2.87** |
| DPO (aligned) | **59.2%** | 679/1147 | 2.89 |

- **Perplexity drop:** 6.60 → 2.87 (base → SFT)
- **Accuracy gain:** +5.8pp from SFT, +1.2pp from DPO
- **DPO win rate vs SFT:** 0.630 — but on only 27 contested examples (17/27),
  so the DPO delta is **not statistically significant** (exact binomial p = 0.25,
  95% CI [0.42, 0.81]). DPO did not hurt; the evidence isn't strong enough to
  claim it helped.

Reference point: FinQANet (the FinQA paper's retriever-generator baseline) reports 61.24%
execution accuracy on the same test set.

Raw numbers in [`results/eval_20260805_210727.json`](results/eval_20260805_210727.json).

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
