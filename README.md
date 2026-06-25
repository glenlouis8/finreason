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
- **Serving:** HuggingFace Hub + production K8s layer (see [`serving/`](serving/README.md))

---

## Repo Layout

```
configs/    # YAML hyperparams
src/        # data_utils, model_utils, eval_utils
scripts/    # one entry point per pipeline stage
results/    # versioned eval JSON
notebooks/  # colab_sft.ipynb, colab_dpo.ipynb
serving/    # production serving: vLLM, AWQ, Docker, K8s, HPA, load test, Grafana
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

## Serving (Kubernetes)

Production serving layer in [`serving/`](serving/README.md): vLLM inference,
AWQ 4-bit quantization, Docker, Kubernetes (Deployment + Service + HPA
autoscaling), Locust load testing, Prometheus/Grafana.

Proof (`serving/docs/screenshots/`): a real LLM running on Kubernetes —
pod + Service + OpenAI-compatible API answering a FinReason question through
the cluster. Load test: 326 req/s, p99 14ms, 0% errors @ 100 users.
7B FinReason model AWQ-quantized and published:
[glen-louis/finreason-qwen2.5-7b-awq](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-awq).

---

## Production Deployment (AWS SageMaker)

Training jobs and endpoint deployment scripts are included for production use.

**Launch SFT training job on SageMaker:**
```bash
python scripts/train_sagemaker.py --stage sft \
    --bucket my-s3-bucket \
    --role arn:aws:iam::123456789:role/SageMakerRole
```

**Launch DPO training job:**
```bash
python scripts/train_sagemaker.py --stage dpo \
    --bucket my-s3-bucket \
    --role arn:aws:iam::123456789:role/SageMakerRole
```

**Deploy model to endpoint (`ml.g5.2xlarge`):**
```bash
python scripts/deploy_sagemaker.py \
    --role arn:aws:iam::123456789:role/SageMakerRole \
    --deploy
```

**Run inference against live endpoint:**
```bash
python scripts/deploy_sagemaker.py --endpoint finreason-dpo --predict
```

**Delete endpoint (stop billing):**
```bash
python scripts/deploy_sagemaker.py --endpoint finreason-dpo --delete
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
