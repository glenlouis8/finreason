# Context for New Project Session

## Who I Am
- Marian Glen Louis, MS Data Science (UB, Dec 2025), Buffalo NY
- Target roles: ML Engineer, AI Engineer, Data Scientist
- On OPT (STEM ~36 months from Dec 2025)
- GitHub: github.com/glenlouis8

## Why This Project
Current portfolio has SFT (p8: LLaMA 3.2 3B QLoRA on SQL, 94.6% perplexity drop).
Missing: post-training alignment (DPO/RLHF) + FinTech domain.
This project fills both gaps in one shot.
Market data confirms DPO/RLHF = most in-demand LLM skill 2026.

## What to Build
**Financial Reasoning LLM: SFT + DPO pipeline**

Full post-training pipeline on financial QA:
1. SFT fine-tune on FinQA dataset (SEC filings + numerical reasoning)
2. Generate preference pairs (correct vs. hallucinated/wrong calculations)
3. DPO training on top of SFT checkpoint
4. Eval: base → SFT → DPO comparison on answer accuracy

## Stack (match existing p8 stack where possible)
- Base model: LLaMA 3.2 3B (same as p8 for consistency) or Mistral 7B
- QLoRA (4-bit NF4, same config pattern as p8)
- TRL library for DPO trainer
- Hugging Face PEFT
- **AWS SageMaker** Training Jobs for fine-tuning compute (better resume signal than Modal)
- **AWS SageMaker** Endpoints for model serving
- Weights & Biases for experiment tracking
- Eval: exact match + numeric tolerance on FinQA held-out test set

## Dataset
- FinQA: https://github.com/czyssrs/FinQA
- SEC earnings report documents + multi-step numerical reasoning Q&A pairs
- ~8k train examples, ~1k test

## Key Metrics
- SFT: perplexity drop (base → SFT)
- DPO: win rate vs SFT on preference pairs
- Answer accuracy: exact match % on numeric answers, base → SFT → DPO comparison
- Show 3-stage improvement curve

## Repo Structure (suggested)
```
finreason/
  configs/
    sft_config.yaml
    dpo_config.yaml
  src/
    data_utils.py       # FinQA loading + preference pair generation
    model_utils.py      # QLoRA setup, model loading
    eval_utils.py       # exact match, numeric tolerance eval
  scripts/
    prepare_data.py     # download + process FinQA, generate preference pairs
    train_sft.py        # SFT with QLoRA
    train_dpo.py        # DPO on top of SFT checkpoint
    evaluate.py         # base vs SFT vs DPO eval harness
    infer.py            # inference CLI
    push_to_hub.py      # publish with real metrics in model card
  results/
    base_eval.json
    sft_eval.json
    dpo_eval.json
  notebooks/
    colab_sft.ipynb
    colab_dpo.ipynb
  README.md
```

## Reference: p8 Patterns to Reuse
- QLoRA config: NF4 4-bit, double quant, LoRA r=16 a=32 across all 7 projections
- paged_adamw_8bit optimizer, cosine LR, warmup
- Locked seed eval split (seed=42)
- Results committed as versioned JSON artifacts
- Automated Hub publishing with real metrics injected into model card
- SageMaker Training Jobs + Endpoints for serving (replaces Modal + vLLM)

## What NOT to do
- Don't merge with p8 repo — separate project, separate resume entry
- Don't skip the DPO stage — that's the whole point
- Don't use ROUGE-L as primary eval — use answer accuracy (exact match on numeric answers)
- Don't use synthetic preference pairs only — derive from model outputs (correct vs. wrong chain-of-thought)

## Resume Entry (when done)
Project name: "FinReason: Financial QA Fine-Tune with DPO Alignment"
Key highlights to write:
- SFT + DPO pipeline on FinQA (SEC filings)
- Perplexity drop metric
- DPO win rate vs SFT baseline
- 3-stage eval curve (base → SFT → DPO)
- AWS SageMaker Training Jobs + Endpoints
- W&B experiment tracking
