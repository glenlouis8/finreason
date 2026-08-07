# FinReason

**Post-training pipeline for financial numerical reasoning.** QLoRA SFT → DPO alignment →
AWQ 4-bit compression → vLLM serving, measured end to end on FinQA's official test set.

[![tests](https://github.com/glenlouis8/finreason/actions/workflows/tests.yml/badge.svg)](https://github.com/glenlouis8/finreason/actions/workflows/tests.yml)

Given an SEC filing excerpt and a question like *"what is the net change in net revenue during 2015?"*,
the model reads the financial table, works through the arithmetic, and returns a number.

**Models:**
[`finreason-qwen2.5-7b-dpo`](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-dpo) (LoRA adapter) ·
[`finreason-qwen2.5-7b-awq`](https://huggingface.co/glen-louis/finreason-qwen2.5-7b-awq) (merged, AWQ 4-bit, vLLM-ready)

```
┌──────────┐   ┌─────┐   ┌─────┐   ┌──────────┐   ┌───────┐
│  FinQA   │──▶│ SFT │──▶│ DPO │──▶│ AWQ 4-bit│──▶│ vLLM  │
│ 6.2k ex  │   │QLoRA│   │ β0.1│   │  5.2 GB  │   │ serve │
└──────────┘   └─────┘   └─────┘   └──────────┘   └───────┘
                  │         │            │            │
               ppl 2.87  59.2% acc   55.1% acc    19 req/s
```

---

## Results

Official FinQA test set, 1,147 examples. Numeric exact match, ±1% relative tolerance.

| Stage | Accuracy | Correct | Perplexity |
|-------|----------|---------|------------|
| Base (Qwen2.5-7B-Instruct, zero-shot) | 52.2% | 599/1147 | 6.60 |
| SFT (QLoRA) | 58.0% | 665/1147 | **2.87** |
| **DPO (aligned)** | **59.2%** | 679/1147 | 2.89 |

Reference: **FinQANet**, the FinQA paper's retriever-generator baseline, reports **61.24%**
execution accuracy on this split.

- Perplexity 6.60 → 2.87 (base → SFT)
- Accuracy +5.8pp from SFT, +1.2pp from DPO
- DPO win rate 0.630 on contested examples — **but see the caveat below**

Raw JSON: [`results/eval_20260805_210727.json`](results/eval_20260805_210727.json)

### The DPO gain is not statistically significant

Win rate counts only examples where SFT and DPO disagree. There were **27** of them; DPO won 17.
Exact binomial **p = 0.25**, 95% CI **[0.42, 0.81]** — the interval contains 0.5.

DPO didn't hurt, and the direction is positive. But "DPO improved accuracy" is not a claim this
evidence supports, so the README doesn't make it.

---

## Dataset

[FinQA](https://github.com/czyssrs/FinQA) — multi-step numerical reasoning over S&P 500 earnings
reports. Questions require arithmetic across financial tables and surrounding prose.

Official 3-way split, used as published:

| Split | N | Use |
|-------|---|-----|
| train | 6,251 | SFT + DPO pair mining |
| dev | 883 | per-epoch validation, best-checkpoint selection |
| test | 1,147 | reported metrics only |

Earlier runs carved a homemade 5% holdout out of train (313 rows) and called it "test", which made
the numbers non-comparable to the paper. That's fixed; `dev` is FinQA's own validation file.

**Example**

```
Context:  ...entergy corporation and subsidiaries management's discussion...
Table:    Amount (In Millions) | 2014 net revenue | 5,735 | ... | 2015 net revenue | 5,829
Question: what is the net change in net revenue during 2015?

Target:   Step 1: 5829 - 5735 = 94
          Final Answer: 94
```

---

## Compression & serving

The DPO adapter merged into the base, AWQ-quantized to 4-bit (calibrated on 256 real FinQA
examples rather than a generic web corpus), and served with vLLM on a rented A40.

| | Weights | Accuracy | |
|---|---|---|---|
| NF4 4-bit + adapter | 15 GB base | **59.2%** | offline eval path |
| AWQ 4-bit merged | **5.2 GB** | 55.1% | the served model |

**AWQ cost 4.1pp** (679 → 631 of 1147). Verified it wasn't a decoding artifact: vLLM inherits
`repetition_penalty=1.05` from the model's `generation_config.json`, which `evaluate.py` never
applied. Re-ran with it disabled — 55.10% vs 55.01%, a one-example difference. The gap is
quantization and merge, not sampling.

Both figures are 4-bit, so this compares two quantization schemes; no fp16 baseline was measured.

### Load test

Locust against `/v1/chat/completions` with real FinQA prompts — full SEC tables, not a synthetic ping:

| Concurrency | req/s | p50 | p95 | p99 | Failures |
|---|---|---|---|---|---|
| 10 users | 3.95 | 480 ms | 1100 ms | 1500 ms | 0 / 650 |
| 50 users | **18.95** | 440 ms | 850 ms | 1500 ms | 0 / 3337 |

5× the load, 4.8× the throughput, and p50 *dropped*. That's vLLM's continuous batching absorbing
concurrency instead of queueing it. The server reported 37.2 GiB of KV cache (~645k tokens,
~157 concurrent 4k-token requests), so 50 users never came close to saturation.

An earlier version of this README claimed 326 req/s at p99 14 ms. That load test hit an **nginx
placeholder**, not the model. These numbers are the model actually generating tokens.

### Observability

`serving/monitoring/setup_monitoring.sh` stands up Prometheus + Grafana against a running vLLM
server in one command, with the data source and
[dashboard](serving/monitoring/grafana_dashboard.json) provisioned from files.

![Grafana dashboard](serving/docs/screenshots/grafana-vllm-dashboard.png)

At 40 concurrent requests on an A40:

| Metric | Value |
|---|---|
| Generation throughput | ~3,000 tok/s |
| Time to first token | ~60 ms |
| End-to-end latency | ~700 ms |
| Requests waiting | 0 — the scheduler never queued |
| KV cache utilisation | 0.3% of 645k tokens |
| Prefix cache hit rate | 87.3% |

Caveat on the last two: the load generator repeats one short prompt, so KV usage sits far below
what full SEC-table prompts demand, and the prefix hit rate is flattered by that repetition.

`capture_metrics.py --load N` scrapes the same metrics to JSON without the Grafana stack — useful
on a rented box, since the JSON outlives it.

---

## Tests

```bash
pytest        # 30 tests, ~2s, CPU only
```

Runs on every push ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)). No model loading —
GPU eval costs money and doesn't belong in PR CI.

| File | Guards |
|---|---|
| `test_parser.py` | the answer parser — number formats, fallbacks, ground-truth parsing |
| `test_data_contract.py` | split sizes (6251/883/1147), example shape, pair-miner correctness |
| `test_metrics.py` | tolerance semantics, contested-only win rate, accuracy floors |

`test_metrics.py` includes the gates that matter: the build fails if committed accuracy drops below
a floor, if SFT stops beating base, or if base accuracy collapses back toward zero — the signature
of a regressed parser.

ML regressions don't raise exceptions — they produce numbers that are quietly wrong. A broken
answer parser doesn't crash the pipeline; it just scores every prediction as incorrect and
reports a plausible-looking metric. That's what this suite exists to catch.

---

## Pipeline

```
scripts/prepare_data.py              download FinQA, format for SFT, mine DPO pairs
scripts/train_sft.py                 QLoRA SFT, dev-loss checkpoint selection
scripts/train_dpo.py                 DPO on the SFT checkpoint
scripts/evaluate.py                  3-stage offline eval: base → SFT → DPO
scripts/push_to_hub.py               publish adapter + model card
scripts/eval_endpoint.py             score a live vLLM endpoint on the test set
scripts/infer.py                     inference CLI (--interactive for a REPL)

serving/model/quantize_awq.py        merge adapter → AWQ 4-bit → Hub
serving/loadtest/locustfile.py       load test with real FinQA prompts
serving/monitoring/                  Prometheus + Grafana setup, metrics capture
```

Config-driven: hyperparameters live in `configs/*.yaml`, not in the scripts.

---

## Quickstart

```bash
uv sync                                    # or: pip install -r requirements.txt

python scripts/prepare_data.py             # CPU, ~1 min
python scripts/train_sft.py   --config configs/sft.yaml     # A100, ~3h
python scripts/prepare_data.py --dpo       # mine pairs from the SFT model
python scripts/train_dpo.py   --config configs/dpo.yaml
python scripts/evaluate.py    --config configs/eval.yaml
```

Training stages need an A100-class GPU — Colab notebooks are in `notebooks/`.
Only `prepare_data.py` is CPU-safe.

### Serve it

```bash
vllm serve glen-louis/finreason-qwen2.5-7b-awq --quantization awq --max-model-len 4096
python serving/model/test_endpoint.py      # smoke test
python scripts/eval_endpoint.py            # score all 1147 test examples (~1 min)
```

### Or just load the adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "glen-louis/finreason-qwen2.5-7b-dpo")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

---

## Stack

| | |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA — NF4 4-bit, double quant, r=16 α=32, all 7 projections |
| Training | TRL `SFTTrainer` + `DPOTrainer` (β=0.1), `paged_adamw_8bit`, cosine LR |
| Compression | AWQ 4-bit (GEMM, group 128), FinQA-calibrated |
| Serving | vLLM 0.26 — continuous batching, PagedAttention, OpenAI-compatible API |
| Observability | Prometheus + Grafana, provisioned from files |
| Tracking | Weights & Biases — per-epoch dev loss, best-checkpoint selection |
| Compute | Colab A100 (training) · RunPod A40 (serving) |

---

## Repo layout

```
configs/    YAML hyperparams — edit these, not the scripts
src/        data_utils (data contract + parser), model_utils, eval_utils (metrics)
scripts/    one entry point per pipeline stage
tests/      parser, data contract, metric regression gates
serving/    quantize, load test, monitoring
results/    versioned eval JSON
notebooks/  colab_sft.ipynb, colab_dpo.ipynb
```

---

## Known limitations

- **DPO's gain isn't statistically significant** — 27 contested examples, p = 0.25.
- **No fp16 baseline**, so the AWQ delta measures NF4 → AWQ, not quantization vs full precision.
- **±1% relative tolerance is generous on large figures** — 1% of a $4.2B answer is $42M. FinQA's
  own answers are inconsistently rounded, which motivates it, but it's slack.
- **Training targets carry FinQA DSL artifacts.** `format_sft_example` interpolates raw operator
  names, so gold chains read `5829 minus2-1 5735 = 94`. Doesn't affect scored accuracy (only the
  final number counts) but the model imitates the malformed syntax in its output.
- **Live-model regression testing is manual.** CI gates code, data, and committed metrics;
  re-scoring the served model needs a GPU and is run by hand.
- **No Kubernetes layer here.** An earlier version had one (kind/k3s, HPA, Ollama demo), recoverable
  at `b667f0c`. Removed deliberately: vLLM handles intra-node concurrency, and a single replica
  doesn't need an orchestrator.

---

## License

Apache 2.0, matching the Qwen2.5 base model.
