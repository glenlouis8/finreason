#!/usr/bin/env bash
# Serve the AWQ-quantized FinReason model with vLLM.
# OpenAI-compatible endpoint on :8000. Run on a GPU box.
set -euo pipefail

MODEL="glen-louis/finreason-qwen2.5-7b-awq"

vllm serve "$MODEL" \
  --quantization awq \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --port 8000
# endpoint -> http://localhost:8000/v1/chat/completions
# metrics  -> http://localhost:8000/metrics   (Prometheus scrapes this later)
