# FinReason model: merge → AWQ quantize → serve (RunPod runbook)

One-time GPU "factory shift" to produce the AWQ model + the numbers.
Recommended: RunPod, 1× A40/A6000 (48GB), PyTorch+CUDA template. ~30 min, ~$0.20.

## 0. On the RunPod box
```bash
pip install "autoawq" vllm transformers peft accelerate openai huggingface_hub
huggingface-cli login        # paste HF token (write access)
```

## 1. Quantize + push  (Phase 2)
```bash
python quantize_awq.py
```
Produces `glen-louis/finreason-qwen2.5-7b-awq` on the Hub.
Record the on-disk size before (merged fp16 ~14GB) vs after (AWQ ~5GB) → the
**memory-reduction number**.

## 2. Serve  (Phase 1)
```bash
bash serve.sh                # vLLM OpenAI endpoint on :8000
```

## 3. Smoke test  (in a second shell)
```bash
python test_endpoint.py      # one FinQA question -> answer ending 'Final Answer:'
```
curl version:
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "glen-louis/finreason-qwen2.5-7b-awq",
  "messages": [{"role":"user","content":"Revenue rose from $4.2B to $5.7B. Percent growth?"}]
}'
```

## 4. Eval the quantized model  (Phase 2 proof)
Re-run the FinReason accuracy check (scoring logic lives in the training repo's
`evaluate.py`) against the AWQ model. Record accuracy delta vs the full model —
target: within ~1-2pp. This proves quantization didn't break it.

## 5. Capture for the portfolio (then tear the pod down)
- memory: ~14GB → ~5GB  (~3× reduction)
- eval: full XX.X% → AWQ XX.X%  (Δ within 1-2pp)
- vLLM throughput / TTFT from `/metrics`
- screen-record the endpoint answering + Grafana moving = the demo
- `runpodctl` / dashboard → **stop the pod** to end billing

## Docker (Phase 3)
`../Dockerfile` builds a vLLM serving image:
```bash
docker build -t glen-louis/finreason-vllm:latest ..
docker push  glen-louis/finreason-vllm:latest
```
Then deploy with `../k8s/vllm-deployment.yaml` on a GPU cluster.
