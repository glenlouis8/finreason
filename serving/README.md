# FinReason Serving

Production serving layer for the fine-tuned FinReason model
(`glen-louis/finreason-qwen2.5-7b-dpo`): vLLM inference, Docker, Kubernetes,
autoscaling, load testing, and Prometheus/Grafana monitoring.

This is the `serving/` layer of the FinReason repo. The training (SFT + DPO)
code lives at the repo root; this folder only serves the published model.

## Status
- [x] Phase 0 — local kind cluster (`kubectl get nodes` Ready)
- [x] Phase 4 — Deployment + Service on K8s + self-healing (training-wheels nginx)
- [x] Phase 5 — HPA autoscaling (2→8 pods, 50% CPU, both directions)
- [x] Phase 6 — load test: 326 req/s, p99 14ms, 0% errors @ 100 users
- [x] Phase 7 — Prometheus + Grafana dashboard
- [x] Real LLM on K8s — Qwen2.5-0.5B via Ollama on the kind cluster (CPU, $0),
      OpenAI-compatible API, answered a FinReason question through the Service
      (see `docs/screenshots/`). Manifest: `k8s/ollama-deployment.yaml`.
- [x] Phase 1-3 — vLLM + AWQ: model quantized (AWQ 4-bit) and published to
      `glen-louis/finreason-qwen2.5-7b-awq`. GPU-node serving manifest +
      runbook ready (`k8s/vllm-deployment.yaml`, `k8s/RUNBOOK-k3s.md`).
- [ ] Phase 8 — online eval harness

## Two serving tracks (honest scope)
- **Runs on K8s today (this Mac, CPU, $0):** Qwen2.5-0.5B via Ollama —
  proves the full K8s mechanic end-to-end (pod, Service, OpenAI API, live answer).
- **Real 7B FinReason model (needs a GPU node):** AWQ-quantized 7B served by
  vLLM. Quantize step done + published to HF; serving manifest + runbook ready
  for any GPU box (k3s + NVIDIA device plugin). Not run on K8s yet (no local GPU).

## Run on a GPU box (RunPod)
```bash
git clone https://github.com/glenlouis8/finreason.git
cd finreason/serving/model
# then follow model/README.md  (merge → AWQ → serve → test → eval)
```

## Local K8s cluster (Mac, $0)
```bash
kind create cluster --name finreason
kubectl get nodes            # -> Ready
```

## Deploy a real LLM on the local cluster (Mac, CPU, $0)
```bash
kubectl apply -f k8s/ollama-deployment.yaml          # Ollama Deployment + Service
kubectl get pods -l app=finreason-llm -w             # wait for Running
POD=$(kubectl get pod -l app=finreason-llm -o jsonpath='{.items[0].metadata.name}')
kubectl exec "$POD" -- ollama pull qwen2.5:0.5b       # pull model into pod (~400MB)
kubectl port-forward svc/finreason-llm 11434:11434 &  # reach it through the Service
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model":"qwen2.5:0.5b",
  "messages":[{"role":"system","content":"You are a financial analyst. End with a line Final Answer:"},
              {"role":"user","content":"Revenue grew from $4.2B to $5.7B. Percent growth?"}]}'
# -> real model answer, ending "Final Answer:", served through Kubernetes
```

## Deploy (nginx training-wheels — learn K8s mechanics)
```bash
kubectl apply -f k8s/                 # create Deployment + Service
kubectl get pods -w                   # watch pods come up (Ctrl-C to stop)
kubectl port-forward svc/finreason-web 8080:8080
curl localhost:8080                   # nginx welcome page through the Service
```
