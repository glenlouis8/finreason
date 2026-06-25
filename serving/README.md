# FinReason Serving

Production serving layer for the fine-tuned FinReason model
(`glen-louis/finreason-qwen2.5-7b-dpo`): vLLM inference, Docker, Kubernetes,
autoscaling, load testing, and Prometheus/Grafana monitoring.

Training repo (SFT + DPO): separate — this repo only serves the published model.

## Status
- [x] Phase 0 — local kind cluster (`kubectl get nodes` Ready)
- [ ] Phase 4 — Deployment + Service on K8s (in progress, training-wheels nginx)
- [ ] Phase 1-3 — vLLM + AWQ + Docker (needs GPU; Colab/RunPod)
- [ ] Phase 5 — HPA autoscaling
- [ ] Phase 6 — load test + benchmarks
- [ ] Phase 7 — Prometheus + Grafana
- [ ] Phase 8 — online eval harness

## Cluster
```bash
kind create cluster --name finreason
kubectl get nodes            # -> Ready
```

## Deploy (current: nginx training-wheels to learn K8s mechanics)
```bash
kubectl apply -f k8s/                 # create Deployment + Service
kubectl get pods -w                   # watch pods come up (Ctrl-C to stop)
kubectl port-forward svc/finreason-web 8080:8080
curl localhost:8080                   # nginx welcome page through the Service
```
