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
- [ ] Phase 1-3 — vLLM + AWQ + Docker (kit in `model/`; needs a GPU run — RunPod)
- [ ] Phase 8 — online eval harness

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

## Deploy (current: nginx training-wheels to learn K8s mechanics)
```bash
kubectl apply -f k8s/                 # create Deployment + Service
kubectl get pods -w                   # watch pods come up (Ctrl-C to stop)
kubectl port-forward svc/finreason-web 8080:8080
curl localhost:8080                   # nginx welcome page through the Service
```
