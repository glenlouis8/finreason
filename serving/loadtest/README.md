# Load testing (Locust)

Fires concurrent traffic, reports throughput + latency percentiles, and
(bonus) triggers the HPA so you watch autoscaling under real load.

## Run

Terminal 1 — open a tunnel to the Service:
```bash
kubectl port-forward svc/finreason-web 8080:8080
```

Terminal 2 — install + launch Locust:
```bash
pip install locust          # one-time
locust -f loadtest/locustfile.py --host http://localhost:8080
```
Open http://localhost:8089 → set users (e.g. 100) + ramp (e.g. 10/s) → Start.

Terminal 3 (optional) — watch autoscaling react to the load:
```bash
kubectl get hpa -w
```

## What to record
Ramp 1 → 10 → 50 → 100 users. At each level note from Locust:
- **RPS** (throughput)
- **p50 / p95 / p99** latency (ms)
- where errors/latency spike = the **breaking point**

Put the table in the top-level README. Those numbers fill the resume line.
