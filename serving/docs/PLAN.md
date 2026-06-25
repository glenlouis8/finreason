# FinReason Serving — Production LLM Inference on Kubernetes

**Goal:** Take the already-trained FinReason model (`glen-louis/finreason-qwen2.5-7b-dpo`) and wrap a real production serving layer around it: vLLM inference, Docker, Kubernetes, load testing, and monitoring.

**Why this project:** Closes the one real gap in Glen's profile — production/infra/serving/K8s. Every other axis (fine-tuning, agents, RAG, evals) is already strong. This project does NOT require retraining anything. The model is done. You only build the serving stack on top.

**The honest framing of "production at scale":** Right now FinReason just sits as weights on HuggingFace. "Production at scale" means: a live endpoint that handles many requests at once, fast, without crashing, with numbers to prove it (requests/sec, p99 latency, GPU utilization). That's the entire concept. Nothing more mystical than that.

**Learn-by-doing approach:** You don't know this stack yet. That's fine and expected. Each phase below teaches one concept, then has you build it immediately. Do not read ahead and try to learn everything first. Build phase 1, get it working, then move to phase 2. The doing IS the learning.

**Deployment target decision:** Local Kubernetes (kind or k3s) on your machine, model quantized (AWQ) to fit. This gives you the real Kubernetes skill at $0. Recruiters care that you can write manifests and explain autoscaling, not whether it ran on EKS. Optionally spin up one cloud GPU run at the very end for a single "served on cloud K8s" benchmark screenshot.

---

## Target resume line (what you're building toward)

> Served fine-tuned Qwen2.5-7B (FinReason) via vLLM on Kubernetes with horizontal autoscaling; achieved N tokens/sec throughput at p99 TTFT < X ms, AWQ-quantized for ~Y× memory reduction, with online eval harness and Prometheus/Grafana monitoring.

Every word in that line maps to a phase below. By the end you can defend all of it in an interview.

---

## Prerequisites / setup (Phase 0)

**Learn:** What each tool is, in one line.
- **vLLM** — fast LLM inference server (paged attention + continuous batching). The thing that actually runs the model.
- **Docker** — packages your server + dependencies into one runnable image. (You already know this.)
- **Kubernetes (K8s)** — runs and manages containers; handles scaling, restarts, networking. The gap you're closing.
- **kind / k3s** — run a real K8s cluster locally on your laptop. (kind = "Kubernetes in Docker".)
- **kubectl** — command-line tool to talk to a K8s cluster.
- **HPA (Horizontal Pod Autoscaler)** — K8s feature that adds/removes copies of your server based on load.
- **Locust / k6** — fire fake traffic at your endpoint to measure how it holds up.
- **Prometheus / Grafana** — collect metrics (latency, throughput, GPU) and graph them.

**Do:**
- [ ] Install Docker Desktop (have it), `kind`, `kubectl`, `helm`.
- [ ] Create new repo `finreason-serving` (separate from the training repo — keep training and serving clean).
- [ ] `kind create cluster` → confirm `kubectl get nodes` shows a node. That's a live cluster. Done.

**Definition of done:** `kubectl get nodes` returns a Ready node.

---

## Phase 1 — Serve the model with vLLM (no K8s yet)

**Learn:** How an LLM is actually served. vLLM exposes an OpenAI-compatible HTTP endpoint. Concepts: TTFT (time to first token), throughput (tokens/sec), KV cache, continuous batching.

**Do:**
- [ ] Run vLLM locally pointing at your HF model: `vllm serve glen-louis/finreason-qwen2.5-7b-dpo` (on a GPU box — Colab, RunPod, or local GPU).
- [ ] Hit it with a curl request, get a financial-reasoning answer back.
- [ ] Note the problem: 7B model needs ~16GB+ VRAM. This forces Phase 2.

**Definition of done:** One curl request to vLLM returns a correct FinReason answer.

**Resume contribution:** "Served fine-tuned Qwen2.5-7B via vLLM."

---

## Phase 2 — Quantize so it fits (AWQ)

**Learn:** Quantization shrinks model weights (16-bit → 4-bit) so it fits on smaller/cheaper GPUs with minimal accuracy loss. AWQ and GPTQ are the two common methods. This is a real "GPU optimization" talking point.

**Do:**
- [ ] Quantize FinReason to AWQ 4-bit (autoawq library), push quantized version to HF as `finreason-qwen2.5-7b-awq`.
- [ ] Serve the AWQ model with vLLM. Measure memory before/after.
- [ ] Re-run your existing FinReason eval (you have `evaluate.py`) on the quantized model. Record accuracy delta — proves quantization didn't break it. THIS reuses your eval culture strength.

**Definition of done:** Quantized model serves and eval accuracy is within ~1-2pp of the full model. You have the memory-reduction number.

**Resume contribution:** "AWQ-quantized for ~Y× memory reduction" + "eval-validated post-quantization."

---

## Phase 3 — Containerize (Docker)

**Learn:** A reproducible image that anyone can run. Multi-stage builds, slim images, GPU base images (`nvidia/cuda`).

**Do:**
- [ ] Write a `Dockerfile` that installs vLLM and launches the server on the AWQ model.
- [ ] Build, run the container locally, hit the endpoint. Same result as Phase 1 but now portable.
- [ ] Push image to a registry (Docker Hub or GHCR).

**Definition of done:** `docker run <image>` gives a working endpoint.

---

## Phase 4 — Deploy to Kubernetes (THE core gap)

**Learn:** This is the heart of the project. Concepts, each as a YAML file:
- **Deployment** — declares "run N copies of this container."
- **Service** — gives the pods a stable network address.
- **ConfigMap / Secret** — config and HF token, kept out of the image.
- **Resource requests/limits** — how much CPU/RAM/GPU each pod gets.
- **Readiness/liveness probes** — K8s checks if the model finished loading before sending traffic.

**Do:**
- [ ] Write `deployment.yaml`, `service.yaml`, `configmap.yaml`.
- [ ] `kubectl apply -f k8s/` → `kubectl get pods` shows it running.
- [ ] `kubectl port-forward` → hit the endpoint through K8s. (If no local GPU: use a CPU-small model like Qwen2.5-0.5B just to prove the K8s mechanics, then note the GPU version in README. The K8s skill is identical regardless of model size.)
- [ ] Practice: `kubectl logs`, `kubectl describe pod`, kill a pod and watch K8s restart it.

**Definition of done:** Endpoint reachable through a K8s Service; you can explain each YAML out loud.

**Resume contribution:** "Served ... on Kubernetes." This is the unlock for ML Engineer / MLOps roles (Cluster B).

---

## Phase 5 — Autoscaling (HPA)

**Learn:** HPA watches a metric (CPU, or a custom metric like requests-in-flight) and adds/removes pods automatically. This is the literal meaning of "scales."

**Do:**
- [ ] Install `metrics-server` in the cluster.
- [ ] Write `hpa.yaml`: min 1 pod, max N, target ~70% CPU (or custom metric).
- [ ] Generate load (Phase 6) and watch `kubectl get hpa` add pods live.

**Definition of done:** Under load, pod count goes up automatically; when load stops, it scales back down.

**Resume contribution:** "with horizontal autoscaling."

---

## Phase 6 — Load test + benchmark numbers

**Learn:** You can't claim "at scale" without numbers. Throughput (req/s, tokens/s), latency percentiles (p50/p95/p99), and where it breaks.

**Do:**
- [ ] Write a Locust (or k6) script that fires concurrent FinReason questions.
- [ ] Ramp users: 1 → 10 → 50 → 100. Record req/s and p99 latency at each level.
- [ ] Find the breaking point (where latency spikes or errors start). Knowing your ceiling is a strong interview answer.
- [ ] Make a small table/chart of the results for the README.

**Definition of done:** A results table: concurrency vs throughput vs p99 latency. Real numbers fill the X and Y in the resume line.

**Resume contribution:** "N tokens/sec throughput at p99 TTFT < X ms."

---

## Phase 7 — Monitoring (Prometheus + Grafana)

**Learn:** Production means you can SEE what's happening. vLLM exposes Prometheus metrics out of the box (latency, tokens, queue depth, GPU util).

**Do:**
- [ ] Install Prometheus + Grafana via Helm into the cluster.
- [ ] Point Prometheus at vLLM's `/metrics`.
- [ ] Build a Grafana dashboard: throughput, p99 latency, GPU utilization, queue depth.
- [ ] Screenshot it for the README — visual proof sells.

**Definition of done:** A live Grafana dashboard showing serving metrics.

**Resume contribution:** "with Prometheus/Grafana monitoring."

---

## Phase 8 — Online eval harness (your strength, applied to serving)

**Learn:** "Evals are the single biggest separator in 2026." You already do offline eval. The new skill is evaluating the LIVE served endpoint, not a notebook.

**Do:**
- [ ] Build a small harness that periodically sends FinQA golden questions to the live endpoint and scores numeric accuracy (you already have the scoring logic in `evaluate.py`).
- [ ] Log accuracy over time as a metric → Grafana panel. This catches "the served model is degrading."

**Definition of done:** Live endpoint accuracy tracked as a metric.

**Resume contribution:** "with online eval harness."

---

## Phase 9 — Polish + write-up

**Do:**
- [ ] README with architecture diagram (mermaid), the benchmark table, Grafana screenshot, and a clear "what I learned" section.
- [ ] Record a short demo (load test running + HPA scaling + Grafana moving).
- [ ] Write the LinkedIn post (lead with the infra learning curve — honest story sells: "I knew LLMs, didn't know K8s, here's what I built").
- [ ] Update master_data.json with this as a new project (p8/p9), and update the portfolio.

**Definition of done:** A repo someone can read in 3 minutes and understand you can serve LLMs in production.

---

## Optional Phase 10 — One real cloud GPU run

If you want the literal "served on cloud Kubernetes with GPU" line:
- [ ] Spin up a single GPU node on GKE/EKS (or cheaper: RunPod + k3s), deploy the same manifests, run one benchmark, screenshot, tear it down same day to control cost.
- [ ] Adds "deployed on cloud-managed Kubernetes" — but the local version already proves the skill, so this is purely optional polish.

---

## Sequencing reality check

- Phases 1-3 are the easy on-ramp (a day or two).
- Phase 4 is the hard, important one — budget the most time here, it's the actual gap.
- Phases 5-8 each add one resume clause; they're incremental once Phase 4 works.
- Don't skip the benchmark (Phase 6) — without numbers the whole thing is just "I deployed a thing."

**The mindset:** do it, then learn why it worked. Get each phase running before understanding every detail. The understanding solidifies on the second pass and in interview prep.
