# FinReason Serving — Concept Study Sheet

Plain-English notes on the 25 concepts for the LLM serving project. No prior infra knowledge assumed. Read top to bottom; each builds on the last. Analogies are deliberately simple.

---

## Serving the model

### 1. Inference server
A program that keeps your model loaded in memory and answers requests over HTTP. Instead of running a script every time, the model sits ready and replies when someone sends a question.
**Why it matters:** This is what "deploying a model" actually means — a running service, not a notebook.
**Analogy:** A restaurant kitchen that stays open, vs. cooking only when you personally get hungry.

### 2. vLLM
The specific inference server you'll use for LLMs. It's fast because of two tricks: *paged attention* (smart memory use) and *continuous batching* (serves many requests together instead of one-by-one). Exposes an OpenAI-style API, so you call it just like the OpenAI SDK.
**Why it matters:** It's the #1 LLM serving tool in industry. Naming it on a resume signals real serving experience.

### 3. Throughput
How much work the server does per second — measured in requests/sec or tokens/sec.
**Why it matters:** "Handles 200 req/s" is the kind of number recruiters want. Higher = serves more users per GPU = cheaper.

### 4. Latency (p50 / p99)
How long one request takes. p50 = the median (typical) request. p99 = the slowest 1% of requests.
**Why it matters:** Averages lie. p99 tells you the worst real users actually feel. Production teams obsess over p99.
**Analogy:** A line at the DMV — "average wait 5 min" hides the poor guy stuck for 40. p99 is that guy.

### 5. TTFT (Time To First Token)
For LLMs specifically: how long until the model starts streaming its answer. Separate from total time.
**Why it matters:** Users feel TTFT as "responsiveness." A chatbot that starts typing in 200ms feels fast even if the full answer takes 3s.

---

## Making it fit

### 6. Quantization
Shrinking the model's numbers from high precision (16-bit) to low precision (4-bit). Makes the model ~4x smaller and faster, with a small accuracy cost.
**Why it matters:** A 7B model needs a big expensive GPU at full size. Quantized, it fits on a cheap one. This is real "GPU cost optimization" — a high-value talking point.
**Analogy:** Compressing a photo to JPEG — much smaller file, barely noticeable quality drop.

### 7. AWQ
One specific quantization method (Activation-aware Weight Quantization). It's smart about *which* weights matter most, so accuracy stays high. GPTQ is a competing method.
**Why it matters:** You'll quantize FinReason with AWQ and prove (via your eval) accuracy barely moved.

---

## Packaging

### 8. Docker image
Your server + Python + all libraries + the model setup, frozen into one portable box. Runs the same on any machine.
**Why it matters:** "Works on my laptop" stops being a problem. K8s only runs containers, so this is the entry ticket.
**Analogy:** A shipping container — pack once, runs identically anywhere. (You already use this.)

### 9. Container registry
A storage place for Docker images so other machines (or your cluster) can pull them. Docker Hub and GHCR are common ones.
**Why it matters:** Your K8s cluster downloads the image from here to run it.
**Analogy:** App Store for your containers.

---

## Kubernetes (the big section)

### 10. Kubernetes (K8s)
A system that runs and manages your containers for you: starts them, restarts them if they crash, scales them up/down, and networks them together.
**Why it matters:** THE gap in your profile. It's how real companies run services. "I deployed on K8s" unlocks ML Engineer / MLOps roles.
**Analogy:** An air traffic controller for containers — you say what you want running, it makes it happen and keeps it alive.

### 11. Cluster / Node
A **cluster** is the whole K8s system. A **node** is one machine inside it (your laptop, or a cloud server). A cluster has one or more nodes.
**Why it matters:** Everything runs "on the cluster," spread across nodes.

### 12. Pod
The smallest thing K8s runs. A pod wraps your container (usually one container = one pod).
**Why it matters:** When people say "the model is running in a pod," they mean one live copy of your server.
**Analogy:** A single worker doing the job.

### 13. Deployment
A rule that says "always keep N copies (pods) of this container running." If one dies, K8s makes a new one.
**Why it matters:** This is how you get reliability and multiple copies. You write this as a YAML file.
**Analogy:** A manager told "always keep 3 workers on shift" — someone leaves, manager hires a replacement automatically.

### 14. Service
A stable network address that points to your pods, no matter how many there are or where they move.
**Why it matters:** Pods come and go (and get new IPs). The Service gives one fixed door to knock on.
**Analogy:** A company phone number that routes to whichever employee is free.

### 15. ConfigMap / Secret
Ways to feed settings into your container from outside. **ConfigMap** = plain config. **Secret** = sensitive stuff (API keys, your HF token).
**Why it matters:** You never bake passwords into the image. Keeps things secure and changeable.

### 16. Probes (readiness / liveness)
Health checks K8s runs on your pod. **Readiness** = "is it done loading and ready for traffic?" **Liveness** = "is it still alive or stuck?"
**Why it matters:** A 7B model takes time to load. The readiness probe stops K8s sending requests before the model is ready (avoids errors).
**Analogy:** A restaurant flipping the "OPEN" sign only once the kitchen's actually ready.

### 17. kubectl
The command-line tool to talk to your cluster. `kubectl get pods`, `kubectl logs`, `kubectl apply -f file.yaml`.
**Why it matters:** Your main hands-on tool. Most of your K8s work is kubectl commands.

### 18. kind / k3s
Tools to run a real K8s cluster locally on your Mac, inside Docker. **kind** = "Kubernetes IN Docker." **k3s** = a lightweight version.
**Why it matters:** Lets you learn and prove K8s skill for free, no cloud bill. Same kubectl, same YAML as the real thing.

---

## Scaling

### 19. HPA (Horizontal Pod Autoscaler)
A K8s feature that watches load and automatically adds pods when busy, removes them when quiet.
**Why it matters:** This is literally what "autoscaling" / "scales" means on a resume. You set min/max and a target (e.g. 70% CPU).
**Analogy:** A store that calls in extra cashiers when lines get long, sends them home when it's slow.

### 20. metrics-server
A small component that measures how much CPU/memory each pod uses and reports it. The HPA reads these numbers to decide when to scale.
**Why it matters:** Without it, the autoscaler is blind. Required for HPA to work.

---

## Proving it works

### 21. Load testing
Firing lots of fake traffic at your endpoint on purpose, to see how fast it is and where it breaks.
**Why it matters:** No load test = no real numbers = "I deployed a thing" with nothing to back it. The numbers fill your resume line.

### 22. Locust / k6
The tools that do load testing. You write a small script ("send this request, ramp from 1 to 100 users"), it reports req/s and latency.
**Why it matters:** Locust is Python (familiar to you). Produces the throughput + p99 numbers and a breaking point.

---

## Watching it

### 23. Prometheus
A tool that continuously collects metrics (latency, requests, GPU use) from your server and stores them over time. vLLM exposes these automatically at a `/metrics` URL.
**Why it matters:** Production = you can see what's happening right now and historically. This is "observability."
**Analogy:** The black box flight recorder for your service.

### 24. Grafana
A dashboard tool that graphs Prometheus data. You build panels: throughput, p99 latency, GPU utilization, queue depth.
**Why it matters:** A screenshot of a live Grafana dashboard is visual proof on your portfolio. Sells instantly.
**Analogy:** The car dashboard — speed, fuel, temperature, all at a glance.

---

## Your strength, applied

### 25. Online eval
Continuously scoring the LIVE endpoint's answers (not just a one-time offline test). Send known FinQA questions periodically, check the answers stay correct, graph accuracy over time.
**Why it matters:** "Evals are the biggest separator in 2026." You already do offline eval (RAGAS, FinReason accuracy). Doing it on a live served model is the next-level version, and it builds directly on your existing strength.
**Analogy:** Not just a pre-launch test drive, but a check-engine light that keeps watching after launch.

---

## How to use this sheet
- Concepts 1-9: easy, you'll get them in a day.
- Concepts 10-18 (Kubernetes): the real learning. Go slow here. This is the gap.
- Concepts 19-25: quick once K8s clicks.
- Don't memorize — build each phase from the plan, come back here when a word confuses you. Understanding sticks through doing, not reading.
