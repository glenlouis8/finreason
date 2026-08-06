"""
Snapshot vLLM's Prometheus metrics from the live endpoint.

vLLM exposes /metrics in Prometheus text format. Running a full Prometheus +
Grafana stack on a rented pod for one hour is not worth the setup time, so this
scrapes the same numbers directly and writes them to JSON — the metrics survive
the pod's death, which a Grafana instance on that pod would not.

Run it DURING a load test to capture the interesting values (queue depth, KV
cache utilisation, throughput) rather than an idle snapshot.

    python serving/monitoring/capture_metrics.py --interval 5 --duration 120
    python serving/monitoring/capture_metrics.py --once
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The metrics worth keeping. Everything else vLLM emits is either a constant or
# a per-request histogram bucket that isn't readable without a Grafana query.
KEEP_PREFIXES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",          # >0 means requests are queueing
    "vllm:gpu_cache_usage_perc",          # KV cache pressure
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_success_total",
    "vllm:time_to_first_token_seconds_sum",
    "vllm:time_to_first_token_seconds_count",
    "vllm:time_per_output_token_seconds_sum",
    "vllm:time_per_output_token_seconds_count",
    "vllm:e2e_request_latency_seconds_sum",
    "vllm:e2e_request_latency_seconds_count",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000/metrics")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between scrapes")
    p.add_argument("--duration", type=float, default=120.0, help="total seconds")
    p.add_argument("--once", action="store_true", help="single snapshot, then exit")
    p.add_argument("--out", default="results/vllm_metrics.json")
    return p.parse_args()


def scrape(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        text = r.read().decode()

    sample = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(KEEP_PREFIXES):
            continue
        name, _, value = line.rpartition(" ")
        try:
            sample[name.strip()] = float(value)
        except ValueError:
            continue
    return sample


def derive(sample: dict) -> dict:
    """Turn Prometheus counter pairs into the averages a human wants to read."""
    out = {}

    def ratio(sum_key, count_key, label):
        s = next((v for k, v in sample.items() if k.startswith(sum_key)), None)
        c = next((v for k, v in sample.items() if k.startswith(count_key)), None)
        if s is not None and c:
            out[label] = round(s / c, 4)

    ratio("vllm:time_to_first_token_seconds_sum",
          "vllm:time_to_first_token_seconds_count", "avg_ttft_s")
    ratio("vllm:time_per_output_token_seconds_sum",
          "vllm:time_per_output_token_seconds_count", "avg_time_per_output_token_s")
    ratio("vllm:e2e_request_latency_seconds_sum",
          "vllm:e2e_request_latency_seconds_count", "avg_e2e_latency_s")

    if out.get("avg_time_per_output_token_s"):
        out["tokens_per_s_per_request"] = round(1 / out["avg_time_per_output_token_s"], 2)
    return out


def main():
    args = parse_args()
    samples = []
    deadline = time.time() + (0 if args.once else args.duration)

    while True:
        now = datetime.now(timezone.utc).isoformat()
        try:
            raw = scrape(args.url)
        except Exception as e:                       # pod died / server restarting
            print(f"[{now}] scrape failed: {e}")
            raw = {}

        if raw:
            snap = {"timestamp": now, "raw": raw, "derived": derive(raw)}
            samples.append(snap)
            running = next((v for k, v in raw.items()
                            if k.startswith("vllm:num_requests_running")), 0)
            waiting = next((v for k, v in raw.items()
                            if k.startswith("vllm:num_requests_waiting")), 0)
            cache = next((v for k, v in raw.items()
                          if k.startswith("vllm:gpu_cache_usage_perc")), 0)
            print(f"[{now}] running={running:.0f} waiting={waiting:.0f} "
                  f"kv_cache={cache:.1%} {snap['derived']}")

        if args.once or time.time() >= deadline:
            break
        time.sleep(args.interval)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, indent=2))
    print(f"\n{len(samples)} samples -> {out}")


if __name__ == "__main__":
    main()
