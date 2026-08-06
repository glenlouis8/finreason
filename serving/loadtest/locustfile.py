"""
Load test the live vLLM endpoint with real FinQA prompts.

Reports throughput (req/s) and latency percentiles for actual LLM inference —
not a static-file benchmark. The old version of this file hit nginx '/', which
is why the previously reported "326 req/s, p99 14ms" was meaningless as a model
number: it measured a web server, not generation.

    pip install locust
    python scripts/prepare_data.py          # needs data/sft_test.jsonl
    locust -f serving/loadtest/locustfile.py --host http://localhost:8000

Headless, 10 users, 3 minutes, HTML report:
    locust -f serving/loadtest/locustfile.py --host http://localhost:8000 \
      --headless -u 10 -r 2 -t 3m --html loadtest_report.html

Tune concurrency with -u. Generation takes seconds per request, so 10-50 users
is a realistic range; hundreds just queues up inside vLLM's scheduler.
"""
import json
import random
from pathlib import Path

from locust import HttpUser, task, between, events

MODEL = "glen-louis/finreason-qwen2.5-7b-awq"
MAX_TOKENS = 256          # cap so one slow generation can't skew the run
TEST_DATA = Path(__file__).resolve().parents[2] / "data" / "sft_test.jsonl"

_prompts: list[tuple[str, str]] = []   # (system, user) pairs


@events.test_start.add_listener
def load_prompts(environment, **kwargs):
    """Load real FinQA prompts once, before users spawn."""
    global _prompts
    if not TEST_DATA.exists():
        raise SystemExit(
            f"{TEST_DATA} not found. Run: python scripts/prepare_data.py"
        )
    with TEST_DATA.open() as f:
        for i, line in enumerate(f):
            if i >= 200:
                break
            msgs = json.loads(line)["messages"]
            _prompts.append((msgs[0]["content"], msgs[1]["content"]))
    print(f"[loadtest] loaded {len(_prompts)} FinQA prompts")


class FinReasonUser(HttpUser):
    # Real users reading an answer before asking again. Not a tight hammer loop —
    # that would only measure how fast vLLM rejects a saturated queue.
    wait_time = between(1, 3)

    @task
    def ask_finqa_question(self):
        system, user = random.choice(_prompts)
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
            },
            name="/v1/chat/completions",
            catch_response=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            if not content.strip():
                resp.failure("empty completion")
                return
            resp.success()
