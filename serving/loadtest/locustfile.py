"""
Locust load test.
Fires concurrent traffic at the served endpoint, ramps users, and
reports throughput (req/s) + latency percentiles (p50/p95/p99).

Run against the K8s Service via port-forward (see loadtest/README.md):
    kubectl port-forward svc/finreason-web 8080:8080
    locust -f loadtest/locustfile.py --host http://localhost:8080

Training-wheels: hits nginx '/'. Later, swap to the vLLM
/v1/chat/completions endpoint to load-test the real model.
"""
from locust import HttpUser, task, between


class WebUser(HttpUser):
    # each simulated user waits 0.1-0.5s between requests (think-time)
    wait_time = between(0.1, 0.5)

    @task
    def hit_root(self):
        # one request = one transaction Locust times and counts
        self.client.get("/")
