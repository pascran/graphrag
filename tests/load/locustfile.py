"""Locust load test for the graphrag LLM Engine.

Run from the repo root:

  export LLM_ENGINE_API_KEY=graphrag_xxx
  locust -f tests/load/locustfile.py \\
      --host=http://localhost:8000 \\
      --users=40 --spawn-rate=4 --run-time=2m --headless

Mixes 70% non-stream casual + factual queries with 30% list calls.
The streaming /v1/query path is exercised by the SSE task at lower
weight to keep the GPU bucket from saturating during a baseline run.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task


CASUAL = ["안녕!", "Hi there", "오늘 날씨 어때?", "tell me a joke"]
FACTUAL = [
    "휴가 정책은 어떻게 돼?",
    "출장 신청은 누구에게 해야 해?",
    "보안 사고 발생 시 절차를 알려줘",
    "What is the procurement policy?",
]


@events.test_start.add_listener
def _check_key(environment, **_):
    if not os.environ.get("LLM_ENGINE_API_KEY"):
        environment.runner.quit()
        raise RuntimeError(
            "LLM_ENGINE_API_KEY env var is required (mint via "
            "scripts/create_api_key.py and export it before running)"
        )


class RagUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.client.headers.update(
            {"Authorization": f"Bearer {os.environ['LLM_ENGINE_API_KEY']}"}
        )

    @task(5)
    def query_casual_json(self):
        self.client.post(
            "/v1/query",
            json={
                "question": random.choice(CASUAL),
                "mode": "auto",
                "top_k": 3,
                "stream": False,
            },
            name="POST /v1/query [casual,json]",
        )

    @task(4)
    def query_factual_json(self):
        self.client.post(
            "/v1/query",
            json={
                "question": random.choice(FACTUAL),
                "mode": "auto",
                "top_k": 5,
                "stream": False,
            },
            name="POST /v1/query [fact,json]",
        )

    @task(3)
    def list_documents(self):
        self.client.get("/v1/documents", name="GET /v1/documents")

    @task(2)
    def query_streaming(self):
        with self.client.post(
            "/v1/query",
            json={
                "question": random.choice(CASUAL),
                "mode": "auto",
                "top_k": 3,
                "stream": True,
            },
            name="POST /v1/query [stream]",
            stream=True,
            catch_response=True,
        ) as resp:
            saw_done = False
            for line in resp.iter_lines(decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                if line.startswith("event:") and "done" in line:
                    saw_done = True
                    break
            if saw_done:
                resp.success()
            else:
                resp.failure("no done event")
