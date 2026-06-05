"""Shared fixtures for integration tests.

Skips the entire integration suite when the live API is unreachable so
unit-only test runs stay green in CI without docker compose up.
"""
from __future__ import annotations

import os
import secrets
import subprocess

import httpx
import pytest

API_BASE = os.environ.get("LLM_ENGINE_BASE_URL", "http://localhost:8000")


def _api_alive() -> bool:
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_live_api():
    if not _api_alive():
        pytest.skip(f"live API at {API_BASE} unreachable; skipping integration tests")


@pytest.fixture(scope="session")
def api_key() -> str:
    suffix = secrets.token_hex(4)
    tenant_name = f"itest-{suffix}"
    issue = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "app",
            "python", "scripts/create_api_key.py", "issue",
            "--new-tenant", tenant_name, "--name", "integration",
        ],
        capture_output=True, text=True, check=True,
    )
    plain = next(
        (line for line in issue.stdout.splitlines() if line.startswith("graphrag_")),
        "",
    )
    if not plain:
        pytest.skip(f"could not mint test key:\n{issue.stdout}\n{issue.stderr}")

    yield plain

    listing = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "app",
            "python", "scripts/create_api_key.py", "list",
            "--tenant", tenant_name,
        ],
        capture_output=True, text=True,
    )
    for line in listing.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "True":
            subprocess.run(
                [
                    "docker", "compose", "exec", "-T", "app",
                    "python", "scripts/create_api_key.py", "revoke",
                    "--key-id", parts[0],
                ],
                capture_output=True,
            )


@pytest.fixture()
def client(api_key: str):
    with httpx.Client(
        base_url=API_BASE,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    ) as c:
        yield c
