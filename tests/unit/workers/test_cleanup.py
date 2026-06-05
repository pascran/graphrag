"""Unit tests for app.workers.cleanup — orphan :Entity scan/delete task.

The Neo4j driver is mocked using the same async-context-manager pattern as
``tests/unit/test_graph_indexer.py``. We assert on the Cypher strings and
parameters recorded by the fake session, never against a live database.
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import ServiceUnavailable

from app.workers import cleanup as cleanup_mod


# --------------------------------------------------------------------------- #
# Fake Neo4j driver — records every cypher + params, returns scripted records.
# --------------------------------------------------------------------------- #


def _make_driver(*, scan_records=None, delete_count=0):
    """Build a fake AsyncDriver with scripted scan + delete results.

    Parameters
    ----------
    scan_records : list[dict] | None
        Records returned by the SCAN query (``RETURN e.name, e.tenant_id, id``).
    delete_count : int
        Value returned by ``record["deleted"]`` on the DELETE query.
    """
    scan_records = scan_records or []
    statements: list[tuple[str, dict]] = []

    # --- "record" returned by SCAN: behaves like a Neo4j Record (.get/[key]) -
    class _FakeRecord:
        def __init__(self, payload: dict):
            self._payload = payload

        def get(self, key, default=None):
            return self._payload.get(key, default)

        def __getitem__(self, key):
            return self._payload[key]

    # --- async iterator returned by SCAN's await session.run(...) -------------
    class _AsyncIterResult:
        def __init__(self, records, delete_count):
            self._records = [_FakeRecord(r) for r in records]
            self._delete_count = delete_count
            self.consume = AsyncMock()

        def __aiter__(self):
            self._iter = iter(self._records)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as e:
                raise StopAsyncIteration from e

        async def single(self):
            return _FakeRecord({"deleted": self._delete_count})

    sess = MagicMock()

    async def fake_run(cypher, **kwargs):
        statements.append((cypher, dict(kwargs)))
        return _AsyncIterResult(scan_records, delete_count)

    sess.run = fake_run

    @asynccontextmanager
    async def fake_session():
        yield sess

    driver = MagicMock()
    driver.session = fake_session
    driver._statements = statements
    return driver


@pytest.fixture()
def patched_driver(monkeypatch):
    """Default empty-result driver, swappable inside tests via the holder."""
    holder = {"driver": _make_driver()}

    def _get_driver():
        return holder["driver"]

    monkeypatch.setattr(cleanup_mod, "get_driver", _get_driver)
    return holder


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


async def test_dry_run_scans_only_no_delete(patched_driver):
    """Spec (a): dry_run=True returns scanned + would_delete, does NOT delete."""
    orphan_rows = [
        {"name": "Acme", "tenant_id": str(uuid.uuid4()), "id": "4:abc:1"},
        {"name": "Bobcorp", "tenant_id": str(uuid.uuid4()), "id": "4:abc:2"},
        {"name": "Cogswell", "tenant_id": str(uuid.uuid4()), "id": "4:abc:3"},
    ]
    patched_driver["driver"] = _make_driver(scan_records=orphan_rows)
    drv = patched_driver["driver"]

    summary = await cleanup_mod._run(dry_run=True, tenant_id=None, limit=500)

    assert summary["dry_run"] is True
    assert summary["scanned"] == 3
    assert summary["would_delete"] == 3
    assert "deleted" not in summary
    assert len(summary["sample"]) == 3
    assert summary["sample"][0]["name"] == "Acme"

    cyphers = [c for c, _ in drv._statements]
    # Exactly one statement was issued (the scan); no DETACH DELETE.
    assert len(cyphers) == 1
    assert "DETACH DELETE" not in cyphers[0]
    assert "MATCH (e:Entity)" in cyphers[0]
    assert "NOT (e)<-[:MENTIONS]-(:Chunk)" in cyphers[0]


async def test_apply_executes_detach_delete(patched_driver):
    """Spec (b): dry_run=False executes DETACH DELETE."""
    orphan_rows = [
        {"name": "Acme", "tenant_id": str(uuid.uuid4()), "id": "4:abc:1"},
        {"name": "Bobcorp", "tenant_id": str(uuid.uuid4()), "id": "4:abc:2"},
    ]
    patched_driver["driver"] = _make_driver(
        scan_records=orphan_rows, delete_count=2
    )
    drv = patched_driver["driver"]

    summary = await cleanup_mod._run(dry_run=False, tenant_id=None, limit=100)

    assert summary["dry_run"] is False
    assert summary["scanned"] == 2
    assert summary["deleted"] == 2
    assert "would_delete" not in summary

    cyphers = [c for c, _ in drv._statements]
    assert len(cyphers) == 2
    assert "DETACH DELETE" in cyphers[1]
    assert "MATCH (e:Entity)" in cyphers[1]


async def test_tenant_id_filter_in_cypher(patched_driver):
    """Spec (c): tenant_id filter scopes the Cypher correctly."""
    tid = str(uuid.uuid4())
    orphan_rows = [{"name": "Acme", "tenant_id": tid, "id": "4:abc:1"}]
    patched_driver["driver"] = _make_driver(
        scan_records=orphan_rows, delete_count=1
    )
    drv = patched_driver["driver"]

    summary = await cleanup_mod._run(dry_run=False, tenant_id=tid, limit=50)

    assert summary["tenant_id"] == tid
    assert summary["deleted"] == 1

    # Both statements (scan + delete) must include the tenant predicate
    # AND receive tenant_id as a bound parameter.
    for cy, params in drv._statements:
        assert "e.tenant_id = $tenant_id" in cy
        assert params.get("tenant_id") == tid


async def test_limit_parameter_passed_through(patched_driver):
    """Spec (d): limit parameter is bound on every cypher invocation."""
    patched_driver["driver"] = _make_driver(scan_records=[], delete_count=0)
    drv = patched_driver["driver"]

    await cleanup_mod._run(dry_run=True, tenant_id=None, limit=42)

    assert len(drv._statements) == 1
    _, params = drv._statements[0]
    assert params["limit"] == 42


async def test_empty_result_no_orphans(patched_driver):
    """Spec (e): empty result handled cleanly — zero counts, empty sample."""
    patched_driver["driver"] = _make_driver(scan_records=[], delete_count=0)
    drv = patched_driver["driver"]

    summary = await cleanup_mod._run(dry_run=False, tenant_id=None, limit=1000)

    assert summary["scanned"] == 0
    assert summary["deleted"] == 0
    assert summary["sample"] == []
    # Scan + delete both ran (even with zero results) so the operator
    # gets an unambiguous "we tried and there was nothing" signal.
    assert len(drv._statements) == 2


async def test_sample_truncated_to_20_rows(patched_driver):
    """Sample list is capped at 20 entries regardless of scanned count."""
    rows = [
        {"name": f"e{i}", "tenant_id": "t", "id": f"4:x:{i}"}
        for i in range(50)
    ]
    patched_driver["driver"] = _make_driver(scan_records=rows, delete_count=0)

    summary = await cleanup_mod._run(dry_run=True, tenant_id=None, limit=1000)

    assert summary["scanned"] == 50
    assert summary["would_delete"] == 50
    assert len(summary["sample"]) == 20


def test_celery_task_retries_on_transient_neo4j_error(monkeypatch):
    """Spec (f): a transient Neo4j error must trigger Celery retry."""

    async def boom(**kwargs):
        raise ServiceUnavailable("neo4j is down")

    monkeypatch.setattr(cleanup_mod, "_scan_orphans", boom)

    class _RetryRaised(Exception):
        def __init__(self, exc, countdown):
            self.exc = exc
            self.countdown = countdown

    captured: dict = {}

    def fake_retry(*, exc=None, countdown=None, **_):
        captured["exc"] = exc
        captured["countdown"] = countdown
        raise _RetryRaised(exc=exc, countdown=countdown)

    # Bind the fake retry to the task instance so `self.retry(...)` calls it.
    monkeypatch.setattr(
        cleanup_mod.cleanup_orphan_entities, "retry", fake_retry
    )

    with pytest.raises(_RetryRaised) as excinfo:
        cleanup_mod.cleanup_orphan_entities.run(
            dry_run=True, tenant_id=None, limit=10
        )

    assert isinstance(excinfo.value.exc, ServiceUnavailable)
    # First-attempt backoff is 30 * 2^0 = 30 seconds.
    assert excinfo.value.countdown == 30
