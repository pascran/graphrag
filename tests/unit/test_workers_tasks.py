"""Unit tests for app.workers.tasks._run + ingest_document_task."""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from app.workers import tasks


@pytest.fixture()
def patched_status(monkeypatch):
    set_status = AsyncMock()
    monkeypatch.setattr(tasks, "_set_status", set_status)
    return set_status


async def test_run_marks_running_then_completed_on_success(patched_status, monkeypatch):
    ingest = AsyncMock(return_value=type("R", (), {"chunk_count": 7, "document_id": "x", "page_count": 1})())
    monkeypatch.setattr(tasks, "ingest_document", ingest)

    tid, did, jid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await tasks._run(
        tenant_id=tid, document_id=did, job_id=jid,
        file_path="/tmp/x.pdf", filename="x.pdf", doc_type=None,
    )
    assert patched_status.await_count == 2
    first = patched_status.await_args_list[0].kwargs
    last = patched_status.await_args_list[1].kwargs
    assert first["status"] == "running"
    assert first["progress"] == 0.0
    assert last["status"] == "completed"
    assert last["progress"] == 1.0
    ingest.assert_awaited_once()


async def test_run_marks_failed_and_re_raises_on_pipeline_error(patched_status, monkeypatch):
    monkeypatch.setattr(tasks, "ingest_document", AsyncMock(side_effect=RuntimeError("ocr boom")))

    with pytest.raises(RuntimeError, match="ocr boom"):
        await tasks._run(
            tenant_id=uuid.uuid4(), document_id=uuid.uuid4(), job_id=uuid.uuid4(),
            file_path="/tmp/x.pdf", filename="x.pdf", doc_type=None,
        )

    statuses = [c.kwargs["status"] for c in patched_status.await_args_list]
    assert statuses == ["running", "failed"]
    failed = patched_status.await_args_list[-1].kwargs
    assert "RuntimeError" in failed["error"]
    assert "ocr boom" in failed["error"]


def test_celery_task_wires_uuids_through_run(monkeypatch):
    captured = {}

    async def fake_run(**kw):
        captured.update(kw)

    monkeypatch.setattr(tasks, "_run", fake_run)
    monkeypatch.setattr(tasks.asyncio, "run", lambda coro: asyncio.new_event_loop().run_until_complete(coro))

    tid, did, jid = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    out = tasks.ingest_document_task.run(
        tenant_id=tid, document_id=did, job_id=jid,
        file_path="/tmp/a.pdf", filename="a.pdf", doc_type="policy",
    )
    assert out == {"document_id": did, "status": "completed"}
    assert str(captured["tenant_id"]) == tid
    assert str(captured["document_id"]) == did
    assert str(captured["job_id"]) == jid
    assert captured["doc_type"] == "policy"
