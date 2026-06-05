"""Unit tests for Pydantic request/response schemas."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    DocumentOut,
    HealthCheckOut,
    JobOut,
    QueryRequest,
    QueryResponse,
    TenantOut,
    UploadAccepted,
)


def test_query_request_defaults():
    q = QueryRequest(question="hi")
    assert q.mode == "auto"
    assert q.top_k == 5
    assert q.filters is None
    assert q.stream is True
    assert q.session_id is None


def test_query_request_rejects_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_top_k_bounds():
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", top_k=21)
    QueryRequest(question="ok", top_k=1)
    QueryRequest(question="ok", top_k=20)


def test_query_request_session_id_max_length():
    QueryRequest(question="ok", session_id="x" * 128)
    with pytest.raises(ValidationError):
        QueryRequest(question="ok", session_id="x" * 129)


def test_query_request_filters_accepts_primitives_and_none_values():
    q = QueryRequest(
        question="ok",
        filters={"doc_type": "manual", "page": 3, "active": True, "removed": None},
    )
    assert q.filters == {"doc_type": "manual", "page": 3, "active": True, "removed": None}


def test_query_response_shape():
    r = QueryResponse(
        answer="A", sources=[{"filename": "f.pdf", "page": 1}], mode_used="fact", latency_ms=12
    )
    assert r.answer == "A"
    assert r.sources[0]["filename"] == "f.pdf"


def test_health_check_payload():
    h = HealthCheckOut(status="ok", checks={"qdrant": "ok", "neo4j": "ok"})
    assert h.status == "ok"
    assert h.checks["qdrant"] == "ok"


def test_upload_accepted_payload():
    u = UploadAccepted(
        job_id=uuid.uuid4(),
        accepted_files=["a.pdf"],
        rejected_files=[{"name": "b.exe", "reason": "ext"}],
    )
    assert u.accepted_files == ["a.pdf"]
    assert u.rejected_files[0]["reason"] == "ext"


def test_tenant_out_from_attributes():
    class _Row:
        id = uuid.uuid4()
        name = "default"
        created_at = datetime.now(tz=timezone.utc)

    t = TenantOut.model_validate(_Row())
    assert t.name == "default"


def test_job_out_progress_bounds():
    j = JobOut(
        id=uuid.uuid4(),
        status="running",
        progress=0.5,
        error=None,
        created_at=datetime.now(tz=timezone.utc),
        updated_at=datetime.now(tz=timezone.utc),
    )
    assert 0.0 <= j.progress <= 1.0
    with pytest.raises(ValidationError):
        JobOut(
            id=uuid.uuid4(),
            status="running",
            progress=1.5,
            error=None,
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
        )


def test_document_out_optional_fields_can_be_null():
    d = DocumentOut(
        id=uuid.uuid4(),
        filename="x.pdf",
        file_hash="0" * 64,
        doc_type=None,
        page_count=None,
        status="pending",
        created_at=datetime.now(tz=timezone.utc),
    )
    assert d.doc_type is None
    assert d.page_count is None
