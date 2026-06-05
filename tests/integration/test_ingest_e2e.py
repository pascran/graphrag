"""End-to-end ingest test:  POST /v1/upload  ->  poll /v1/jobs/{id}  ->  /v1/query.

Drives the full pipeline (Chandra OCR -> chunker -> embedder -> Qdrant + Neo4j).
Skipped automatically when the fixture PDF is not present so that fast unit
runs stay green; running it requires the live stack from conftest.py.
"""
from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_pdfs" / "korean_form.pdf"


@pytest.fixture(scope="module")
def fixture_pdf() -> Path:
    if not FIXTURE.exists():
        pytest.fail(
            f"fixture missing: {FIXTURE}. Generate it with "
            "scripts/build_fixture_pdf.py before running the E2E suite."
        )
    return FIXTURE


def _poll_job(client: httpx.Client, job_id: str, *, deadline_s: int = 240) -> dict:
    deadline = time.monotonic() + deadline_s
    last: dict = {}
    while time.monotonic() < deadline:
        r = client.get(f"/v1/jobs/{job_id}")
        if r.status_code == 200:
            last = r.json()
            if last.get("status") in {"completed", "failed"}:
                return last
        time.sleep(2.0)
    last["_timed_out"] = True
    return last


def test_upload_then_query_round_trip(client: httpx.Client, fixture_pdf: Path):
    with fixture_pdf.open("rb") as f:
        upload = client.post(
            "/v1/upload",
            files={"files": (fixture_pdf.name, f, "application/pdf")},
            data={"doc_type": "policy"},
        )
    assert upload.status_code == 202, upload.text
    body = upload.json()
    assert body["job_id"]
    assert len(body["accepted_files"]) == 1
    assert body["rejected_files"] == []
    accepted_filename = body["accepted_files"][0]["name"]

    job = _poll_job(client, body["job_id"])
    assert not job.get("_timed_out"), f"ingest exceeded deadline (last={job})"
    assert job["status"] == "completed", f"job did not complete: {job}"

    docs = client.get("/v1/documents").json()
    assert any(d["filename"] == accepted_filename for d in docs)

    q = client.post(
        "/v1/query",
        json={
            "question": "이 문서의 제목이 뭐야?",
            "mode": "fact",
            "top_k": 5,
            "stream": False,
        },
    )
    assert q.status_code == 200
    qbody = q.json()
    assert qbody["mode_used"] == "fact"
    assert qbody["sources"], "expected at least one source citation"
    assert any(s["filename"] == accepted_filename for s in qbody["sources"])


def test_upload_rejects_unsupported_extension(client: httpx.Client, tmp_path: Path):
    bad = tmp_path / "thing.exe"
    bad.write_bytes(b"MZ")
    with bad.open("rb") as f:
        r = client.post(
            "/v1/upload",
            files={"files": (bad.name, f, "application/octet-stream")},
        )
    assert r.status_code == 202
    body = r.json()
    assert body["accepted_files"] == []
    assert len(body["rejected_files"]) == 1
    assert ".exe" in body["rejected_files"][0]["reason"]


def test_upload_dedupes_same_content(client: httpx.Client, fixture_pdf: Path):
    """Re-uploading the same bytes should be rejected as a duplicate."""
    with fixture_pdf.open("rb") as f:
        r = client.post(
            "/v1/upload",
            files={"files": (fixture_pdf.name, f, "application/pdf")},
        )
    assert r.status_code == 202
    body = r.json()
    assert body["accepted_files"] == []
    assert len(body["rejected_files"]) == 1
    assert "duplicate" in body["rejected_files"][0]["reason"].lower()
