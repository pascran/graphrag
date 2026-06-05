"""POST /v1/query — RAG question answering with SSE streaming."""
from __future__ import annotations

from collections.abc import AsyncIterator
from time import perf_counter

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse

from app.core.auth import AuthenticatedTenant
from app.db import qdrant as qdrant_db
from app.deps import current_tenant
from app.generate.llm import stream_chat
from app.generate.prompt import render_rag_prompt
from app.generate.session import append_turn, load as load_session
from app.generate.streamer import sse
from app.generate.llm import chat_once
from app.models.schemas import QueryRequest, QueryResponse
from app.retrieve.orchestrator import retrieve
from app.utils.logging import get_logger

router = APIRouter(prefix="/v1", tags=["query"])
log = get_logger("app.api.query")


async def _generate(req: QueryRequest, tenant_id) -> AsyncIterator[bytes]:
    t0 = perf_counter()
    qclient = qdrant_db.get_client()
    try:
        retrieval = await retrieve(
            qclient,
            tenant_id=tenant_id,
            question=req.question,
            mode=req.mode,  # type: ignore[arg-type]
            top_k=req.top_k,
            payload_filters=req.filters,
        )
    except Exception as e:
        log.exception("retrieval_failed")
        yield sse("error", {"message": f"retrieval failed: {type(e).__name__}: {e}"})
        return

    sources = [{"filename": c.filename, "page": c.page} for c in retrieval.chunks]
    yield sse("citation", {"sources": sources})

    prior_summary, prior_turns = "", []
    if req.session_id:
        try:
            ctx = await load_session(tenant_id, req.session_id)
            prior_summary, prior_turns = ctx.summary, ctx.turns
        except Exception as e:
            log.warning("session_load_failed", error=str(e))

    messages = render_rag_prompt(
        req.question,
        retrieval.chunks,
        prior_turns=prior_turns,
        prior_summary=prior_summary,
    )

    answer_buf: list[str] = []
    try:
        async for delta in stream_chat(messages):
            answer_buf.append(delta)
            yield sse("token", {"text": delta})
    except Exception as e:
        log.exception("generation_failed")
        yield sse("error", {"message": f"generation failed: {type(e).__name__}: {e}"})
        return

    if req.session_id:
        try:
            await append_turn(
                tenant_id,
                req.session_id,
                user_question=req.question,
                assistant_answer="".join(answer_buf),
            )
        except Exception as e:
            log.warning("session_append_failed", error=str(e))

    yield sse("done", {"mode_used": retrieval.mode_used,
                       "latency_ms": int((perf_counter() - t0) * 1000)})


@router.post("/query", status_code=status.HTTP_200_OK, response_model=None)
async def query(
    req: QueryRequest,
    auth: AuthenticatedTenant = Depends(current_tenant),
):
    if req.stream:
        return StreamingResponse(
            _generate(req, auth.tenant_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-stream JSON path — convenient for Swagger UI manual testing.
    t0 = perf_counter()
    qclient = qdrant_db.get_client()
    retrieval = await retrieve(
        qclient,
        tenant_id=auth.tenant_id,
        question=req.question,
        mode=req.mode,  # type: ignore[arg-type]
        top_k=req.top_k,
        payload_filters=req.filters,
    )
    sources = [{"filename": c.filename, "page": c.page} for c in retrieval.chunks]

    prior_summary, prior_turns = "", []
    if req.session_id:
        try:
            ctx = await load_session(auth.tenant_id, req.session_id)
            prior_summary, prior_turns = ctx.summary, ctx.turns
        except Exception as e:
            log.warning("session_load_failed", error=str(e))

    messages = render_rag_prompt(
        req.question,
        retrieval.chunks,
        prior_turns=prior_turns,
        prior_summary=prior_summary,
    )
    answer = await chat_once(messages)

    if req.session_id:
        try:
            await append_turn(
                auth.tenant_id,
                req.session_id,
                user_question=req.question,
                assistant_answer=answer,
            )
        except Exception as e:
            log.warning("session_append_failed", error=str(e))

    return QueryResponse(
        answer=answer,
        sources=sources,
        mode_used=retrieval.mode_used,
        latency_ms=int((perf_counter() - t0) * 1000),
    )
