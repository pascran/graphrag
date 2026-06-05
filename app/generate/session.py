"""Per-session conversation memory in Redis.

Layout (one session = one conversation thread, scoped to a tenant):

  sess:{tenant_id}:{session_id}:turns    LIST of JSON {"role","content"}
  sess:{tenant_id}:{session_id}:summary  STRING — running summary of turns that
                                          have been rolled off the live window

Both keys share the same sliding TTL (settings.session_ttl_seconds, default 30min).
Every read / write touches both keys' TTLs so an active conversation does not
expire mid-thread.

Turn budget: up to settings.session_max_turns *user/assistant pairs* are kept
verbatim; once the window overflows, the oldest pair is folded into the summary
via Gemma. Summary degrades gracefully — if the summarizer fails we drop the
oldest turn rather than blocking the request.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis

from app.config import get_settings
from app.db import redis as redis_db
from app.generate.llm import chat_once
from app.utils.logging import get_logger

log = get_logger("app.generate.session")


@dataclass(frozen=True)
class SessionContext:
    summary: str
    turns: list[dict[str, str]]


def _turns_key(tenant_id: uuid.UUID, session_id: str) -> str:
    return f"sess:{tenant_id}:{session_id}:turns"


def _summary_key(tenant_id: uuid.UUID, session_id: str) -> str:
    return f"sess:{tenant_id}:{session_id}:summary"


async def _touch_ttl(client: aioredis.Redis, *keys: str, ttl: int) -> None:
    pipe = client.pipeline()
    for k in keys:
        pipe.expire(k, ttl)
    await pipe.execute()


async def load(tenant_id: uuid.UUID, session_id: str) -> SessionContext:
    settings = get_settings()
    client = redis_db.get_client()
    tk, sk = _turns_key(tenant_id, session_id), _summary_key(tenant_id, session_id)
    raw_turns = await client.lrange(tk, 0, -1)
    summary = await client.get(sk)
    turns: list[dict[str, str]] = []
    for s in raw_turns or []:
        try:
            t = json.loads(s)
            if isinstance(t, dict) and "role" in t and "content" in t:
                turns.append({"role": str(t["role"]), "content": str(t["content"])})
        except json.JSONDecodeError:
            continue
    if turns or summary:
        await _touch_ttl(client, tk, sk, ttl=settings.session_ttl_seconds)
    return SessionContext(summary=summary or "", turns=turns)


async def append_turn(
    tenant_id: uuid.UUID,
    session_id: str,
    *,
    user_question: str,
    assistant_answer: str,
) -> None:
    """Append the latest exchange and roll older turns into the summary if over budget."""
    settings = get_settings()
    client = redis_db.get_client()
    tk = _turns_key(tenant_id, session_id)
    sk = _summary_key(tenant_id, session_id)

    pipe = client.pipeline()
    pipe.rpush(
        tk,
        json.dumps({"role": "user", "content": user_question}, ensure_ascii=False),
        json.dumps({"role": "assistant", "content": assistant_answer}, ensure_ascii=False),
    )
    pipe.expire(tk, settings.session_ttl_seconds)
    pipe.expire(sk, settings.session_ttl_seconds)
    await pipe.execute()

    over = await client.llen(tk) - settings.session_max_turns * 2
    if over <= 0:
        return

    overflow_raw = await client.lrange(tk, 0, over - 1)
    prior_summary = await client.get(sk) or ""
    new_summary = await _summarize(prior_summary, overflow_raw)
    pipe = client.pipeline()
    pipe.ltrim(tk, over, -1)
    if new_summary:
        pipe.set(sk, new_summary, ex=settings.session_ttl_seconds)
    pipe.expire(tk, settings.session_ttl_seconds)
    await pipe.execute()


async def reset(tenant_id: uuid.UUID, session_id: str) -> None:
    client = redis_db.get_client()
    await client.delete(_turns_key(tenant_id, session_id), _summary_key(tenant_id, session_id))


async def _summarize(prior_summary: str, overflow_raw: list[str]) -> str:
    overflow_msgs: list[str] = []
    for s in overflow_raw:
        try:
            t = json.loads(s)
            overflow_msgs.append(f"{t.get('role','?')}: {t.get('content','')}")
        except json.JSONDecodeError:
            continue
    if not overflow_msgs:
        return prior_summary

    body = "\n".join(overflow_msgs)
    sys_prompt = (
        "Summarize the conversation so a downstream RAG system can stay on topic. "
        "Keep it under 6 sentences, factual, neutral, in the same language as the "
        "conversation. Preserve names, numbers, decisions, and unresolved questions."
    )
    user_prompt = (
        f"Existing summary (may be empty):\n{prior_summary or '(none)'}\n\n"
        f"New turns to fold in:\n{body}\n\n"
        f"Updated summary:"
    )
    try:
        return (
            await chat_once(
                [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=400,
                timeout=20.0,
            )
        ).strip()
    except Exception as e:
        log.warning("session_summarize_failed", error=str(e))
        return prior_summary
