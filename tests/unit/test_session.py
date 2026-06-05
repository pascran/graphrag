"""Unit tests for app.generate.session — Redis-backed conversation memory.

Uses fakeredis to give a real-shaped FakeAsyncRedis to the module under test
without touching a live broker. The Gemma summarizer is monkeypatched so the
rolling-summary path is observable without an LLM call.
"""
from __future__ import annotations

import uuid

import fakeredis.aioredis
import pytest

from app.db import redis as redis_db
from app.generate import session


@pytest.fixture()
def fake_redis(monkeypatch):
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(redis_db, "_client", client)
    monkeypatch.setattr(redis_db, "get_client", lambda: client)
    return client


@pytest.fixture()
def patched_summarizer(monkeypatch):
    calls = []

    async def fake_summarize(prior_summary: str, overflow_raw: list[str]) -> str:
        calls.append((prior_summary, list(overflow_raw)))
        return f"SUMMARY({len(overflow_raw)} msgs over '{prior_summary}')"

    monkeypatch.setattr(session, "_summarize", fake_summarize)
    return calls


async def test_load_returns_empty_for_unknown_session(fake_redis):
    ctx = await session.load(uuid.uuid4(), "never-seen")
    assert ctx.summary == ""
    assert ctx.turns == []


async def test_append_turn_persists_user_and_assistant(fake_redis):
    tid = uuid.uuid4()
    sid = "thread-1"
    await session.append_turn(
        tid, sid, user_question="안녕", assistant_answer="네, 안녕하세요"
    )
    ctx = await session.load(tid, sid)
    assert ctx.summary == ""
    assert [t["role"] for t in ctx.turns] == ["user", "assistant"]
    assert ctx.turns[0]["content"] == "안녕"
    assert ctx.turns[1]["content"] == "네, 안녕하세요"


async def test_load_skips_corrupt_json_entries(fake_redis):
    tid = uuid.uuid4()
    sid = "thread-corrupt"
    key = session._turns_key(tid, sid)
    await fake_redis.rpush(key, "not-valid-json")
    await fake_redis.rpush(key, '{"role":"user","content":"ok"}')
    ctx = await session.load(tid, sid)
    assert len(ctx.turns) == 1
    assert ctx.turns[0]["content"] == "ok"


async def test_window_does_not_summarize_under_budget(
    fake_redis, patched_summarizer, monkeypatch
):
    monkeypatch.setattr(
        session, "get_settings",
        lambda: type("S", (), {"session_max_turns": 3, "session_ttl_seconds": 1800})(),
    )
    tid = uuid.uuid4()
    sid = "thread-under"
    for i in range(3):
        await session.append_turn(tid, sid, user_question=f"q{i}", assistant_answer=f"a{i}")
    assert patched_summarizer == []
    ctx = await session.load(tid, sid)
    assert len(ctx.turns) == 6
    assert ctx.summary == ""


async def test_window_overflow_triggers_summary_and_trims(
    fake_redis, patched_summarizer, monkeypatch
):
    monkeypatch.setattr(
        session, "get_settings",
        lambda: type("S", (), {"session_max_turns": 2, "session_ttl_seconds": 1800})(),
    )
    tid = uuid.uuid4()
    sid = "thread-over"
    for i in range(3):
        await session.append_turn(tid, sid, user_question=f"q{i}", assistant_answer=f"a{i}")

    assert len(patched_summarizer) == 1
    prior, overflow = patched_summarizer[0]
    assert prior == ""
    assert len(overflow) == 2

    ctx = await session.load(tid, sid)
    assert len(ctx.turns) == 4
    assert "SUMMARY" in ctx.summary
    assert ctx.turns[0]["content"] == "q1"
    assert ctx.turns[-1]["content"] == "a2"


async def test_reset_drops_both_keys(fake_redis):
    tid = uuid.uuid4()
    sid = "thread-reset"
    await session.append_turn(tid, sid, user_question="hi", assistant_answer="hello")
    await fake_redis.set(session._summary_key(tid, sid), "old summary")

    await session.reset(tid, sid)
    ctx = await session.load(tid, sid)
    assert ctx.turns == []
    assert ctx.summary == ""


async def test_isolated_sessions_do_not_leak_into_each_other(fake_redis):
    tid_a, tid_b = uuid.uuid4(), uuid.uuid4()
    await session.append_turn(tid_a, "shared", user_question="A", assistant_answer="a")
    await session.append_turn(tid_b, "shared", user_question="B", assistant_answer="b")

    a_turns = (await session.load(tid_a, "shared")).turns
    b_turns = (await session.load(tid_b, "shared")).turns
    assert [t["content"] for t in a_turns] == ["A", "a"]
    assert [t["content"] for t in b_turns] == ["B", "b"]
