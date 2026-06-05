"""Unit tests for app.generate.prompt.render_rag_prompt."""
from __future__ import annotations

from app.generate.prompt import RetrievedChunk, render_rag_prompt


def test_no_chunks_emits_no_passages_marker():
    msgs = render_rag_prompt("hello", [])
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "(no passages retrieved)" in msgs[-1]["content"]


def test_chunks_are_numbered_and_cite_filename_and_page():
    chunks = [
        RetrievedChunk(filename="alpha.pdf", page=3, text="alpha body"),
        RetrievedChunk(filename="beta.pdf", page=12, text="beta body"),
    ]
    user_msg = render_rag_prompt("Q?", chunks)[-1]["content"]
    assert "[#1 alpha.pdf p.3]" in user_msg
    assert "alpha body" in user_msg
    assert "[#2 beta.pdf p.12]" in user_msg
    assert "beta body" in user_msg


def test_prior_summary_appears_as_system_note_before_user():
    msgs = render_rag_prompt(
        "follow-up question",
        [],
        prior_summary="user introduced themselves as Park.",
    )
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "system"
    assert "Park" in msgs[1]["content"]
    assert msgs[-1]["role"] == "user"


def test_prior_turns_are_inserted_between_summary_and_current_user():
    prior_turns = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    msgs = render_rag_prompt(
        "second question",
        [],
        prior_turns=prior_turns,
        prior_summary="topic: testing",
    )
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "first question"
    assert msgs[3]["content"] == "first answer"
    assert "second question" in msgs[4]["content"]


def test_no_session_args_keeps_two_message_shape():
    msgs = render_rag_prompt("plain", [])
    assert [m["role"] for m in msgs] == ["system", "user"]
