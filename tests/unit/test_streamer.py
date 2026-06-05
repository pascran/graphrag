"""Unit tests for app.generate.streamer.sse — SSE wire-format encoder."""
from __future__ import annotations

import json

from app.generate.streamer import sse


def test_dict_data_is_json_encoded():
    out = sse("token", {"text": "hi"})
    assert isinstance(out, bytes)
    text = out.decode("utf-8")
    assert text.startswith("event: token\ndata: ")
    assert text.endswith("\n\n")
    payload = text.split("data: ", 1)[1].rstrip("\n")
    assert json.loads(payload) == {"text": "hi"}


def test_unicode_is_not_escaped_in_payload():
    out = sse("token", {"text": "안녕하세요"}).decode("utf-8")
    assert "안녕하세요" in out


def test_string_data_passes_through_verbatim():
    out = sse("done", "literal").decode("utf-8")
    assert "data: literal\n\n" in out


def test_event_name_is_used_as_field_name():
    for evt in ("citation", "token", "done", "error"):
        line = sse(evt, {}).decode("utf-8")
        assert line.startswith(f"event: {evt}\n")


def test_empty_dict_renders_empty_json_object():
    out = sse("done", {}).decode("utf-8")
    assert "data: {}\n\n" in out
