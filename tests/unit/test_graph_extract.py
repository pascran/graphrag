"""Unit tests for app.ingest.graph_extract — LLM-driven entity + relation
extraction from a chunk."""
from __future__ import annotations

from app.ingest import graph_extract


def _patch_llm(monkeypatch, response):
    async def fake_chat_once(messages, **kw):
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(graph_extract, "chat_once", fake_chat_once)


async def test_extract_parses_well_formed_json(monkeypatch):
    payload = """{
      "entities": [
        {"name": "Acme Corp", "type": "ORG", "description": "fictional company"},
        {"name": "Seoul", "type": "LOCATION", "description": "korean capital"}
      ],
      "relations": [
        {"source": "Acme Corp", "target": "Seoul",
         "kind": "headquartered_in", "description": "HQ since 1999"}
      ]
    }"""
    _patch_llm(monkeypatch, payload)

    out = await graph_extract.extract_entities_and_relations("synthetic chunk text")
    assert len(out.entities) == 2
    assert out.entities[0].name == "Acme Corp"
    assert out.entities[0].type == "ORG"
    assert out.entities[1].type == "LOCATION"
    assert len(out.relations) == 1
    assert out.relations[0].source == "Acme Corp"
    assert out.relations[0].kind == "headquartered_in"


async def test_extract_unwraps_markdown_fenced_json(monkeypatch):
    payload = '```json\n{"entities":[{"name":"X","type":"OTHER","description":""}],"relations":[]}\n```'
    _patch_llm(monkeypatch, payload)
    out = await graph_extract.extract_entities_and_relations("text")
    assert [e.name for e in out.entities] == ["X"]


async def test_extract_returns_empty_for_blank_input(monkeypatch):
    chat_called = {"hit": False}

    async def fake_chat_once(messages, **kw):
        chat_called["hit"] = True
        return "{}"

    monkeypatch.setattr(graph_extract, "chat_once", fake_chat_once)
    out = await graph_extract.extract_entities_and_relations("")
    assert out.entities == [] and out.relations == []
    assert chat_called["hit"] is False


async def test_extract_returns_empty_on_malformed_json(monkeypatch):
    _patch_llm(monkeypatch, "this is not json at all")
    out = await graph_extract.extract_entities_and_relations("text")
    assert out.entities == [] and out.relations == []


async def test_extract_returns_empty_on_llm_error(monkeypatch):
    _patch_llm(monkeypatch, RuntimeError("vllm down"))
    out = await graph_extract.extract_entities_and_relations("text")
    assert out.entities == [] and out.relations == []


async def test_extract_skips_entities_with_empty_name(monkeypatch):
    payload = (
        '{"entities":[{"name":"","type":"ORG","description":""},'
        '{"name":"  ","type":"PERSON","description":""},'
        '{"name":"Real","type":"ORG","description":""}],'
        '"relations":[]}'
    )
    _patch_llm(monkeypatch, payload)
    out = await graph_extract.extract_entities_and_relations("text")
    assert [e.name for e in out.entities] == ["Real"]


async def test_extract_drops_relations_referencing_unknown_entities(monkeypatch):
    payload = (
        '{"entities":[{"name":"A","type":"ORG","description":""},'
        '{"name":"B","type":"ORG","description":""}],'
        '"relations":['
        '{"source":"A","target":"B","kind":"related","description":""},'
        '{"source":"A","target":"GHOST","kind":"unknown","description":""},'
        '{"source":"NOPE","target":"B","kind":"unknown","description":""}'
        ']}'
    )
    _patch_llm(monkeypatch, payload)
    out = await graph_extract.extract_entities_and_relations("text")
    assert len(out.relations) == 1
    assert out.relations[0].source == "A" and out.relations[0].target == "B"


async def test_extract_normalizes_entity_type_to_uppercase(monkeypatch):
    payload = (
        '{"entities":[{"name":"Acme","type":"org","description":""}],'
        '"relations":[]}'
    )
    _patch_llm(monkeypatch, payload)
    out = await graph_extract.extract_entities_and_relations("text")
    assert out.entities[0].type == "ORG"


async def test_extract_handles_missing_fields_gracefully(monkeypatch):
    payload = '{"entities":[{"name":"X","type":"ORG"}],"relations":[]}'
    _patch_llm(monkeypatch, payload)
    out = await graph_extract.extract_entities_and_relations("text")
    assert out.entities[0].description == ""


async def test_extract_truncates_extremely_long_input(monkeypatch):
    captured = {}

    async def fake_chat_once(messages, **kw):
        captured["user_msg"] = messages[-1]["content"]
        return '{"entities":[],"relations":[]}'

    monkeypatch.setattr(graph_extract, "chat_once", fake_chat_once)
    huge = "policy text. " * 5000
    out = await graph_extract.extract_entities_and_relations(huge)
    assert out.entities == []
    assert "policy text" in captured["user_msg"]
