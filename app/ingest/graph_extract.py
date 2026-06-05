"""LLM-driven entity + relation extraction for one chunk of text.

The orchestrator calls this once per chunk during ingest. Output feeds
graph_indexer.upsert_chunks_with_graph which materialises :Entity nodes,
:MENTIONS edges from chunks, and :RELATES_TO edges between entities.

Failure modes degrade to an empty extraction so a flaky LLM does not break
the whole ingest pipeline — we lose graph density on a chunk, not the doc.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.generate.llm import chat_once
from app.utils.logging import get_logger

log = get_logger("app.ingest.graph_extract")

_VALID_TYPES = {
    "PERSON", "ORG", "LOCATION", "EVENT", "PRODUCT",
    "DATE", "MONEY", "POLICY", "CONCEPT", "OTHER",
}

_MAX_CHARS_PER_CHUNK = 4000


_SYSTEM = (
    "You extract a knowledge graph from a single passage of an enterprise "
    "document. Reply with EXACTLY one JSON object (no commentary, no "
    "markdown fence) of the form:\n"
    "{\n"
    '  "entities": [{"name": "...", "type": "...", "description": "..."}],\n'
    '  "relations": [{"source": "...", "target": "...", "kind": "...", "description": "..."}]\n'
    "}\n"
    "Rules:\n"
    "- type ∈ {PERSON, ORG, LOCATION, EVENT, PRODUCT, DATE, MONEY, POLICY, CONCEPT, OTHER}\n"
    "- name is the canonical surface form actually present in the passage\n"
    "- relation source/target MUST appear in the entities list above\n"
    "- relation.kind is a short snake_case verb phrase (e.g. employs, "
    "approves, headquartered_in)\n"
    "- 0-15 entities, 0-15 relations is the typical range\n"
    "- Empty arrays are valid when the passage has no clear named items"
)


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    type: str
    description: str = ""


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    kind: str
    description: str = ""


@dataclass(frozen=True)
class ChunkExtraction:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


_EMPTY = ChunkExtraction(entities=[], relations=[])


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else s


def _parse(raw: str) -> ChunkExtraction:
    body = _strip_fences(raw)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        log.warning("graph_extract_parse_fail", body_preview=body[:200])
        return _EMPTY
    if not isinstance(obj, dict):
        return _EMPTY

    entities: list[ExtractedEntity] = []
    seen_names: set[str] = set()
    for raw_e in obj.get("entities") or []:
        if not isinstance(raw_e, dict):
            continue
        name = str(raw_e.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        type_raw = str(raw_e.get("type", "OTHER")).strip().upper()
        etype = type_raw if type_raw in _VALID_TYPES else "OTHER"
        desc = str(raw_e.get("description", "") or "")
        entities.append(ExtractedEntity(name=name, type=etype, description=desc))
        seen_names.add(name)

    relations: list[ExtractedRelation] = []
    for raw_r in obj.get("relations") or []:
        if not isinstance(raw_r, dict):
            continue
        src = str(raw_r.get("source", "")).strip()
        tgt = str(raw_r.get("target", "")).strip()
        if not src or not tgt or src not in seen_names or tgt not in seen_names:
            continue
        kind = str(raw_r.get("kind", "related")).strip() or "related"
        desc = str(raw_r.get("description", "") or "")
        relations.append(
            ExtractedRelation(source=src, target=tgt, kind=kind, description=desc)
        )

    return ChunkExtraction(entities=entities, relations=relations)


async def extract_entities_and_relations(chunk_text: str) -> ChunkExtraction:
    if not chunk_text or not chunk_text.strip():
        return _EMPTY
    snippet = chunk_text[:_MAX_CHARS_PER_CHUNK]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Passage:\n\n{snippet}"},
    ]
    try:
        raw = await chat_once(messages, temperature=0.0, max_tokens=1024, timeout=30.0)
    except Exception as e:
        log.warning("graph_extract_llm_fail", error=str(e))
        return _EMPTY
    return _parse(raw)
