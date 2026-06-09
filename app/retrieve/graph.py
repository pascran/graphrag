"""GraphRAG local search over the Neo4j knowledge graph built in Phase 3i.

Strategy (local search):
  1. Lower-case the question.
  2. Seed :Entity nodes for the tenant. Two matching modes:
     - default: substring match (`question CONTAINS toLower(e.name)`), which
       handles multi-word names ("Bob Park") without tokenisation but produces
       false positives on short Korean entities ("원장" inside "부원장").
     - token-boundary (Fix A, `graph_seed_use_token_boundary`): match the
       entity name against the question's token set (`toLower(name) IN $tokens`)
       with a minimum length guard. See `app.retrieve.tokenizer`.
  3. Expand seed entities by one [:RELATES_TO] hop in either direction to
     pick up neighbours the question implicitly references.
  4. Find :Chunk nodes that :MENTIONS any entity in the expanded set,
     score by the number of matched entities per chunk, ORDER BY score DESC.

Tenant-scoped on :Entity and :Document. :Entity dedup is already
(tenant_id, name, type) from upsert, so the MATCH is safe across tenants.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from neo4j import AsyncDriver

from app.config import get_settings
from app.retrieve.tokenizer import question_tokens
from app.utils.logging import get_logger

log = get_logger("app.retrieve.graph")


@dataclass(frozen=True)
class GraphHit:
    filename: str
    page: int
    text: str
    score: float
    document_id: str
    chunk_index: int


_CYPHER = """
MATCH (seed:Entity {tenant_id: $tenant_id})
WHERE $question CONTAINS toLower(seed.name)
WITH collect(DISTINCT seed) AS seeds, $tenant_id AS tid
UNWIND seeds AS s
OPTIONAL MATCH (s)-[:RELATES_TO*0..1]-(n:Entity {tenant_id: tid})
WITH collect(DISTINCT s) + collect(DISTINCT n) AS ents
UNWIND ents AS e
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (c)-[:PART_OF]->(d:Document {tenant_id: $tenant_id})
WITH c, d, count(DISTINCT e) AS score
RETURN
  c.id           AS chunk_id,
  d.filename     AS filename,
  c.page         AS page,
  c.text         AS text,
  d.id           AS document_id,
  c.chunk_index  AS chunk_index,
  score          AS score
ORDER BY score DESC
LIMIT $top_k
"""


# Fix A: exact token-set seeding. Entities whose lower-cased name is one of the
# question tokens (and long enough to be meaningful) seed the local search.
# Replaces the substring CONTAINS match that produced false positives on short
# Korean entities (e.g. "원장" inside "부원장").
_CYPHER_TOKENS = """
MATCH (seed:Entity {tenant_id: $tenant_id})
WHERE toLower(seed.name) IN $tokens
  AND size(seed.name) >= $min_len
WITH collect(DISTINCT seed) AS seeds, $tenant_id AS tid
UNWIND seeds AS s
OPTIONAL MATCH (s)-[:RELATES_TO*0..1]-(n:Entity {tenant_id: tid})
WITH collect(DISTINCT s) + collect(DISTINCT n) AS ents
UNWIND ents AS e
MATCH (c:Chunk)-[:MENTIONS]->(e)
MATCH (c)-[:PART_OF]->(d:Document {tenant_id: $tenant_id})
WITH c, d, count(DISTINCT e) AS score
RETURN
  c.id           AS chunk_id,
  d.filename     AS filename,
  c.page         AS page,
  c.text         AS text,
  d.id           AS document_id,
  c.chunk_index  AS chunk_index,
  score          AS score
ORDER BY score DESC
LIMIT $top_k
"""


async def graph_search(
    driver: AsyncDriver,
    *,
    tenant_id: uuid.UUID,
    question: str,
    top_k: int = 5,
) -> list[GraphHit]:
    q = (question or "").strip().lower()
    if not q:
        return []

    limit = max(1, int(top_k))
    settings = get_settings()
    use_tokens = settings.graph_seed_use_token_boundary
    min_len = settings.graph_seed_min_entity_len

    tokens: list[str] = []
    if use_tokens:
        tokens = question_tokens(question, min_len=min_len)
        if not tokens:  # nothing matchable — skip the driver entirely
            return []

    out: list[GraphHit] = []
    async with driver.session() as session:
        if use_tokens:
            result = await session.run(
                _CYPHER_TOKENS,
                tenant_id=str(tenant_id),
                tokens=tokens,
                min_len=min_len,
                top_k=limit,
            )
        else:
            result = await session.run(
                _CYPHER,
                tenant_id=str(tenant_id),
                question=q,
                top_k=limit,
            )
        async for row in result:
            out.append(
                GraphHit(
                    filename=row.get("filename") or "?",
                    page=int(row.get("page") or 0),
                    text=row.get("text") or "",
                    score=float(row.get("score") or 0.0),
                    document_id=str(row.get("document_id") or ""),
                    chunk_index=int(row.get("chunk_index") or 0),
                )
            )
    log.info(
        "graph_search",
        question_chars=len(q), hits=len(out), top_k=limit,
    )
    return out
