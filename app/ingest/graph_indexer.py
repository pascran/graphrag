"""Graph indexer — Microsoft GraphRAG → Neo4j import.

Two upsert modes:

- upsert_chunks_skeleton: :Document + :Chunk + [:PART_OF] only. Used
  when entity extraction is disabled or every chunk has no extractable graph.
- upsert_chunks_with_graph: full GraphRAG — also writes :Entity nodes,
  [:MENTIONS] edges from chunks, and [:RELATES_TO] edges between entities.

delete_document_graph cascades :Chunk + :Document removal. It does NOT
touch :Entity nodes — entities can be referenced by other tenants'
chunks; orphan-cleanup is a separate maintenance job.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from neo4j import AsyncDriver

from app.ingest.graph_extract import ChunkExtraction
from app.utils.logging import get_logger

log = get_logger("app.ingest.graph_indexer")


@dataclass(frozen=True)
class GraphChunk:
    chunk_id: uuid.UUID
    tenant_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int
    text: str


@dataclass(frozen=True)
class ChunkWithGraph:
    """A chunk plus the entities/relations the LLM extracted from it."""
    chunk: GraphChunk
    extraction: ChunkExtraction


async def upsert_chunks_skeleton(driver: AsyncDriver, chunks: list[GraphChunk]) -> None:
    """Minimal skeleton — :Document and :Chunk nodes only. No entity extraction."""
    if not chunks:
        return
    async with driver.session() as session:
        doc_rows = list({(str(c.tenant_id), str(c.document_id), c.filename) for c in chunks})
        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (d:Document {id: r[1]}) "
                "SET d.tenant_id = r[0], d.filename = r[2]",
                rows=doc_rows,
            )
        ).consume()
        chunk_rows = [
            {
                "id": str(c.chunk_id),
                "tenant_id": str(c.tenant_id),
                "document_id": str(c.document_id),
                "page": c.page,
                "text": c.text[:500],
            }
            for c in chunks
        ]
        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (c:Chunk {id: r.id}) "
                "SET c.tenant_id = r.tenant_id, c.page = r.page, c.text = r.text "
                "WITH c, r "
                "MATCH (d:Document {id: r.document_id}) "
                "MERGE (c)-[:PART_OF]->(d)",
                rows=chunk_rows,
            )
        ).consume()
    log.info("neo4j_chunk_upsert", count=len(chunks))


async def upsert_chunks_with_graph(
    driver: AsyncDriver, items: list[ChunkWithGraph]
) -> None:
    """Skeleton + GraphRAG entity/relation upsert in a single session.

    Entity dedup key: (tenant_id, name, type). Two tenants extracting
    "Acme/ORG" each get their own :Entity to keep the graph tenant-scoped.
    """
    if not items:
        return

    base_chunks = [it.chunk for it in items]

    entity_rows: dict[tuple[str, str, str], dict] = {}
    mention_rows: list[dict] = []
    relation_rows: list[dict] = []

    for it in items:
        c = it.chunk
        tenant = str(c.tenant_id)
        names_in_chunk: dict[str, str] = {e.name: e.type for e in it.extraction.entities}
        for e in it.extraction.entities:
            key = (tenant, e.name, e.type)
            if key not in entity_rows:
                entity_rows[key] = {
                    "tenant_id": tenant,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                }
            mention_rows.append(
                {
                    "chunk_id": str(c.chunk_id),
                    "tenant_id": tenant,
                    "name": e.name,
                    "type": e.type,
                }
            )
        for r in it.extraction.relations:
            if r.source not in names_in_chunk or r.target not in names_in_chunk:
                continue
            relation_rows.append(
                {
                    "tenant_id": tenant,
                    "source_name": r.source,
                    "source_type": names_in_chunk[r.source],
                    "target_name": r.target,
                    "target_type": names_in_chunk[r.target],
                    "kind": r.kind,
                    "description": r.description,
                }
            )

    async with driver.session() as session:
        doc_rows = list(
            {(str(c.tenant_id), str(c.document_id), c.filename) for c in base_chunks}
        )
        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (d:Document {id: r[1]}) "
                "SET d.tenant_id = r[0], d.filename = r[2]",
                rows=doc_rows,
            )
        ).consume()

        chunk_rows = [
            {
                "id": str(c.chunk_id),
                "tenant_id": str(c.tenant_id),
                "document_id": str(c.document_id),
                "page": c.page,
                "text": c.text[:500],
            }
            for c in base_chunks
        ]
        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (c:Chunk {id: r.id}) "
                "SET c.tenant_id = r.tenant_id, c.page = r.page, c.text = r.text "
                "WITH c, r "
                "MATCH (d:Document {id: r.document_id}) "
                "MERGE (c)-[:PART_OF]->(d)",
                rows=chunk_rows,
            )
        ).consume()

        if not entity_rows:
            log.info(
                "neo4j_graph_upsert",
                chunks=len(base_chunks), entities=0, mentions=0, relations=0,
            )
            return

        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (e:Entity {tenant_id: r.tenant_id, name: r.name, type: r.type}) "
                "SET e.description = coalesce(e.description, r.description)",
                rows=list(entity_rows.values()),
            )
        ).consume()

        await (
            await session.run(
                "UNWIND $rows AS r "
                "MATCH (c:Chunk {id: r.chunk_id}) "
                "MATCH (e:Entity {tenant_id: r.tenant_id, name: r.name, type: r.type}) "
                "MERGE (c)-[:MENTIONS]->(e)",
                rows=mention_rows,
            )
        ).consume()

        if relation_rows:
            await (
                await session.run(
                    "UNWIND $rows AS r "
                    "MATCH (s:Entity {tenant_id: r.tenant_id, name: r.source_name, type: r.source_type}) "
                    "MATCH (t:Entity {tenant_id: r.tenant_id, name: r.target_name, type: r.target_type}) "
                    "MERGE (s)-[rel:RELATES_TO {kind: r.kind}]->(t) "
                    "SET rel.description = coalesce(rel.description, r.description)",
                    rows=relation_rows,
                )
            ).consume()

    log.info(
        "neo4j_graph_upsert",
        chunks=len(base_chunks),
        entities=len(entity_rows),
        mentions=len(mention_rows),
        relations=len(relation_rows),
    )


async def delete_document_graph(
    driver: AsyncDriver, tenant_id: uuid.UUID, document_id: uuid.UUID
) -> None:
    async with driver.session() as session:
        await (
            await session.run(
                "MATCH (d:Document {id: $doc_id, tenant_id: $tenant_id}) "
                "OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d) "
                "DETACH DELETE c, d",
                doc_id=str(document_id),
                tenant_id=str(tenant_id),
            )
        ).consume()
