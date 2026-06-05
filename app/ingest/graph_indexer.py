"""Graph indexer — Microsoft GraphRAG → Neo4j import.

Phase 3i: stub. Records a chunk-level :Chunk node + :MENTIONS edges to a
synthetic placeholder so cascade delete logic works end-to-end. Real
GraphRAG-driven entity/relationship/community extraction is a separate
sprint and will replace _extract_stub.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from neo4j import AsyncDriver

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


async def upsert_chunks_skeleton(driver: AsyncDriver, chunks: list[GraphChunk]) -> None:
    """Minimal skeleton — :Document and :Chunk nodes only. No entity extraction yet."""
    if not chunks:
        return
    async with driver.session() as session:
        # batch document upserts
        doc_rows = list({(str(c.tenant_id), str(c.document_id), c.filename) for c in chunks})
        await (
            await session.run(
                "UNWIND $rows AS r "
                "MERGE (d:Document {id: r[1]}) "
                "SET d.tenant_id = r[0], d.filename = r[2]",
                rows=doc_rows,
            )
        ).consume()
        # batch chunk upserts + (Chunk)-[:PART_OF]->(Document)
        chunk_rows = [
            {
                "id": str(c.chunk_id),
                "tenant_id": str(c.tenant_id),
                "document_id": str(c.document_id),
                "page": c.page,
                "text": c.text[:500],  # cap stored snippet
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
