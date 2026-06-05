"""Initialise Neo4j constraints/indexes for entities, documents, chunks, communities."""
from __future__ import annotations

import asyncio
import sys

from neo4j import AsyncGraphDatabase

from app.config import get_settings

CYPHER = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (n:Community) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX entity_tenant IF NOT EXISTS FOR (n:Entity) ON (n.tenant_id)",
    "CREATE INDEX document_tenant IF NOT EXISTS FOR (n:Document) ON (n.tenant_id)",
    "CREATE INDEX chunk_tenant IF NOT EXISTS FOR (n:Chunk) ON (n.tenant_id)",
    "CREATE INDEX community_tenant IF NOT EXISTS FOR (n:Community) ON (n.tenant_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
]


async def main() -> int:
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_url, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    try:
        async with driver.session() as session:
            for stmt in CYPHER:
                await (await session.run(stmt)).consume()
                print(f"[init_neo4j] {stmt}")
        return 0
    finally:
        await driver.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
