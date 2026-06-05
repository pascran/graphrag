"""Initialise Qdrant collection with named dense + sparse vectors."""
from __future__ import annotations

import asyncio
import sys

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from app.config import get_settings


async def main() -> int:
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
        if settings.qdrant_collection in existing:
            print(f"[init_qdrant] collection '{settings.qdrant_collection}' already exists")
            return 0

        await client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False)),
            },
        )
        await client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="tenant_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="document_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="doc_type",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        print(f"[init_qdrant] created collection '{settings.qdrant_collection}'")
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
