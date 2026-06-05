"""Orphan :Entity cleanup Celery task.

An :Entity is "orphan" iff no :Chunk has a :MENTIONS edge to it. Such
nodes accumulate after :Document / :Chunk hard-deletes (see
:func:`app.ingest.graph_indexer.delete_document_graph`, which intentionally
leaves :Entity nodes behind because they can be referenced by other
chunks). This task scans for orphans and (optionally) DETACH-deletes
them. ``DETACH`` removes any dangling ``:RELATES_TO`` edges in the same
operation, so isolated entity-only components are cleaned up as well.

Entity dedup key is ``(tenant_id, name, type)`` — the :Entity node DOES
carry a ``tenant_id`` property (see graph_indexer.upsert_chunks_with_graph
MERGE clause), so we can scope cleanup to a single tenant when desired.

Defaults to ``dry_run=True``: the task NEVER deletes unless explicitly
asked. The Celery beat entry registered in :mod:`app.workers.celery_app`
also passes ``dry_run=True`` so the nightly run is observe-only by default.
"""
from __future__ import annotations

from typing import Any

from celery.exceptions import Retry
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError

from app.db.neo4j import get_driver
from app.utils.logging import configure_logging, get_logger
from app.workers.celery_app import celery_app

log = get_logger("app.workers.cleanup")

# Exceptions treated as transient → trigger Celery retry with backoff.
_TRANSIENT_NEO4J = (ServiceUnavailable, TransientError)


async def _scan_orphans(
    *,
    tenant_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` orphan :Entity rows ({name, tenant_id, id})."""
    if tenant_id is None:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE NOT (e)<-[:MENTIONS]-(:Chunk) "
            "WITH e LIMIT $limit "
            "RETURN e.name AS name, e.tenant_id AS tenant_id, elementId(e) AS id"
        )
        params: dict[str, Any] = {"limit": int(limit)}
    else:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.tenant_id = $tenant_id "
            "  AND NOT (e)<-[:MENTIONS]-(:Chunk) "
            "WITH e LIMIT $limit "
            "RETURN e.name AS name, e.tenant_id AS tenant_id, elementId(e) AS id"
        )
        params = {"limit": int(limit), "tenant_id": tenant_id}

    rows: list[dict[str, Any]] = []
    async with get_driver().session() as session:
        result = await session.run(cypher, **params)
        async for record in result:
            rows.append(
                {
                    "name": record.get("name"),
                    "tenant_id": record.get("tenant_id"),
                    "id": record.get("id"),
                }
            )
    return rows


async def _delete_orphans(
    *,
    tenant_id: str | None,
    limit: int,
) -> int:
    """DETACH DELETE up to ``limit`` orphan :Entity rows. Returns deleted count."""
    if tenant_id is None:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE NOT (e)<-[:MENTIONS]-(:Chunk) "
            "WITH e LIMIT $limit "
            "DETACH DELETE e "
            "RETURN count(*) AS deleted"
        )
        params: dict[str, Any] = {"limit": int(limit)}
    else:
        cypher = (
            "MATCH (e:Entity) "
            "WHERE e.tenant_id = $tenant_id "
            "  AND NOT (e)<-[:MENTIONS]-(:Chunk) "
            "WITH e LIMIT $limit "
            "DETACH DELETE e "
            "RETURN count(*) AS deleted"
        )
        params = {"limit": int(limit), "tenant_id": tenant_id}

    async with get_driver().session() as session:
        result = await session.run(cypher, **params)
        record = await result.single()
        deleted = int(record["deleted"]) if record is not None else 0
        await result.consume()
    return deleted


async def _run(
    *,
    dry_run: bool,
    tenant_id: str | None,
    limit: int,
) -> dict[str, Any]:
    """Async core. Scans, optionally deletes, and returns a summary dict."""
    configure_logging()
    rows = await _scan_orphans(tenant_id=tenant_id, limit=limit)
    sample = rows[:20]

    if dry_run:
        summary = {
            "dry_run": True,
            "tenant_id": tenant_id,
            "limit": limit,
            "scanned": len(rows),
            "would_delete": len(rows),
            "sample": sample,
        }
        log.info(
            "orphan_cleanup",
            action="orphan_cleanup",
            dry_run=True,
            tenant_id=tenant_id,
            limit=limit,
            scanned=len(rows),
            would_delete=len(rows),
        )
        return summary

    deleted = await _delete_orphans(tenant_id=tenant_id, limit=limit)
    summary = {
        "dry_run": False,
        "tenant_id": tenant_id,
        "limit": limit,
        "scanned": len(rows),
        "deleted": deleted,
        "sample": sample,
    }
    log.info(
        "orphan_cleanup",
        action="orphan_cleanup",
        dry_run=False,
        tenant_id=tenant_id,
        limit=limit,
        scanned=len(rows),
        deleted=deleted,
    )
    return summary


@celery_app.task(
    name="cleanup.orphan_entities",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def cleanup_orphan_entities(
    self,
    dry_run: bool = True,
    tenant_id: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Celery entry point. Runs :func:`_run` and retries on transient Neo4j errors.

    Parameters mirror the spec exactly:
      * ``dry_run`` (default ``True``) — when True, never executes DETACH DELETE.
      * ``tenant_id`` — optional tenant scoping. When ``None``, all tenants.
      * ``limit`` — Cypher LIMIT applied to the scan/delete to bound write locks.

    Returns a dict with keys ``dry_run``, ``tenant_id``, ``limit``, ``scanned``,
    ``sample`` (first 20 rows) and exactly one of ``would_delete`` (dry_run=True)
    or ``deleted`` (dry_run=False).
    """
    import asyncio

    try:
        return asyncio.run(
            _run(dry_run=dry_run, tenant_id=tenant_id, limit=limit)
        )
    except Retry:
        # Bubble up Retry without wrapping — Celery handles it.
        raise
    except _TRANSIENT_NEO4J as exc:
        log.warning(
            "orphan_cleanup_transient",
            action="orphan_cleanup",
            dry_run=dry_run,
            tenant_id=tenant_id,
            error=str(exc),
            attempt=self.request.retries + 1,
        )
        # Exponential backoff: 30s, 60s, 120s
        countdown = 30 * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=countdown)
    except Neo4jError as exc:
        # Non-transient driver error — log and re-raise so Celery marks failed.
        log.error(
            "orphan_cleanup_failed",
            action="orphan_cleanup",
            dry_run=dry_run,
            tenant_id=tenant_id,
            error=str(exc),
        )
        raise
    except Exception as exc:  # noqa: BLE001 — last-resort logging
        log.exception(
            "orphan_cleanup_unhandled",
            action="orphan_cleanup",
            dry_run=dry_run,
            tenant_id=tenant_id,
            error=str(exc),
        )
        raise
