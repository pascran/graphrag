#!/usr/bin/env python
"""Orphan :Entity cleanup CLI runner.

Calls :func:`app.workers.cleanup._run` *synchronously* (via its
underlying async core) so operators can preview / execute a cleanup
without going through Celery.

Default mode is dry-run. Pass ``--apply`` to actually DETACH DELETE.

Examples
--------

    # 1. Preview orphans across all tenants (default limit=1000)
    docker compose exec app python scripts/cleanup_orphans.py

    # 2. Preview for a single tenant
    docker compose exec app python scripts/cleanup_orphans.py \\
        --tenant 8cf7e9b0-...

    # 3. Actually delete (max 500 nodes this run)
    docker compose exec app python scripts/cleanup_orphans.py --apply --limit 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow `python scripts/cleanup_orphans.py ...` without PYTHONPATH=/app.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.workers.cleanup import _run  # noqa: E402 — sys.path tweak above


def _print_summary(summary: dict) -> None:
    dry = summary.get("dry_run", True)
    scanned = summary.get("scanned", 0)
    action_count = summary.get("would_delete" if dry else "deleted", 0)
    tenant = summary.get("tenant_id") or "(all tenants)"
    limit = summary.get("limit")

    print("=" * 72)
    print(f"  orphan :Entity cleanup — {'DRY RUN' if dry else 'APPLY'}")
    print("=" * 72)
    print(f"  tenant_id         : {tenant}")
    print(f"  limit             : {limit}")
    print(f"  scanned           : {scanned}")
    label = "would_delete" if dry else "deleted"
    print(f"  {label:<18}: {action_count}")
    print("-" * 72)
    sample = summary.get("sample") or []
    if sample:
        print(f"  sample (first {len(sample)}):")
        print(f"    {'name':<40} {'tenant_id':<38} id")
        print(f"    {'-' * 40} {'-' * 38} {'-' * 12}")
        for row in sample:
            name = (row.get("name") or "")[:40]
            tid = (row.get("tenant_id") or "")[:38]
            nid = row.get("id") or ""
            print(f"    {name:<40} {tid:<38} {nid}")
    else:
        print("  sample            : (no orphans found)")
    print("=" * 72)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cleanup_orphans",
        description=__doc__.splitlines()[0],
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually DETACH DELETE the orphan entities. Default is dry-run.",
    )
    p.add_argument(
        "--tenant",
        dest="tenant_id",
        default=None,
        help="Optional tenant UUID to scope the cleanup. Default: all tenants.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Max orphan rows to scan/delete per invocation. Default 1000.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw summary as JSON to stdout (in addition to the table).",
    )
    return p


def main() -> int:
    args = _build_parser().parse_args()
    dry_run = not args.apply
    summary = asyncio.run(
        _run(dry_run=dry_run, tenant_id=args.tenant_id, limit=args.limit)
    )
    _print_summary(summary)
    if args.json:
        print(json.dumps(summary, default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
