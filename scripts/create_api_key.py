"""API Key admin CLI — issue, list, revoke keys for any tenant.

Usage (run inside the `app` container so it can resolve POSTGRES_DSN):

  # 1. List all tenants
  docker compose exec app python -m scripts.create_api_key tenants

  # 2. Issue a fresh key for an existing tenant by id (or name)
  docker compose exec app python -m scripts.create_api_key issue \\
      --tenant-id 8cf7... --name "ops-laptop"
  docker compose exec app python -m scripts.create_api_key issue \\
      --tenant "acme-corp" --name "ops-laptop"

  # 3. Issue a key AND create a new tenant in one shot
  docker compose exec app python -m scripts.create_api_key issue \\
      --new-tenant "acme-corp" --name "first-key"

  # 4. List keys for a tenant (hashes only — plain key is never stored)
  docker compose exec app python -m scripts.create_api_key list --tenant "acme-corp"

  # 5. Revoke a key (sets is_active=false; soft-delete keeps audit trail)
  docker compose exec app python -m scripts.create_api_key revoke --key-id 4d2e...

The plain key is printed exactly ONCE to stdout. Re-running cannot recover it —
mint a new key if it's lost.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

# Allow `python scripts/create_api_key.py ...` from /app without PYTHONPATH=/app.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.auth import generate_api_key, hash_api_key
from app.models.orm import ApiKey, Tenant


async def _resolve_tenant(
    session: AsyncSession, *, tenant_id: str | None, tenant_name: str | None
) -> Tenant | None:
    if tenant_id:
        try:
            tid = uuid.UUID(tenant_id)
        except ValueError:
            print(f"error: invalid tenant id: {tenant_id}", file=sys.stderr)
            return None
        return (
            await session.execute(select(Tenant).where(Tenant.id == tid))
        ).scalar_one_or_none()
    if tenant_name:
        return (
            await session.execute(select(Tenant).where(Tenant.name == tenant_name))
        ).scalar_one_or_none()
    return None


async def cmd_tenants(args: argparse.Namespace) -> int:
    engine = create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        rows = (
            await session.execute(select(Tenant).order_by(Tenant.created_at.asc()))
        ).scalars().all()
        if not rows:
            print("(no tenants)")
        else:
            print(f"{'id':<38} {'name':<30} created_at")
            print("-" * 90)
            for t in rows:
                print(f"{str(t.id):<38} {t.name:<30} {t.created_at.isoformat()}")
    await engine.dispose()
    return 0


async def cmd_issue(args: argparse.Namespace) -> int:
    engine = create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with Session() as session:
            if args.new_tenant:
                existing = (
                    await session.execute(
                        select(Tenant).where(Tenant.name == args.new_tenant)
                    )
                ).scalar_one_or_none()
                if existing:
                    print(
                        f"error: tenant '{args.new_tenant}' already exists "
                        f"(id={existing.id}). Use --tenant instead of --new-tenant.",
                        file=sys.stderr,
                    )
                    return 2
                tenant = Tenant(name=args.new_tenant)
                session.add(tenant)
                await session.flush()
            else:
                tenant = await _resolve_tenant(
                    session, tenant_id=args.tenant_id, tenant_name=args.tenant
                )
                if tenant is None:
                    print(
                        "error: tenant not found. Use 'tenants' to list, or "
                        "--new-tenant <name> to create.",
                        file=sys.stderr,
                    )
                    return 2

            plain = generate_api_key()
            key = ApiKey(
                tenant_id=tenant.id,
                key_hash=hash_api_key(plain),
                name=args.name,
                is_active=True,
            )
            session.add(key)
            await session.commit()

            print("=" * 64)
            print(f"tenant   : {tenant.name} ({tenant.id})")
            print(f"key id   : {key.id}")
            print(f"key name : {args.name or '(unnamed)'}")
            print("PLAIN KEY (shown ONCE — store it now):")
            print(plain)
            print("=" * 64)
    except IntegrityError as e:
        print(f"error: integrity violation: {e.orig}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    engine = create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        tenant = await _resolve_tenant(
            session, tenant_id=args.tenant_id, tenant_name=args.tenant
        )
        if tenant is None:
            print("error: tenant not found.", file=sys.stderr)
            return 2
        rows = (
            await session.execute(
                select(ApiKey)
                .where(ApiKey.tenant_id == tenant.id)
                .order_by(ApiKey.created_at.asc())
            )
        ).scalars().all()
        print(f"tenant: {tenant.name} ({tenant.id})")
        if not rows:
            print("(no keys)")
        else:
            print(f"{'id':<38} {'name':<24} {'active':<7} {'hash[:12]':<14} created_at")
            print("-" * 110)
            for k in rows:
                print(
                    f"{str(k.id):<38} "
                    f"{(k.name or ''):<24} "
                    f"{str(k.is_active):<7} "
                    f"{k.key_hash[:12]:<14} "
                    f"{k.created_at.isoformat()}"
                )
    await engine.dispose()
    return 0


async def cmd_revoke(args: argparse.Namespace) -> int:
    try:
        kid = uuid.UUID(args.key_id)
    except ValueError:
        print(f"error: invalid key id: {args.key_id}", file=sys.stderr)
        return 2
    engine = create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        result = await session.execute(
            update(ApiKey).where(ApiKey.id == kid).values(is_active=False)
        )
        await session.commit()
        if result.rowcount == 0:
            print("error: key not found.", file=sys.stderr)
            await engine.dispose()
            return 2
        print(f"revoked key {kid}")
    await engine.dispose()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="create_api_key", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tenants", help="list all tenants")

    p_issue = sub.add_parser("issue", help="mint a new API key")
    g = p_issue.add_mutually_exclusive_group(required=True)
    g.add_argument("--tenant-id", help="existing tenant UUID")
    g.add_argument("--tenant", help="existing tenant name")
    g.add_argument("--new-tenant", help="create a new tenant with this name and issue")
    p_issue.add_argument("--name", help="human-readable label for this key", default=None)

    p_list = sub.add_parser("list", help="list keys for a tenant (hashes only)")
    g2 = p_list.add_mutually_exclusive_group(required=True)
    g2.add_argument("--tenant-id", help="tenant UUID")
    g2.add_argument("--tenant", help="tenant name")

    p_revoke = sub.add_parser("revoke", help="deactivate an API key by id")
    p_revoke.add_argument("--key-id", required=True, help="api_keys.id UUID")

    return p


HANDLERS = {
    "tenants": cmd_tenants,
    "issue": cmd_issue,
    "list": cmd_list,
    "revoke": cmd_revoke,
}


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(HANDLERS[args.cmd](args))


if __name__ == "__main__":
    sys.exit(main())
