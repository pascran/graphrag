"""Bootstrap PostgreSQL: run alembic migrations, seed first tenant + API key.

Idempotent: re-running it does NOT re-issue keys (returns existing if env var set
already matches a row). Prints the freshly issued key ONCE to stdout, then writes
it back into .env so subsequent containers can read INITIAL_API_KEY.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.core.auth import generate_api_key, hash_api_key
from app.models.orm import ApiKey, Tenant

DEFAULT_TENANT_NAME = "default"
DEFAULT_KEY_NAME = "initial-key"


def run_migrations() -> None:
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", get_settings().postgres_dsn_sync)
    command.upgrade(cfg, "head")


async def seed_tenant_and_key(session: AsyncSession) -> tuple[bool, str | None]:
    """Returns (issued, plain_key). issued=False if a tenant+key already exists."""
    existing = (await session.execute(select(Tenant).limit(1))).scalar_one_or_none()
    if existing is not None:
        any_key = (
            await session.execute(select(ApiKey).where(ApiKey.tenant_id == existing.id).limit(1))
        ).scalar_one_or_none()
        if any_key is not None:
            return False, None

    tenant = existing or Tenant(name=DEFAULT_TENANT_NAME)
    if existing is None:
        session.add(tenant)
        await session.flush()

    plain = generate_api_key()
    api_key = ApiKey(
        tenant_id=tenant.id,
        key_hash=hash_api_key(plain),
        name=DEFAULT_KEY_NAME,
        is_active=True,
    )
    session.add(api_key)
    await session.commit()
    return True, plain


def inject_env_initial_key(plain_key: str, env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        env_path.write_text(f"INITIAL_API_KEY={plain_key}\n", encoding="utf-8")
        return

    txt = env_path.read_text(encoding="utf-8")
    if re.search(r"^INITIAL_API_KEY=", txt, flags=re.MULTILINE):
        new = re.sub(r"^INITIAL_API_KEY=.*$", f"INITIAL_API_KEY={plain_key}", txt, flags=re.MULTILINE)
    else:
        new = txt.rstrip() + f"\nINITIAL_API_KEY={plain_key}\n"
    env_path.write_text(new, encoding="utf-8")


async def main() -> int:
    print("[init_db] running alembic upgrade head...", file=sys.stderr)
    run_migrations()

    settings = get_settings()
    engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as session:
        issued, plain = await seed_tenant_and_key(session)

    if issued and plain:
        env_target = Path(os.getenv("INIT_DB_ENV_PATH", ".env"))
        inject_env_initial_key(plain, env_target)
        print("=" * 64)
        print("INITIAL API KEY (shown ONCE — also written to .env):")
        print(plain)
        print("=" * 64)
    else:
        print("[init_db] tenant/api_key already present — no new key issued", file=sys.stderr)

    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
