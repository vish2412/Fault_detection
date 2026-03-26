"""
database/db.py
==============
Async SQLAlchemy engine, session factory, and database initialisation.
"""

import logging
import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./network_monitor.db",
)

_is_sqlite = DATABASE_URL.startswith("sqlite")

_engine_kwargs: dict = {
    "echo": os.getenv("DB_ECHO", "false").lower() == "true",
}

if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    _engine_kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "10"))
    _engine_kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    _engine_kwargs["pool_pre_ping"] = True

engine: AsyncEngine = create_async_engine(DATABASE_URL, **_engine_kwargs)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Create all tables and indexes that do not yet exist.
    Uses checkfirst=True so it is safe to call on every startup,
    even when the database file already exists from a previous run.
    This prevents the 'index already exists' OperationalError on restart.
    """
    from database import models  # noqa: F401

    logger.info("Initialising database: %s", DATABASE_URL)

    async with engine.begin() as conn:
        # ----------------------------------------------------------------
        # FIX: pass checkfirst=True via a lambda so SQLAlchemy emits
        # CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS for every
        # object, preventing OperationalError on repeated startups.
        # ----------------------------------------------------------------
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True)
        )

    if _is_sqlite:
        await _enable_sqlite_wal()

    logger.info("Database tables ready.")


async def _enable_sqlite_wal() -> None:
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        await session.execute(text("PRAGMA journal_mode=WAL;"))
        await session.execute(text("PRAGMA synchronous=NORMAL;"))
        await session.execute(text("PRAGMA foreign_keys=ON;"))
        await session.commit()
    logger.debug("SQLite WAL mode enabled.")


async def close_db() -> None:
    await engine.dispose()
    logger.info("Database engine disposed.")