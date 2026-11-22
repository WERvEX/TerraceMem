"""Async psycopg v3 connection pool lifecycle helper."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from terracemem.config import Settings


class DatabasePool:
    """Thin wrapper over psycopg AsyncConnectionPool with explicit lifecycle."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: AsyncConnectionPool[AsyncConnection[tuple[object, ...]]] | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = AsyncConnectionPool(
            conninfo=self._settings.database_url,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            open=False,
        )
        await self._pool.open()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection[tuple[object, ...]]]:
        if self._pool is None:
            raise RuntimeError(
                "DatabasePool is not open. Call open() before acquiring connections."
            )
        async with self._pool.connection() as conn:
            yield conn

    async def ping(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self.connection() as conn, conn.cursor() as cur:
                await cur.execute("SELECT 1;")
                row = await cur.fetchone()
                return row is not None and row[0] == 1
        except Exception:
            return False
