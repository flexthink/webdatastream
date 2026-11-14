"""Persistent tensor caching with one producer and blocking SQLite readers.

Connections are opened lazily in the process that uses them. SQL statements
are fixed and parameterized; sqlite3 caches their prepared statements.
"""

from __future__ import annotations

import io
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch


_READ_ITEM = (
    "SELECT items.payload, control.version, control.status, control.error, control.writer_pid "
    "FROM control LEFT JOIN items ON items.id=? WHERE control.singleton=1"
)


class CacheWriterError(RuntimeError):
    """The producer failed or exited before publishing a requested item."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _encode(value: Any) -> Any:
    """Convert containers to torch's safe serialization vocabulary."""
    if isinstance(value, torch.Tensor):
        # A slice otherwise serializes its entire backing batch storage.
        return ("tensor", value.detach().cpu().clone())
    if isinstance(value, dict):
        return ("dict", [(key, _encode(item)) for key, item in value.items()])
    if (
        isinstance(value, tuple)
        and type(value).__name__ == "PaddedData"
        and getattr(value, "_fields", None) == ("data", "lengths")
    ):
        return ("paddeddata", [_encode(item) for item in value])
    if isinstance(value, (tuple, list)):
        return (
            "tuple" if isinstance(value, tuple) else "list",
            [_encode(item) for item in value],
        )
    if value is None or type(value) in (str, bytes, bool, int, float):
        return ("scalar", value)
    # TensorDict is optional for users of the storage layer alone.
    try:
        from tensordict import TensorDictBase
    except ImportError:
        raise TypeError(f"Unsupported cached value: {type(value).__name__}") from None

    if isinstance(value, TensorDictBase):
        return (
            "tensordict",
            list(value.batch_size),
            {key: _encode(item) for key, item in value.items()},
        )
    raise TypeError(f"Unsupported cached value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    kind, payload, *rest = value
    if kind in ("tensor", "scalar"):
        return payload
    if kind == "dict":
        return {key: _decode(item) for key, item in payload}
    if kind in ("tuple", "list"):
        items = [_decode(item) for item in payload]
        return tuple(items) if kind == "tuple" else items
    if kind == "paddeddata":
        from speechbrain.dataio.batch import PaddedData

        return PaddedData(*[_decode(item) for item in payload])
    if kind == "tensordict":
        from tensordict import TensorDict

        return TensorDict(
            {key: _decode(item) for key, item in rest[0].items()}, batch_size=payload
        )
    raise ValueError(f"Unknown cached value type: {kind!r}")


def _serialize(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(_encode(value), buffer)
    return buffer.getvalue()


def _deserialize(payload: bytes) -> Any:
    return _decode(
        torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    )


class SQLiteCache:
    """An append-only cache of string IDs and complete sample BLOBs.

    ``mode='write'`` claims the sole producer role on first use. Readers may
    start before the writer and block until their requested ID is committed.
    Use ``with SQLiteCache(..., mode='write')`` to publish completion or a
    producer error automatically. A completed cache can be resumed by a new
    writer; existing IDs are retained. Use a fresh path to invalidate a cache.

    WAL permits readers while the writer commits arbitrarily far ahead. Each
    connection keeps a bounded SQLite page cache and prepared-statement cache;
    neither the dataset nor a Python ID index is accumulated in memory.
    The database must be on a local filesystem shared by the processes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        mode: str = "read",
        timeout: float | None = None,
        poll_interval: float = 0.05,
    ):
        if mode not in ("read", "write"):
            raise ValueError("mode must be 'read' or 'write'")
        if poll_interval <= 0 or (timeout is not None and timeout <= 0):
            raise ValueError("timeout and poll_interval must be positive")
        self.path = Path(path).expanduser().absolute()
        self.mode = mode
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._connection: sqlite3.Connection | None = None
        self._pid: int | None = None
        self._token: str | None = None
        self._terminal = False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.update(_connection=None, _pid=None, _token=None)
        return state

    def _connect(self) -> sqlite3.Connection | None:
        if self._pid is not None and self._pid != os.getpid():
            raise RuntimeError(
                "Use spawn, or construct a cache separately in each process"
            )
        if self._connection is not None:
            return self._connection
        if self.mode == "read":
            if not self.path.exists():
                return None
            db = sqlite3.connect(
                self.path.as_uri() + "?mode=ro",
                uri=True,
                isolation_level=None,
                cached_statements=128,
            )
            # The writer may have created the file but not its schema yet.
            if not db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='control'"
            ).fetchone():
                db.close()
                return None
        else:
            if self._terminal:
                raise RuntimeError("This writer has already finished")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.path, isolation_level=None, cached_statements=128)
            try:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, payload BLOB NOT NULL) WITHOUT ROWID"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS control (singleton INTEGER PRIMARY KEY CHECK(singleton=1), version INTEGER NOT NULL, status TEXT NOT NULL, error TEXT, writer_pid INTEGER, writer_token TEXT)"
                )
                row = db.execute(
                    "SELECT version, writer_pid FROM control WHERE singleton=1"
                ).fetchone()
                if row and row[0] != 1:
                    raise ValueError("Unsupported cache format version")
                if row and row[1] is not None and _alive(row[1]):
                    raise RuntimeError(
                        f"Cache already has a writer in process {row[1]}"
                    )
                token = uuid.uuid4().hex
                db.execute(
                    "INSERT OR REPLACE INTO control VALUES (1, 1, 'writing', NULL, ?, ?)",
                    (os.getpid(), token),
                )
                db.commit()
            except BaseException:
                db.close()
                raise
            self._token = token
        db.execute("PRAGMA cache_size=-8192")
        self._connection, self._pid = db, os.getpid()
        return db

    @staticmethod
    def _id(uid: str) -> None:
        if not isinstance(uid, str):
            raise TypeError("Cache IDs must be strings")

    def contains(self, uid: str) -> bool:
        """Check readiness without blocking or deserializing the sample."""
        self._id(uid)
        db = self._connect()
        return (
            db is not None
            and db.execute("SELECT 1 FROM items WHERE id=?", (uid,)).fetchone()
            is not None
        )

    def put_many(self, items: Iterable[tuple[str, Any]]) -> None:
        """Publish a bounded batch atomically, preserving previously cached IDs."""
        if self.mode != "write" or self._terminal:
            raise RuntimeError("Only an active writer can publish items")
        # Serialize before acquiring SQLite's write lock. Memory is bounded by
        # this submitted batch, independently of how far the producer is ahead.
        rows = []
        for uid, value in items:
            self._id(uid)
            rows.append((uid, sqlite3.Binary(_serialize(value))))
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            db.executemany(
                "INSERT OR IGNORE INTO items (id, payload) VALUES (?, ?)", rows
            )
            db.commit()
        except BaseException:
            db.rollback()
            raise

    def put(self, uid: str, value: Any) -> None:
        """Publish one complete sample."""
        self.put_many([(uid, value)])

    def get(self, uid: str) -> Any:
        """Wait for a sample; raise on completion, producer failure, or timeout."""
        self._id(uid)
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            db = self._connect()
            if db is not None:
                # One statement gives the item and producer state from the
                # same snapshot, avoiding a finish-versus-last-write race.
                row = db.execute(_READ_ITEM, (uid,)).fetchone()
                if row is not None:
                    payload, version, status, error, pid = row
                    if version != 1:
                        raise ValueError("Unsupported cache format version")
                    if payload is not None:
                        return _deserialize(payload)
                    if status == "complete":
                        raise KeyError(f"ID {uid!r} is absent from the completed cache")
                    if status == "failed":
                        raise CacheWriterError(error or "Cache writer failed")
                    if pid is not None and not _alive(pid):
                        # The producer may have committed and exited between
                        # our snapshot and the liveness check. Recheck before
                        # declaring its last requested sample unavailable.
                        latest = db.execute(_READ_ITEM, (uid,)).fetchone()
                        if latest != row:
                            continue
                        raise CacheWriterError(
                            f"Cache writer process {pid} exited before publishing {uid!r}"
                        )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for cached ID {uid!r}")
            delay = (
                self.poll_interval
                if deadline is None
                else min(self.poll_interval, max(0, deadline - time.monotonic()))
            )
            time.sleep(delay)

    def _end(self, status: str, error: str | None = None) -> None:
        if self.mode != "write":
            raise RuntimeError("Only the writer can set completion state")
        if self._terminal:
            return
        db = self._connect()
        db.execute(
            "UPDATE control SET status=?, error=?, writer_pid=NULL, writer_token=NULL WHERE singleton=1 AND writer_token=?",
            (status, error, self._token),
        )
        self._terminal = True

    def finish(self) -> None:
        """Publish successful completion; missing IDs will now raise KeyError."""
        self._end("complete")

    def fail(self, error: str | BaseException) -> None:
        """Publish a producer failure so waiting readers raise CacheWriterError."""
        self._end("failed", str(error))

    def close(self) -> None:
        """Release this process's connection, failing an unfinished writer."""
        if self._connection is not None:
            if self.mode == "write" and not self._terminal:
                self.fail("Cache writer closed before finish()")
            self._connection.close()
            self._connection = None
            self._pid = None

    def __enter__(self) -> SQLiteCache:
        self._connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self.mode == "write":
                self.finish() if exc is None else self.fail(exc)
        finally:
            self.close()
