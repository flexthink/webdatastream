"""Checkpointable PyTorch WebDataset streams."""

from .stream import WebDataStream

__all__ = [
    "WebDataStream",
    "SQLiteCache",
    "CacheWriterError",
    "SQLiteCachedDynamicItem",
    "ParallelCachedDataset",
]


def __getattr__(name):
    # Keep SpeechBrain and TensorDict optional for ordinary shard streaming.
    if name in ("SQLiteCache", "CacheWriterError"):
        from . import cache

        return getattr(cache, name)
    if name == "SQLiteCachedDynamicItem":
        from .pipeline import SQLiteCachedDynamicItem

        return SQLiteCachedDynamicItem
    if name == "ParallelCachedDataset":
        from .dataset import ParallelCachedDataset

        return ParallelCachedDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
