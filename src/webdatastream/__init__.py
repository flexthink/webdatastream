"""Checkpointable PyTorch WebDataset streams."""

from .stream import WebDataStream

__all__ = [
    "WebDataStream",
    "SQLiteCache",
    "CacheWriterError",
    "SQLiteCachedDynamicItem",
    "ParallelCachedDataset",
    "BatchedPipelineDataset",
]


def __getattr__(name):
    # Keep SpeechBrain and TensorDict optional for ordinary shard streaming.
    if name in ("SQLiteCache", "CacheWriterError"):
        from . import cache

        return getattr(cache, name)
    if name in ("SQLiteCachedDynamicItem", "BatchedPipelineDataset"):
        from . import pipeline

        return getattr(pipeline, name)
    if name == "ParallelCachedDataset":
        from .dataset import ParallelCachedDataset

        return ParallelCachedDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
