"""Bounded remote shard prefetch with a consumed-sample checkpoint cursor."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

import fsspec
from torch.utils.data import IterableDataset, get_worker_info


def _download(url: str, destination: Path) -> Path:
    """Copy a shard into the iterator's private cache."""
    try:
        with fsspec.open(url, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _samples(path: Path) -> Iterator[dict[str, Any]]:
    """Group consecutive tar members by WebDataset sample key."""
    key: str | None = None
    fields: dict[str, bytes] = {}
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            archive.members.clear()  # Streaming readers otherwise retain headers.
            if not member.isfile():
                continue
            stem, dot, extension = member.name.removeprefix("./").rpartition(".")
            if not dot or not extension:
                continue
            if key is not None and stem != key:
                yield {"__key__": key, **fields}
                fields = {}
            key = stem
            if extension in fields:
                raise ValueError(f"Duplicate field {extension!r} for sample {stem!r}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read tar member {member.name!r}")
            fields[extension] = source.read()
        if key is not None:
            yield {"__key__": key, **fields}


class WebDataStream(IterableDataset[dict[str, Any]]):
    """Read tar shards while downloading the next bounded window.

    Keep this dataset in the training process with ``DataLoader(num_workers=0)``
    so ``state_dict`` captures the last sample returned to training.

    Arguments
    ---------
    shards : sequence of str
        Ordered local paths or fsspec URLs.
    prefetch_shards : int
        Maximum staged shards, including the current shard.
    repeat : bool
        Cycle through all shards indefinitely.
    """

    def __init__(self, shards: Sequence[str], prefetch_shards: int = 2, repeat: bool = False) -> None:
        super().__init__()
        if not shards:
            raise ValueError("At least one shard is required")
        if prefetch_shards < 1:
            raise ValueError("prefetch_shards must be positive")
        self.shards = tuple(str(shard) for shard in shards)
        self.prefetch_shards = prefetch_shards
        self.repeat = repeat
        self._cycle = self._shard = self._sample = 0
        self._active = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield field bytes with ``__key__`` and ``__url__`` metadata.

        Yields
        ------
        dict[str, Any]
            One WebDataset sample.
        """
        if get_worker_info() is not None:
            raise RuntimeError("Use DataLoader(num_workers=0) for checkpointable iteration")
        if self._active:
            raise RuntimeError("Only one active iterator is supported")
        self._active = True
        try:
            with tempfile.TemporaryDirectory(prefix="webdatastream-") as cache:
                with ThreadPoolExecutor(max_workers=self.prefetch_shards) as executor:
                    futures: dict[int, Future[Path]] = {}
                    cycle, shard_index, sample_index = self._cycle, self._shard, self._sample
                    while True:
                        if shard_index >= len(self.shards):
                            if not self.repeat:
                                break
                            cycle += 1
                            shard_index = sample_index = 0
                        for offset in range(self.prefetch_shards):
                            index = shard_index + offset
                            if index >= len(self.shards) or index in futures:
                                continue
                            url = self.shards[index]
                            digest = hashlib.sha256(url.encode()).hexdigest()[:16]
                            target = Path(cache) / f"{index}-{digest}.tar"
                            futures[index] = executor.submit(_download, url, target)
                        path = futures[shard_index].result()
                        seen = 0
                        for index, sample in enumerate(_samples(path)):
                            seen += 1
                            if index < sample_index:
                                continue
                            sample["__url__"] = self.shards[shard_index]
                            self._cycle, self._shard, self._sample = cycle, shard_index, index + 1
                            yield sample
                        if seen == 0:
                            raise ValueError(f"Shard contains no samples: {self.shards[shard_index]}")
                        if sample_index > seen:
                            raise ValueError("Checkpoint sample index exceeds shard length")
                        del futures[shard_index]
                        path.unlink(missing_ok=True)
                        shard_index += 1
                        sample_index = 0
                        self._cycle, self._shard, self._sample = cycle, shard_index, 0
        finally:
            self._active = False

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint of the next unconsumed sample.

        Returns
        -------
        dict[str, Any]
            Versioned shard list, options, and cursor.
        """
        return {"version": 1, "shards": list(self.shards), "repeat": self.repeat,
                "position": [self._cycle, self._shard, self._sample]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a cursor after validating the stream identity.

        Arguments
        ---------
        state : dict[str, Any]
            Value returned by ``state_dict``.

        Returns
        -------
        None
        """
        if self._active:
            raise RuntimeError("Close the active iterator before loading state")
        if state.get("version") != 1:
            raise ValueError("Unsupported stream state version")
        if state.get("shards") != list(self.shards) or state.get("repeat") != self.repeat:
            raise ValueError("Checkpoint shards or repeat setting do not match")
        position = state.get("position")
        if (not isinstance(position, (list, tuple)) or len(position) != 3
                or any(type(value) is not int or value < 0 for value in position)
                or position[1] > len(self.shards)
                or (position[1] == len(self.shards) and position[2] != 0)):
            raise ValueError("Invalid stream position")
        self._cycle, self._shard, self._sample = position
