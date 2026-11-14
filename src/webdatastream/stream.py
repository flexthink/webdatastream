"""Bounded remote shard prefetch with a consumed-sample checkpoint cursor."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import unquote, urlparse

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


def _ssh_location(url: str) -> tuple[str, str, int | None] | None:
    """Parse SSH and scp-style shard locations for the rsync transport."""
    parsed = urlparse(url)
    if parsed.scheme == "ssh":
        if not parsed.hostname or not parsed.path or parsed.query or parsed.fragment:
            raise ValueError(f"Invalid SSH shard URL: {url}")
        host = parsed.hostname
        if parsed.username:
            host = f"{unquote(parsed.username)}@{host}"
        path, port = unquote(parsed.path), parsed.port
    elif "://" not in url and re.fullmatch(r"[^:/\s]+(?:@[^:/\s]+)?:/[^\n\r]+", url):
        host, path = url.split(":", 1)
        port = None
    else:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or any(char in path for char in "\n\r\0"):
        raise ValueError(f"Invalid SSH shard URL: {url}")
    return host, path, port


def _download_ssh(
    url: str, destination: Path, location: tuple[str, str, int | None],
) -> Path:
    """Copy one SSH shard with rsync using the user's SSH configuration."""
    host, path, port = location
    remote_shell = shlex.join(["ssh", "-p", str(port)]) if port is not None else "ssh"
    try:
        subprocess.run(
            ["rsync", "-e", remote_shell, "--", f"{host}:{shlex.quote(path)}", str(destination)],
            check=True,
        )
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


def _local_path(url: str) -> Path | None:
    """Return a readable filesystem path for a local shard URL."""
    parsed = urlparse(url)
    if parsed.scheme == "file" and parsed.netloc in ("", "localhost"):
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(url)
    else:
        return None
    return path if path.is_file() else None


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
    local_access : {"stage", "direct"}
        ``"direct"`` reads existing local shards in place. The default
        ``"stage"`` copies them to the temporary cache like remote shards.
    ssh_rsync : bool
        Intercept SSH and scp-style URLs with rsync using the user's SSH
        configuration. Enabled by default. Disable to use fsspec instead.
    cache_dir : str or pathlib.Path, optional
        Parent directory for temporary downloaded shards. Defaults to the
        system temporary directory. Each iterator uses its own subdirectory.
    """

    def __init__(
        self, shards: Sequence[str], prefetch_shards: int = 2,
        repeat: bool = False, local_access: str = "stage", ssh_rsync: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        if not shards:
            raise ValueError("At least one shard is required")
        if prefetch_shards < 1:
            raise ValueError("prefetch_shards must be positive")
        if local_access not in ("stage", "direct"):
            raise ValueError("local_access must be 'stage' or 'direct'")
        self.shards = tuple(str(shard) for shard in shards)
        self.prefetch_shards = prefetch_shards
        self.repeat = repeat
        self.local_access = local_access
        self.ssh_rsync = ssh_rsync
        self.cache_dir = Path(cache_dir).expanduser().absolute() if cache_dir is not None else None
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
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="webdatastream-", dir=self.cache_dir) as cache:
                with ThreadPoolExecutor(max_workers=self.prefetch_shards) as executor:
                    futures: dict[int, Future[Path]] = {}
                    cycle, shard_index, sample_index = self._cycle, self._shard, self._sample
                    yield from self._iterate_shards(cache, executor, futures, cycle, shard_index, sample_index)
        finally:
            self._active = False

    def _iterate_shards(
        self, cache: str, executor: ThreadPoolExecutor, futures: dict[int, Future[Path]],
        cycle: int, shard_index: int, sample_index: int,
    ) -> Iterator[dict[str, Any]]:
        """Stage or directly open shards while maintaining the consumed cursor."""
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
                if self.local_access == "direct" and _local_path(url) is not None:
                    continue
                digest = hashlib.sha256(url.encode()).hexdigest()[:16]
                target = Path(cache) / f"{index}-{digest}.tar"
                location = _ssh_location(url) if self.ssh_rsync else None
                if location is None:
                    futures[index] = executor.submit(_download, url, target)
                else:
                    futures[index] = executor.submit(_download_ssh, url, target, location)
            direct_path = _local_path(self.shards[shard_index]) if self.local_access == "direct" else None
            path = direct_path if direct_path is not None else futures[shard_index].result()
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
            if direct_path is None:
                del futures[shard_index]
                path.unlink(missing_ok=True)
            shard_index += 1
            sample_index = 0
            self._cycle, self._shard, self._sample = cycle, shard_index, 0

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint of the next unconsumed sample.

        Returns
        -------
        dict[str, Any]
            Versioned shard list, options, and cursor.
        """
        return {"version": 1, "shards": list(self.shards), "repeat": self.repeat,
                "position": [self._cycle, self._shard, self._sample]}

    def seek(self, shard: str | int, index: int = 0, *, cycle: int = 0) -> None:
        """Set the next sample by shard URL (or ordinal) and sample index.

        Earlier shards are not opened. Samples preceding ``index`` in the
        selected shard are counted while reading that shard sequentially.
        """
        if self._active:
            raise RuntimeError("Close the active iterator before seeking")
        if isinstance(shard, str):
            try:
                shard_index = self.shards.index(shard)
            except ValueError as error:
                raise ValueError(f"Shard is not part of this stream: {shard}") from error
        elif type(shard) is int and 0 <= shard < len(self.shards):
            shard_index = shard
        else:
            raise ValueError("Shard must be a URL in this stream or a valid shard index")
        if type(index) is not int or index < 0:
            raise ValueError("Sample index must be a non-negative integer")
        if type(cycle) is not int or cycle < 0:
            raise ValueError("Cycle must be a non-negative integer")
        self._cycle, self._shard, self._sample = cycle, shard_index, index

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
