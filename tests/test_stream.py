"""Behavioral checks for shard prefetch and checkpoint recovery."""

import io
import tarfile
from pathlib import Path
from threading import Event

import pytest

from webdatastream import WebDataStream
from webdatastream import stream


def make_shard(path: Path, names: list[str]) -> None:
    """Write a small WebDataset shard with two fields per sample."""
    with tarfile.open(path, "w") as archive:
        for name in names:
            for extension in ("txt", "json"):
                payload = f"{name}.{extension}".encode()
                member = tarfile.TarInfo(f"{name}.{extension}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def test_resume_tracks_consumed_sample(tmp_path: Path) -> None:
    """Prefetched shards must not advance the saved training cursor."""
    paths = [tmp_path / f"{index}.tar" for index in range(3)]
    for index, path in enumerate(paths):
        make_shard(path, [f"{index}a", f"{index}b"])
    urls = [str(path) for path in paths]
    dataset = WebDataStream(urls, prefetch_shards=3)
    iterator = iter(dataset)
    assert next(iterator)["txt"] == b"0a.txt"
    assert next(iterator)["__key__"] == "0b"
    assert dataset.state_dict()["position"] == [0, 0, 2]
    state = dataset.state_dict()
    iterator.close()
    resumed = WebDataStream(urls)
    resumed.load_state_dict(state)
    assert [sample["__key__"] for sample in resumed] == ["1a", "1b", "2a", "2b"]
    assert resumed.state_dict()["position"] == [0, 3, 0]


def test_next_shard_download_starts_during_current_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The next shard begins staging before the current shard is exhausted."""
    paths = [tmp_path / f"{index}.tar" for index in range(2)]
    for index, path in enumerate(paths):
        make_shard(path, [str(index)])
    started = Event()
    original = stream._download

    def observed(url: str, destination: Path) -> Path:
        if url == str(paths[1]):
            started.set()
        return original(url, destination)

    monkeypatch.setattr(stream, "_download", observed)
    iterator = iter(WebDataStream([str(path) for path in paths]))
    assert next(iterator)["__key__"] == "0"
    assert started.wait(2)
    iterator.close()


def test_rejects_changed_shards() -> None:
    """A cursor must not silently resume on a different shard order."""
    state = WebDataStream(["a.tar", "b.tar"]).state_dict()
    with pytest.raises(ValueError, match="shards"):
        WebDataStream(["b.tar", "a.tar"]).load_state_dict(state)


def test_repeat_resume(tmp_path: Path) -> None:
    """A repeated stream recovers at the same cycle and sample."""
    path = tmp_path / "one.tar"
    make_shard(path, ["a", "b"])
    dataset = WebDataStream([str(path)], repeat=True)
    iterator = iter(dataset)
    assert [next(iterator)["__key__"] for _ in range(3)] == ["a", "b", "a"]
    state = dataset.state_dict()
    iterator.close()
    resumed = WebDataStream([str(path)], repeat=True)
    resumed.load_state_dict(state)
    continuation = iter(resumed)
    assert [next(continuation)["__key__"] for _ in range(3)] == ["b", "a", "b"]
    continuation.close()
