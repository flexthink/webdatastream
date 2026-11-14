"""Behavioral checks for shard prefetch and checkpoint recovery."""

import io
import shutil
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


def test_downloads_use_configured_cache_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage downloads under cache_dir and clean up when the stream closes."""
    shard = tmp_path / "one.tar"
    make_shard(shard, ["a", "b"])
    cache_dir = tmp_path / "training_cache"
    destinations: list[Path] = []

    def download(url: str, destination: Path) -> Path:
        destinations.append(destination)
        shutil.copyfile(shard, destination)
        return destination

    monkeypatch.setattr(stream, "_download", download)
    iterator = iter(WebDataStream(["https://example.invalid/one.tar"], cache_dir=cache_dir))
    assert next(iterator)["__key__"] == "a"
    assert destinations[0].parent.parent == cache_dir
    assert destinations[0].is_file()
    iterator.close()
    assert list(cache_dir.iterdir()) == []


def test_rejects_changed_shards() -> None:
    """A cursor must not silently resume on a different shard order."""
    state = WebDataStream(["a.tar", "b.tar"]).state_dict()
    with pytest.raises(ValueError, match="shards"):
        WebDataStream(["b.tar", "a.tar"]).load_state_dict(state)


def test_seek_starts_at_named_shard_and_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [tmp_path / f"{index}.tar" for index in range(3)]
    for index, path in enumerate(paths):
        make_shard(path, [f"{index}a", f"{index}b"])
    urls = [str(path) for path in paths]
    original = stream._download
    staged: list[str] = []

    def observed(url: str, destination: Path) -> Path:
        staged.append(url)
        return original(url, destination)

    monkeypatch.setattr(stream, "_download", observed)
    dataset = WebDataStream(urls)
    dataset.seek(urls[1], 1)
    assert dataset.state_dict()["position"] == [0, 1, 1]
    assert [sample["__key__"] for sample in dataset] == ["1b", "2a", "2b"]
    assert urls[0] not in staged


def test_seek_rejects_invalid_position() -> None:
    dataset = WebDataStream(["a.tar"])
    with pytest.raises(ValueError, match="not part"):
        dataset.seek("missing.tar")
    with pytest.raises(ValueError, match="non-negative"):
        dataset.seek("a.tar", -1)
    with pytest.raises(ValueError, match="Cycle"):
        dataset.seek("a.tar", cycle=-1)


def test_seek_preserves_repeat_cycle(tmp_path: Path) -> None:
    path = tmp_path / "one.tar"
    make_shard(path, ["a", "b"])
    dataset = WebDataStream([str(path)], repeat=True)
    dataset.seek(0, 1, cycle=3)
    iterator = iter(dataset)
    assert next(iterator)["__key__"] == "b"
    assert dataset.state_dict()["position"] == [3, 0, 2]
    assert next(iterator)["__key__"] == "a"
    assert dataset.state_dict()["position"] == [4, 0, 1]
    iterator.close()


def test_repeated_mid_shard_restarts_preserve_sequence(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.tar" for index in range(3)]
    for index, path in enumerate(paths):
        make_shard(path, [f"{index}a", f"{index}b", f"{index}c"])
    urls = [str(path) for path in paths]
    state = None
    received: list[str] = []
    for _ in range(4):
        dataset = WebDataStream(urls, prefetch_shards=3)
        if state is not None:
            dataset.load_state_dict(state)
        iterator = iter(dataset)
        for _ in range(2):
            received.append(next(iterator)["__key__"])
        state = dataset.state_dict()
        iterator.close()
    resumed = WebDataStream(urls)
    resumed.load_state_dict(state)
    received.extend(sample["__key__"] for sample in resumed)
    assert received == [f"{index}{letter}" for index in range(3) for letter in "abc"]


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


def test_direct_local_read_skips_staging_and_resumes_in_stage_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct local reads never copy shards and retain checkpoint portability."""
    paths = [tmp_path / f"{index}.tar" for index in range(2)]
    for index, path in enumerate(paths):
        make_shard(path, [f"{index}a", f"{index}b"])
    original = stream._download

    def forbidden(url: str, destination: Path) -> Path:
        raise AssertionError("Direct local access must not stage shards")

    monkeypatch.setattr(stream, "_download", forbidden)
    dataset = WebDataStream([str(path) for path in paths], local_access="direct")
    iterator = iter(dataset)
    assert next(iterator)["__key__"] == "0a"
    state = dataset.state_dict()
    iterator.close()
    monkeypatch.setattr(stream, "_download", original)
    resumed = WebDataStream([str(path) for path in paths], local_access="stage")
    resumed.load_state_dict(state)
    assert [sample["__key__"] for sample in resumed] == ["0b", "1a", "1b"]


def test_direct_file_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file URL can also be read without staging."""
    path = tmp_path / "one.tar"
    make_shard(path, ["a"])

    def forbidden(url: str, destination: Path) -> Path:
        raise AssertionError("Direct file URL must not stage")

    monkeypatch.setattr(stream, "_download", forbidden)
    assert [sample["__key__"] for sample in WebDataStream([path.as_uri()], local_access="direct")] == ["a"]


def test_ssh_urls_use_rsync_and_ssh_config_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SSH shards use rsync with plain ssh, leaving ControlMaster to config."""
    paths = [tmp_path / f"{index}.tar" for index in range(2)]
    for index, path in enumerate(paths):
        make_shard(path, [str(index)])
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert command[:4] == ["rsync", "-e", "ssh", "--"]
        assert command[4].startswith("cluster:")
        index = int(command[4].rsplit("-", 1)[1].split(".")[0])
        shutil.copyfile(paths[index], command[5])

    monkeypatch.setattr(stream.subprocess, "run", fake_run)
    urls = [f"ssh://cluster/project/shard-{index}.tar" for index in range(2)]
    assert [sample["__key__"] for sample in WebDataStream(urls)] == ["0", "1"]
    assert len(calls) == 2


def test_ssh_rsync_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The transport switch leaves SSH URLs to the original fsspec path."""
    path = tmp_path / "one.tar"
    make_shard(path, ["a"])
    seen: list[str] = []

    def fake_download(url: str, destination: Path) -> Path:
        seen.append(url)
        shutil.copyfile(path, destination)
        return destination

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("rsync must not run")

    monkeypatch.setattr(stream, "_download", fake_download)
    monkeypatch.setattr(stream.subprocess, "run", forbidden)
    assert [sample["__key__"] for sample in WebDataStream(["ssh://cluster/project/one.tar"], ssh_rsync=False)] == ["a"]
    assert seen == ["ssh://cluster/project/one.tar"]
