"""Tensor serialization, transactional visibility, and spawned cache readers."""

import os
import pickle
import sqlite3

import pytest
import torch
import torch.multiprocessing as mp

from webdatastream import CacheWriterError, SQLiteCache


def _reader(path, started, received, connection):
    """Request an ID before the producer has created its database."""
    try:
        with SQLiteCache(path, timeout=20, poll_interval=0.01) as cache:
            started.set()
            value = cache.get("last")
            assert torch.equal(value["tokens"], torch.arange(7))
            assert value["features"].shape == (7, 3)
            received.set()
            connection.send("ok")
    except BaseException as error:
        connection.send(repr(error))
    finally:
        connection.close()


def _writer(path, proceed):
    with SQLiteCache(path, mode="write") as cache:
        cache.put("first", torch.ones(1))
        if not proceed.wait(20):
            raise TimeoutError("test did not release producer")
        cache.put("last", {"tokens": torch.arange(7), "features": torch.ones(7, 3)})


def _crashing_writer(path):
    cache = SQLiteCache(path, mode="write")
    cache.put("committed", torch.arange(3))
    os._exit(7)


def test_tensor_roundtrip_preserves_shapes_dtypes_and_containers(tmp_path):
    path = tmp_path / "cache.sqlite3"
    values = {
        "scalar": torch.tensor(2.5),
        "empty": torch.empty(0, 3, dtype=torch.int16),
        "bf16": torch.arange(6).to(torch.bfloat16).reshape(2, 3),
        "complex": torch.tensor([1 + 2j], dtype=torch.complex64),
        "nested": [torch.ones(3, dtype=torch.bool), (None, 9, "text")],
        "noncontiguous": torch.arange(24).reshape(4, 6).T,
    }
    with SQLiteCache(path, mode="write") as cache:
        cache.put("sample", values)
    with SQLiteCache(path) as cache:
        actual = cache.get("sample")
    for key in values.keys() - {"nested"}:
        assert actual[key].dtype == values[key].dtype
        assert actual[key].shape == values[key].shape
        assert torch.equal(actual[key], values[key])
        assert actual[key].device.type == "cpu"
    assert torch.equal(actual["nested"][0], values["nested"][0])
    assert actual["nested"][1] == (None, 9, "text")


def test_sliced_tensor_blob_does_not_include_backing_batch(tmp_path):
    with SQLiteCache(tmp_path / "cache.db", mode="write") as cache:
        cache.put("tiny", torch.zeros(100_000, 2)[0])
    with sqlite3.connect(tmp_path / "cache.db") as db:
        assert db.execute("SELECT length(payload) FROM items").fetchone()[0] < 10_000


def test_batch_is_atomic_on_serialization_failure(tmp_path):
    path = tmp_path / "cache.db"
    with SQLiteCache(path, mode="write") as cache:
        with pytest.raises(TypeError, match="Unsupported cached value"):
            cache.put_many([("good", torch.ones(2)), ("bad", object())])
        assert not cache.contains("good")
        cache.put_many([("good", torch.ones(2)), ("other", torch.zeros(3))])
    with SQLiteCache(path) as cache:
        assert cache.contains("good") and cache.contains("other")


def test_parameterized_ids_and_immutable_existing_items(tmp_path):
    path = tmp_path / "cache.db"
    uid = "x'); DROP TABLE items; -- / 🔊"
    with SQLiteCache(path, mode="write") as cache:
        cache.put(uid, torch.ones(2))
        cache.put(uid, torch.zeros(9))
    with SQLiteCache(path) as cache:
        assert torch.equal(cache.get(uid), torch.ones(2))


def test_completion_failure_and_reopening_writer(tmp_path):
    path = tmp_path / "cache.db"
    with SQLiteCache(path, mode="write") as cache:
        cache.put("first", torch.ones(2))
    with SQLiteCache(path, timeout=0.2) as reader:
        with pytest.raises(KeyError, match="absent"):
            reader.get("later")
        with pytest.raises(RuntimeError, match="codec failed"):
            with SQLiteCache(path, mode="write") as cache:
                cache.put("second", torch.zeros(4))
                raise RuntimeError("codec failed")
        with pytest.raises(CacheWriterError, match="codec failed"):
            reader.get("later")
        assert torch.equal(reader.get("second"), torch.zeros(4))
        with SQLiteCache(path, mode="write") as cache:
            cache.put("later", torch.arange(3))
        assert torch.equal(reader.get("later"), torch.arange(3))


def test_close_unfinished_writer_and_reject_second_writer(tmp_path):
    path = tmp_path / "cache.db"
    writer = SQLiteCache(path, mode="write")
    writer.put("first", torch.ones(2))
    try:
        with pytest.raises(RuntimeError, match="already has a writer"):
            with SQLiteCache(path, mode="write"):
                pass
    finally:
        writer.close()
    with SQLiteCache(path) as reader:
        with pytest.raises(CacheWriterError, match="before finish"):
            reader.get("absent")


@pytest.mark.parametrize("started", [False, True])
def test_timeout_before_or_after_writer_start(tmp_path, started):
    path = tmp_path / "cache.db"
    writer = SQLiteCache(path, mode="write")
    try:
        if started:
            writer.put("other", torch.zeros(1))
        with SQLiteCache(path, timeout=0.03, poll_interval=0.005) as reader:
            with pytest.raises(TimeoutError, match="missing"):
                reader.get("missing")
    finally:
        writer.close()


def test_connection_is_not_pickled_or_closed_in_parent(tmp_path):
    path = tmp_path / "cache.db"
    with SQLiteCache(path, mode="write") as writer:
        writer.put("sample", torch.ones(2))
    with SQLiteCache(path) as reader:
        reader.get("sample")
        restored = pickle.loads(pickle.dumps(reader))
        try:
            assert restored._connection is None
            assert torch.equal(restored.get("sample"), reader.get("sample"))
        finally:
            restored.close()


def test_two_spawned_readers_block_until_writer_publishes(tmp_path):
    ctx = mp.get_context("spawn")
    path = tmp_path / "cache.db"
    proceed = ctx.Event()
    readers = []
    writer = ctx.Process(target=_writer, args=(path, proceed))
    try:
        for _ in range(2):
            started, received = ctx.Event(), ctx.Event()
            parent, child = ctx.Pipe(duplex=False)
            process = ctx.Process(target=_reader, args=(path, started, received, child))
            process.start()
            child.close()
            readers.append((process, started, received, parent))
        for _, started, received, _ in readers:
            assert started.wait(20)
            assert not received.is_set()
        writer.start()
        # Ensure the producer can publish earlier IDs while readers wait.
        with SQLiteCache(path, timeout=20, poll_interval=0.01) as cache:
            assert torch.equal(cache.get("first"), torch.ones(1))
        assert all(not received.is_set() for _, _, received, _ in readers)
        proceed.set()
        for process, _, received, connection in readers:
            assert connection.poll(20)
            assert connection.recv() == "ok"
            process.join(20)
            assert process.exitcode == 0
            assert received.is_set()
        writer.join(20)
        assert writer.exitcode == 0
    finally:
        for process in [writer, *(item[0] for item in readers)]:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
        for *_, connection in readers:
            connection.close()


def test_abrupt_producer_death_preserves_commits_and_wakes_reader(tmp_path):
    path = tmp_path / "cache.db"
    process = mp.get_context("spawn").Process(target=_crashing_writer, args=(path,))
    process.start()
    try:
        process.join(20)
        assert process.exitcode == 7
        with SQLiteCache(path, timeout=1) as reader:
            assert torch.equal(reader.get("committed"), torch.arange(3))
            with pytest.raises(CacheWriterError, match="exited"):
                reader.get("missing")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
