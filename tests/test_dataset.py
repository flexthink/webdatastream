"""Ordered producer iteration, bounded read-ahead, and consumed checkpoints."""

import io
import os
import random
import sqlite3
import tarfile
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

pytest.importorskip("speechbrain")
pytest.importorskip("cloudpickle")

from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.checkpoints import Checkpointer

from webdatastream import (
    CacheWriterError,
    ParallelCachedDataset,
    SQLiteCachedDynamicItem,
    WebDataStream,
)


class _Map:
    def __init__(self, count=7, log=None, fail_at=None, hard_exit=False):
        self.count, self.log = count, log
        self.fail_at, self.hard_exit = fail_at, hard_exit
        self.parent_pid = os.getpid()

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        assert os.getpid() != self.parent_pid, (
            "source evaluated in the training process"
        )
        if index == self.fail_at:
            if self.hard_exit:
                os._exit(7)
            raise ValueError("transform exploded")
        if self.log is not None:
            with open(self.log, "a") as file:
                file.write(f"{index}\n")
        return {
            "id": str(index),
            "value": torch.arange(index + 1),
            "pid": os.getpid(),
            "random": torch.tensor(
                [random.random(), np.random.random(), torch.rand(()).item()]
            ),
        }


class _RandomIterable(IterableDataset):
    def __iter__(self):
        for value in np.random.permutation(5):
            yield {"id": str(value), "random": torch.rand(2)}


def _wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("producer did not reach the expected state")
        time.sleep(0.01)


def _read_log(path):
    return path.read_text().splitlines() if path.exists() else []


def _shard(path, keys):
    with tarfile.open(path, "w") as archive:
        for key in keys:
            data = key.encode()
            member = tarfile.TarInfo(key + ".txt")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_source_runs_only_in_spawned_process_and_preserves_order(tmp_path):
    parent = os.getpid()
    with ParallelCachedDataset(
        _Map(), tmp_path / "cache.db", buffer_size=2, timeout=20
    ) as dataset:
        rows = list(dataset)
        assert [row["id"] for row in rows] == list(map(str, range(7)))
        assert all(row["pid"] != parent for row in rows)
        assert dataset.total_samples == 7
        assert dataset.state_dict()["exhausted"]
        stats = dataset.stats()
        assert stats["buffered_samples"] == 0
        assert stats["produced_samples"] == stats["consumed_samples"] == 7
        assert 1 <= stats["peak_buffered_samples"] <= 2
        assert stats["consumer_wait_count"] >= 1
        assert stats["consumer_wait_seconds"] > 0
        assert not stats["producer_alive"]


def test_backpressure_blocks_before_evaluating_another_sample(tmp_path):
    log = tmp_path / "read.log"
    with ParallelCachedDataset(
        _Map(10, log), tmp_path / "cache.db", buffer_size=2, timeout=20
    ) as dataset:
        dataset.start()
        _wait_for(lambda: dataset.buffered_samples == 2)
        _wait_for(lambda: dataset.stats()["producer_waiting"])
        time.sleep(0.05)
        stats = dataset.stats()
        assert stats["fullness"] == 1.0
        assert stats["fullness_percent"] == 100.0
        assert stats["available_slots"] == 0
        assert stats["buffer_capacity"] == 2
        assert stats["peak_buffered_samples"] == 2
        assert stats["producer_wait_count"] == 1
        assert stats["producer_wait_seconds"] > 0
        assert _read_log(log) == ["0", "1"]
        assert next(dataset)["id"] == "0"
        _wait_for(lambda: dataset.buffered_samples == 2)
        _wait_for(lambda: dataset.stats()["producer_wait_count"] == 2)
        assert _read_log(log) == ["0", "1", "2"]


def test_stats_are_side_effect_free_and_retain_a_closed_snapshot(tmp_path):
    path = tmp_path / "cache.db"
    with ParallelCachedDataset(_Map(1), path, buffer_size=2, timeout=20) as dataset:
        stats = dataset.stats()
        assert stats["fullness_percent"] == 0
        assert stats["available_slots"] == 2
        assert dataset._process is None and not path.exists()
        dataset.start()
        _wait_for(lambda: dataset._done.is_set())
        stats = dataset.stats()
        assert stats["buffered_samples"] == stats["peak_buffered_samples"] == 1
        assert stats["fullness"] == 0.5
        assert stats["fullness_percent"] == 50.0
        assert stats["available_slots"] == 1
        assert stats["consumer_wait_count"] == 0
        assert [item["id"] for item in dataset] == ["0"]
    stats = dataset.stats()
    assert stats["fullness"] == 0
    assert stats["peak_buffered_samples"] == 1
    assert not stats["producer_alive"]
    stats["peak_buffered_samples"] = 999
    assert dataset.stats()["peak_buffered_samples"] == 1


def test_main_connection_is_read_only(tmp_path):
    with ParallelCachedDataset(
        _Map(2), tmp_path / "cache.db", buffer_size=1, timeout=20
    ) as dataset:
        next(dataset)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            dataset._reader._connect().execute("DELETE FROM items")


def test_batch_larger_than_buffer_does_not_deadlock(tmp_path):
    with ParallelCachedDataset(
        _Map(5), tmp_path / "cache.db", buffer_size=1, timeout=20
    ) as dataset:
        loader = make_dataloader(
            dataset, batch_size=4, collate_fn=PaddedBatch, num_workers=0
        )
        batches = list(loader)
        assert batches[0].id == ["0", "1", "2", "3"]
        assert batches[1].id == ["4"]


def test_dynamic_item_dataset_closures_work_and_do_not_run_in_parent(tmp_path):
    parent = os.getpid()
    source = DynamicItemDataset(
        {str(i): {"number": i} for i in range(4)}, output_keys=["id", "features", "pid"]
    )
    source.add_dynamic_item(
        lambda number: torch.arange(number + 1), takes=["number"], provides=["features"]
    )

    def producer_pid():
        assert os.getpid() != parent
        return os.getpid()

    source.add_dynamic_item(producer_pid, takes=[], provides=["pid"])
    with ParallelCachedDataset(
        source, tmp_path / "cache.db", buffer_size=1, timeout=20
    ) as dataset:
        rows = list(dataset)
        assert [row["id"] for row in rows] == ["0", "1", "2", "3"]
        assert [len(row["features"]) for row in rows] == [1, 2, 3, 4]
        assert all(row["pid"] != parent for row in rows)


def test_cached_pipeline_writer_is_owned_and_closed_by_producer(tmp_path):
    source = DynamicItemDataset(
        {"a": {"value": 3}, "b": {"value": 4}}, output_keys=["id", "tokens"]
    )
    cache = SQLiteCachedDynamicItem(
        tmp_path / "features.db",
        mode="write",
        takes=["id", "value"],
        provides=["tokens"],
        func=lambda uid, value: torch.arange(value),
    )
    source.add_dynamic_item(cache)
    with ParallelCachedDataset(source, tmp_path / "output.db", timeout=20) as dataset:
        assert [row["id"] for row in dataset] == ["a", "b"]
    assert cache.store._connection is None
    with cache.reader(timeout=0.1) as reader:
        assert torch.equal(reader("b"), torch.arange(4))
        with pytest.raises(KeyError):
            reader("absent")


def test_looped_loader_restarts_finite_source_and_retains_counters(tmp_path):
    with ParallelCachedDataset(
        _Map(3), tmp_path / "cache.db", buffer_size=1, timeout=20
    ) as dataset:
        loader = make_dataloader(
            dataset,
            batch_size=2,
            collate_fn=PaddedBatch,
            num_workers=0,
            looped_nominal_epoch=3,
        )
        assert [batch.id for batch in loader] == [["0", "1"], ["2"], ["0", "1"]]
        assert (
            loader.step == 0 and loader.total_steps == 3 and loader.total_samples == 5
        )
        assert [batch.id for batch in loader] == [["2"], ["0", "1"], ["2"]]
        assert loader.total_steps == 6 and loader.total_samples == 9
        assert dataset.total_samples == 9


def test_repeating_source_keeps_one_continuous_order(tmp_path):
    with ParallelCachedDataset(
        _Map(3), tmp_path / "cache.db", buffer_size=1, repeat=True, timeout=20
    ) as dataset:
        assert [next(dataset)["id"] for _ in range(8)] == [
            "0",
            "1",
            "2",
            "0",
            "1",
            "2",
            "0",
            "1",
        ]
        assert dataset.state_dict()["cursor"]["cycle"] == 2


def test_consumed_checkpoint_reuses_prefetch_without_advancing_cursor(tmp_path):
    log = tmp_path / "read.log"
    path = tmp_path / "cache.db"
    with ParallelCachedDataset(
        _Map(8, log), path, buffer_size=3, timeout=20
    ) as dataset:
        assert next(dataset)["id"] == "0"
        _wait_for(lambda: dataset.buffered_samples == 3)
        state = dataset.state_dict()
        assert state["next_sequence"] == 1
        assert state["cursor"]["offset"] == 1
        assert _read_log(log) == ["0", "1", "2", "3"]
    with ParallelCachedDataset(
        _Map(8, log), path, buffer_size=1, timeout=20
    ) as resumed:
        resumed.load_state_dict(state)
        resumed.start()
        _wait_for(lambda: resumed.buffered_samples == 3)
        assert resumed.stats()["fullness_percent"] == 300.0
        assert [row["id"] for row in resumed] == list(map(str, range(1, 8)))
    assert _read_log(log) == list(map(str, range(8)))


def test_resume_without_cached_database_restores_rng_and_map_order(tmp_path):
    options = dict(buffer_size=2, shuffle=True, repeat=True, seed=71, timeout=20)
    with ParallelCachedDataset(_Map(5), tmp_path / "first.db", **options) as dataset:
        for _ in range(3):
            next(dataset)
        state = dataset.state_dict()
        expected = [next(dataset) for _ in range(8)]
    with ParallelCachedDataset(_Map(5), tmp_path / "rebuilt.db", **options) as resumed:
        resumed.load_state_dict(state)
        actual = [next(resumed) for _ in range(8)]
    assert [row["id"] for row in actual] == [row["id"] for row in expected]
    for left, right in zip(actual, expected):
        assert torch.equal(left["random"], right["random"])


def test_stateless_random_iterable_replays_order_on_recovery(tmp_path):
    options = dict(repeat=True, seed=43, timeout=20, buffer_size=1)
    with ParallelCachedDataset(
        _RandomIterable(), tmp_path / "first.db", **options
    ) as dataset:
        for _ in range(2):
            next(dataset)
        state = dataset.state_dict()
        expected = [next(dataset) for _ in range(7)]
    with ParallelCachedDataset(
        _RandomIterable(), tmp_path / "second.db", **options
    ) as resumed:
        resumed.load_state_dict(state)
        actual = [next(resumed) for _ in range(7)]
    assert [row["id"] for row in actual] == [row["id"] for row in expected]
    for left, right in zip(actual, expected):
        assert torch.equal(left["random"], right["random"])


def test_webdatastream_uses_consumed_native_cursor_and_resumes(tmp_path):
    shards = [tmp_path / f"{i}.tar" for i in range(2)]
    for i, path in enumerate(shards):
        _shard(path, [f"{i}a", f"{i}b"])
    urls = list(map(str, shards))
    with ParallelCachedDataset(
        WebDataStream(urls), tmp_path / "first.db", buffer_size=2, timeout=20
    ) as dataset:
        assert next(dataset)["__key__"] == "0a"
        _wait_for(lambda: dataset.buffered_samples == 2)
        state = dataset.state_dict()
        assert state["cursor"]["source_state"]["position"] == [0, 0, 1]
        assert [row["__key__"] for row in dataset] == ["0b", "1a", "1b"]
    # Recovery also works after the output cache is lost, using native seek.
    with ParallelCachedDataset(
        WebDataStream(urls, local_access="direct"), tmp_path / "rebuilt.db", timeout=20
    ) as resumed:
        resumed.load_state_dict(state)
        assert [row["__key__"] for row in resumed] == ["0b", "1a", "1b"]


def test_webdatastream_finite_source_loops_under_speechbrain(tmp_path):
    path = tmp_path / "shard.tar"
    _shard(path, ["a", "b"])
    with ParallelCachedDataset(
        WebDataStream([str(path)]), tmp_path / "cache.db", timeout=20
    ) as dataset:
        loader = make_dataloader(
            dataset, batch_size=1, num_workers=0, looped_nominal_epoch=5
        )
        assert [batch["__key__"][0] for batch in loader] == ["a", "b", "a", "b", "a"]


def test_speechbrain_checkpointer_recovers_dataset_and_looped_loader(tmp_path):
    path = tmp_path / "cache.db"
    with ParallelCachedDataset(_Map(5), path, buffer_size=2, timeout=20) as dataset:
        loader = make_dataloader(
            dataset,
            batch_size=2,
            collate_fn=PaddedBatch,
            num_workers=0,
            looped_nominal_epoch=4,
        )
        assert next(iter(loader)).id == ["0", "1"]
        checkpointer = Checkpointer(
            tmp_path / "checkpoints", {"cache_dataset": dataset, "loader": loader}
        )
        checkpointer.save_checkpoint(end_of_epoch=False)
        expected = [batch.id for batch in loader]
    with ParallelCachedDataset(_Map(5), path, buffer_size=1, timeout=20) as resumed:
        loader = make_dataloader(
            resumed,
            batch_size=2,
            collate_fn=PaddedBatch,
            num_workers=0,
            looped_nominal_epoch=4,
        )
        checkpointer = Checkpointer(
            tmp_path / "checkpoints", {"cache_dataset": resumed, "loader": loader}
        )
        checkpointer.recover_if_possible()
        assert (
            loader.step == 1 and loader.total_steps == 1 and loader.total_samples == 2
        )
        assert [batch.id for batch in loader] == expected
        assert loader.total_steps == 4 and loader.total_samples == 7


@pytest.mark.parametrize("hard_exit", [False, True])
def test_producer_failure_wakes_consumer_and_releases_resources(tmp_path, hard_exit):
    dataset = ParallelCachedDataset(
        _Map(4, fail_at=1, hard_exit=hard_exit),
        tmp_path / "cache.db",
        buffer_size=1,
        timeout=20,
    )
    assert next(dataset)["id"] == "0"
    with pytest.raises(
        CacheWriterError, match="code 7" if hard_exit else "transform exploded"
    ):
        next(dataset)
    assert dataset._process is None


def test_shutdown_unblocks_producer_waiting_on_full_buffer(tmp_path):
    dataset = ParallelCachedDataset(_Map(20), tmp_path / "cache.db", buffer_size=1)
    dataset.start()
    process = dataset._process
    _wait_for(lambda: dataset.buffered_samples == 1)
    dataset.close()
    assert dataset._process is None
    assert process._closed
    # The saved position is still before the first sample.
    assert dataset.total_samples == 0
    with dataset:
        assert next(dataset)["id"] == "0"


def test_source_factory_and_initialization_run_in_child(tmp_path):
    parent = os.getpid()
    initialized = tmp_path / "initialized"

    def initialize():
        assert os.getpid() != parent
        initialized.write_text("ready")

    def factory():
        assert initialized.exists()
        source = _Map(2)
        source.parent_pid = parent
        return source

    with ParallelCachedDataset(
        factory, tmp_path / "cache.db", worker_init_fn=initialize, timeout=20
    ) as dataset:
        assert [row["id"] for row in dataset] == ["0", "1"]


def test_changed_map_order_is_rejected_on_resume(tmp_path):
    source = DynamicItemDataset(
        {"a": {"number": 1}, "b": {"number": 2}}, output_keys=["id"]
    )
    with ParallelCachedDataset(source, tmp_path / "first.db", timeout=20) as dataset:
        next(dataset)
        state = dataset.state_dict()
    source.data_ids.reverse()
    with ParallelCachedDataset(source, tmp_path / "second.db", timeout=20) as dataset:
        dataset.load_state_dict(state)
        with pytest.raises(CacheWriterError, match="ordering differs"):
            next(dataset)


def test_empty_source_finishes_without_deadlock(tmp_path):
    with ParallelCachedDataset(
        _Map(0), tmp_path / "cache.db", repeat=True, timeout=20
    ) as dataset:
        assert list(dataset) == []


def test_additional_dataloader_workers_are_rejected(tmp_path):
    dataset = ParallelCachedDataset(_Map(2), tmp_path / "cache.db", timeout=20)
    loader = DataLoader(dataset, num_workers=1, multiprocessing_context="spawn")
    with pytest.raises(RuntimeError, match="num_workers=0"):
        next(iter(loader))
    assert not dataset.cache_path.exists()


def _exit_without_explicit_close(path):
    dataset = ParallelCachedDataset(_Map(5), path, buffer_size=1)
    dataset.start()
    _wait_for(lambda: dataset.buffered_samples == 1)
    # Finalize must stop this producer before multiprocessing joins children.
    # Keep the dataset reachable until process shutdown rather than relying
    # on garbage collection of its local variable.
    import builtins

    builtins._cached_dataset_for_exit_test = dataset


def test_interpreter_exit_does_not_join_a_buffer_blocked_producer_forever(tmp_path):
    import torch.multiprocessing as mp

    process = mp.get_context("spawn").Process(
        target=_exit_without_explicit_close, args=(tmp_path / "cache.db",)
    )
    process.start()
    try:
        process.join(40)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_reader_timeout_cancels_startup_and_leaves_a_recoverable_cursor(tmp_path):
    dataset = ParallelCachedDataset(_Map(2), tmp_path / "cache.db", timeout=0.01)
    with pytest.raises(TimeoutError):
        next(dataset)
    assert dataset._process is None
    assert dataset.total_samples == 0


@pytest.mark.parametrize("interval", [0, 2])
def test_automatic_stats_from_both_processes(tmp_path, capfd, interval):
    """Producer output is visible before consumption, with optional silence."""
    with ParallelCachedDataset(
        _Map(4), tmp_path / "stats.db", buffer_size=5,
        stats_interval=interval, timeout=30,
    ) as dataset:
        dataset.start()
        deadline = time.monotonic() + 30
        while dataset.stats()["produced_samples"] < 4:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # Publication precedes display, so join the finite producer first.
        dataset._process.join(30)
        assert not dataset._process.is_alive()
        producer_output = capfd.readouterr().out
        assert producer_output.count("Cache [producer]") == (2 if interval else 0)
        assert "Cache [consumer]" not in producer_output
        assert len(list(dataset)) == 4
        consumer_output = capfd.readouterr().out
        assert consumer_output.count("Cache [consumer]") == (2 if interval else 0)
        if interval:
            assert '"buffer_capacity": 5' in producer_output
            assert '"consumed_samples": 4' in consumer_output


@pytest.mark.parametrize("interval", [-1, True, 1.5, None])
def test_invalid_stats_interval(tmp_path, interval):
    """Reject invalid display intervals before starting a process."""
    with pytest.raises(ValueError, match="stats_interval"):
        ParallelCachedDataset(_Map(1), tmp_path / "stats.db", stats_interval=interval)


def test_preload_waits_without_consuming_and_repeats_after_recovery(tmp_path):
    """Preload the target before returning the first record and retain order."""
    path = tmp_path / "preload.db"
    with ParallelCachedDataset(_Map(6), path, buffer_size=3,
                               preload_samples=3, timeout=30) as dataset:
        dataset.start()
        assert dataset.stats()["buffered_samples"] == 3
        assert dataset.total_samples == 0
        assert next(dataset)["id"] == "0"
        state = dataset.state_dict()
    with ParallelCachedDataset(_Map(6), path, buffer_size=3,
                               preload_samples=3, timeout=30) as restored:
        restored.load_state_dict(state)
        restored.start()
        assert restored.stats()["buffered_samples"] == 3
        assert restored.total_samples == 1
        assert next(restored)["id"] == "1"


@pytest.mark.parametrize("preload", [-1, 4, True, None, 1.5])
def test_invalid_preload_target(tmp_path, preload):
    """Reject impossible preload targets before starting a producer."""
    with pytest.raises(ValueError, match="preload_samples"):
        ParallelCachedDataset(_Map(4), tmp_path / "cache.db",
                              buffer_size=3, preload_samples=preload)


@pytest.mark.parametrize("failing", [False, True])
def test_preload_source_shortfall_or_failure_closes_producer(tmp_path, failing):
    """Stop waiting and release the process on exhausted or failed sources."""
    dataset = ParallelCachedDataset(
        _Map(2, fail_at=1 if failing else None), tmp_path / "preload.db",
        buffer_size=3, preload_samples=3, timeout=30,
    )
    with pytest.raises(CacheWriterError if failing else RuntimeError):
        dataset.start()
    assert dataset._process is None
