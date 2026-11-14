"""An ordered, checkpointable SQLite producer/consumer iterable dataset."""

from __future__ import annotations

import copy
import hashlib
from multiprocessing.util import Finalize
import pickle
import random
import time
import traceback
import uuid
from pathlib import Path

import cloudpickle
import numpy as np
import torch
import torch.multiprocessing as mp
from speechbrain.utils.checkpoints import (
    mark_as_loader,
    mark_as_saver,
    register_checkpoint_hooks,
)
from torch.utils.data import IterableDataset, get_worker_info

from .cache import CacheWriterError, SQLiteCache, _deserialize, _serialize


def _rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
    }
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    algorithm, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state(
        (algorithm, np.asarray(keys, dtype=np.uint32), position, gaussian, cached)
    )
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class _Source:
    """Own source iteration and its exact next-sample recovery cursor."""

    def __init__(self, dataset, cursor, *, repeat, shuffle, seed):
        self.dataset = dataset
        self.repeat, self.shuffle, self.seed = repeat, shuffle, seed
        self.map_style = (
            not isinstance(dataset, IterableDataset)
            and hasattr(dataset, "__getitem__")
            and hasattr(dataset, "__len__")
        )
        self.stateful = callable(getattr(dataset, "state_dict", None)) and callable(
            getattr(dataset, "load_state_dict", None)
        )
        self.iterator = None
        self.order = None
        self.identity = {"kind": "map" if self.map_style else "iterable"}
        if self.map_style:
            self.identity["length"] = len(dataset)
            if hasattr(dataset, "data_ids"):
                self.identity["ids"] = hashlib.sha256(
                    pickle.dumps(tuple(dataset.data_ids))
                ).hexdigest()
        elif shuffle:
            raise ValueError(
                "shuffle is only supported for map-style sources; shuffle iterable sources upstream"
            )
        if cursor is not None and cursor["identity"] != self.identity:
            raise ValueError(
                "Checkpoint source length or item ordering differs from the dataset"
            )
        self.cycle = cursor["cycle"] if cursor else 0
        self.offset = cursor["offset"] if cursor else 0
        self.initial_state = (
            cursor["initial_state"]
            if cursor
            else (copy.deepcopy(dataset.state_dict()) if self.stateful else None)
        )
        self.pass_rng = cursor["pass_rng"] if cursor else _rng_state()
        if cursor and cursor.get("new_pass"):
            _restore_rng(cursor["rng"])
            self.pass_rng = _rng_state()
            if self.stateful:
                dataset.load_state_dict(copy.deepcopy(self.initial_state))
            self._prepare()
        elif cursor and self.stateful:
            dataset.load_state_dict(copy.deepcopy(cursor["source_state"]))
            _restore_rng(cursor["rng"])
            self._prepare()
        else:
            # Replay only sources without a native next-item cursor. This also
            # reconstructs randomized iterable order from its pass-start RNG.
            _restore_rng(self.pass_rng)
            self._prepare()
            if cursor and not self.map_style:
                for _ in range(self.offset):
                    try:
                        next(self.iterator)
                    except StopIteration as error:
                        raise ValueError(
                            "Checkpoint offset exceeds source length"
                        ) from error
            if cursor:
                _restore_rng(cursor["rng"])

    def _prepare(self):
        if self.map_style:
            self.order = (
                torch.randperm(
                    len(self.dataset),
                    generator=torch.Generator().manual_seed(self.seed + self.cycle),
                ).tolist()
                if self.shuffle
                else range(len(self.dataset))
            )
        else:
            self.iterator = iter(self.dataset)
            if self.iterator is self.dataset and not self.stateful:
                raise TypeError("Use a re-iterable dataset, not a one-shot iterator")

    def _restart(self):
        self.close()
        self.cycle += 1
        self.offset = 0
        if self.stateful:
            self.dataset.load_state_dict(copy.deepcopy(self.initial_state))
        self.pass_rng = _rng_state()
        self._prepare()

    def next(self):
        while True:
            try:
                if self.map_style:
                    if self.offset >= len(self.order):
                        raise StopIteration
                    item = self.dataset[self.order[self.offset]]
                else:
                    item = next(self.iterator)
                self.offset += 1
                return item
            except StopIteration:
                if not self.repeat or self.offset == 0:
                    raise
                self._restart()

    def state_dict(self):
        return {
            "identity": self.identity,
            "cycle": self.cycle,
            "offset": self.offset,
            "initial_state": self.initial_state,
            "source_state": copy.deepcopy(self.dataset.state_dict())
            if self.stateful
            else None,
            "pass_rng": self.pass_rng,
            "rng": _rng_state(),
        }

    def close(self):
        if self.iterator is not None:
            close = getattr(self.iterator, "close", None)
            if close is not None:
                close()
            self.iterator = None


def _close_pipeline_caches(dataset, error):
    """Release writer leases of cached dynamic items owned by the producer."""
    from .pipeline import SQLiteCachedDynamicItem

    seen = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        for item in getattr(getattr(dataset, "pipeline", None), "dynamic_items", []):
            if (
                isinstance(item, SQLiteCachedDynamicItem)
                and item.store._connection is not None
            ):
                try:
                    if item.store.mode == "write":
                        item.finish() if error is None else item.fail(error)
                finally:
                    item.close()
        dataset = getattr(dataset, "dataset", None)


def _shutdown_producer(process, stopped, condition):
    """Run both on explicit close and before multiprocessing's exit joins."""
    stopped.set()
    with condition:
        condition.notify_all()
    process.join(2)
    if process.is_alive():
        process.terminate()
        process.join(2)
    if process.is_alive():
        process.kill()
        process.join(2)
    process.close()


def _produce(
    payload,
    path,
    run_id,
    sequence,
    cursor_blob,
    options,
    condition,
    consumed,
    published,
    stopped,
    ready,
    done,
    errors,
    metrics,
):
    """Spawn target; all dataset access and database writes live here."""
    source = dataset = None
    failure = None
    try:
        cursor = _deserialize(cursor_blob) if cursor_blob is not None else None
        random.seed(options["seed"])
        np.random.seed(options["seed"] % (2**32))
        torch.manual_seed(options["seed"])
        dataset, initialize = cloudpickle.loads(payload)
        if initialize is not None:
            initialize()
        if (
            callable(dataset)
            and not hasattr(dataset, "__iter__")
            and not hasattr(dataset, "__getitem__")
        ):
            dataset = dataset()
        with SQLiteCache(path, mode="write") as store:
            # The durable sequence is authoritative. On recovery, retain
            # prefetched outputs and continue after the last committed one.
            db = store._connect()
            latest = db.execute(
                "SELECT id, payload FROM items WHERE id>=? AND id<? ORDER BY id DESC LIMIT 1",
                (run_id + ":", run_id + ";"),
            ).fetchone()
            if latest is not None:
                following = int(latest[0].split(":")[1]) + 1
                if following > sequence:
                    sequence = following
                    cursor = _deserialize(latest[1])["cursor"]
            source = _Source(
                dataset,
                cursor,
                repeat=options["repeat"],
                shuffle=options["shuffle"],
                seed=options["seed"],
            )
            with condition:
                published.value = sequence
                metrics["peak_buffered_samples"].value = max(
                    0, sequence - consumed.value
                )
                ready.set()
                condition.notify_all()
            while not stopped.is_set():
                with condition:
                    if (
                        sequence - consumed.value >= options["buffer_size"]
                        and not stopped.is_set()
                    ):
                        metrics["producer_wait_count"].value += 1
                        started = time.monotonic()
                        metrics["producer_wait_started"].value = started
                        try:
                            while (
                                sequence - consumed.value >= options["buffer_size"]
                                and not stopped.is_set()
                            ):
                                condition.wait(0.1)
                        finally:
                            metrics["producer_wait_seconds"].value += (
                                time.monotonic() - started
                            )
                            metrics["producer_wait_started"].value = 0
                if stopped.is_set():
                    break
                try:
                    item = source.next()
                except StopIteration:
                    break
                store.put(
                    f"{run_id}:{sequence:020d}",
                    {"item": item, "cursor": source.state_dict()},
                )
                sequence += 1
                with condition:
                    published.value = sequence
                    metrics["peak_buffered_samples"].value = max(
                        metrics["peak_buffered_samples"].value,
                        sequence - consumed.value,
                    )
                    condition.notify_all()
    except BaseException:
        failure = traceback.format_exc()
        errors.send(failure[-8000:])
    finally:
        try:
            if source is not None:
                source.close()
            if dataset is not None:
                _close_pipeline_caches(
                    dataset,
                    failure
                    or ("Caching producer stopped" if stopped.is_set() else None),
                )
        except BaseException:
            if failure is None:
                errors.send(traceback.format_exc()[-8000:])
        finally:
            errors.close()
            done.set()
            with condition:
                condition.notify_all()


@register_checkpoint_hooks
class ParallelCachedDataset(IterableDataset):
    """Spawn one source iterator and consume its ordered SQLite output log.

    Wrap a DynamicItemDataset, map-style dataset, re-iterable dataset, or
    zero-argument factory. Only the producer evaluates it. The main process
    opens SQLite read-only and yields exactly the producer's sequence.

    ``buffer_size`` bounds produced samples ahead of consumption, not the
    retained database size. Credits are released for each returned sample,
    independently of downstream batching. Use DataLoader ``num_workers=0``;
    this dataset already owns its spawned process.

    With SpeechBrain, register this dataset as a checkpoint recoverable in
    addition to the LoopedLoader returned by make_dataloader. The loader owns
    epoch/batch counters; this dataset owns the consumed source cursor.
    """

    def __init__(
        self,
        dataset,
        cache_path,
        *,
        buffer_size=128,
        repeat=False,
        shuffle=False,
        seed=0,
        timeout=None,
        poll_interval=0.05,
        worker_init_fn=None,
    ):
        super().__init__()
        if type(buffer_size) is not int or buffer_size < 1:
            raise ValueError("buffer_size must be a positive number of samples")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if poll_interval <= 0 or (timeout is not None and timeout <= 0):
            raise ValueError("timeout and poll_interval must be positive")
        self.dataset = dataset
        self.cache_path = Path(cache_path).expanduser().absolute()
        self.buffer_size, self.repeat = buffer_size, repeat
        self.shuffle, self.seed = shuffle, seed
        self.timeout, self.poll_interval = timeout, poll_interval
        self.worker_init_fn = worker_init_fn
        self._run_id = uuid.uuid4().hex
        self._next_sequence = 0
        self._cursor = None
        self._exhausted = False
        self._process = self._reader = self._errors = None
        self._condition = self._stopped = self._ready = self._done = None
        self._consumed = self._published = None
        self._finalizer = None
        self._metrics = None
        self._last_stats = None

    @property
    def total_samples(self):
        """Samples returned to the caller, excluding producer read-ahead."""
        return self._next_sequence

    @property
    def buffered_samples(self):
        """Number of published, not-yet-consumed samples in this process run."""
        return self.stats()["buffered_samples"]

    def stats(self):
        """Return a logging-friendly snapshot without starting the producer.

        Fullness describes unconsumed samples, not SQLite disk usage. Wait
        times include an ongoing wait; counts measure wait episodes, not poll
        wakeups. Statistics reset when a producer starts and remain available
        after close. They are observational and are not checkpointed.
        """
        if self._condition is None:
            if self._last_stats is not None:
                return dict(self._last_stats)
            return {
                "buffered_samples": 0,
                "buffer_capacity": self.buffer_size,
                "available_slots": self.buffer_size,
                "fullness": 0.0,
                "fullness_percent": 0.0,
                "peak_buffered_samples": 0,
                "produced_samples": self._next_sequence,
                "consumed_samples": self._next_sequence,
                "producer_alive": False,
                "producer_waiting": False,
                "consumer_waiting": False,
                "producer_wait_count": 0,
                "producer_wait_seconds": 0.0,
                "consumer_wait_count": 0,
                "consumer_wait_seconds": 0.0,
            }
        with self._condition:
            produced, consumed = self._published.value, self._consumed.value
            values = {key: value.value for key, value in self._metrics.items()}
            now = time.monotonic()
        buffered = max(0, produced - consumed)
        for role in ("producer", "consumer"):
            started = values.pop(f"{role}_wait_started")
            values[f"{role}_waiting"] = bool(started)
            if started:
                values[f"{role}_wait_seconds"] += now - started
        return {
            "buffered_samples": buffered,
            "buffer_capacity": self.buffer_size,
            "available_slots": max(0, self.buffer_size - buffered),
            "fullness": buffered / self.buffer_size,
            "fullness_percent": 100.0 * buffered / self.buffer_size,
            "produced_samples": produced,
            "consumed_samples": consumed,
            "producer_alive": self._process is not None and self._process.is_alive(),
            **values,
        }

    def _end_consumer_wait(self):
        if self._condition is not None:
            with self._condition:
                started = self._metrics["consumer_wait_started"].value
                if started:
                    self._metrics["consumer_wait_seconds"].value += (
                        time.monotonic() - started
                    )
                    self._metrics["consumer_wait_started"].value = 0

    def start(self):
        """Start prefetch lazily (or explicitly before requesting a sample)."""
        if get_worker_info() is not None:
            raise RuntimeError(
                "Use num_workers=0: ParallelCachedDataset owns its producer process"
            )
        if self._process is not None or self._exhausted:
            return
        payload = cloudpickle.dumps((self.dataset, self.worker_init_fn))
        ctx = mp.get_context("spawn")
        self._condition = ctx.Condition()
        self._consumed = ctx.Value("q", self._next_sequence, lock=False)
        self._published = ctx.Value("q", self._next_sequence, lock=False)
        self._last_stats = None
        self._metrics = {
            key: ctx.Value(
                "q" if key.endswith(("_count", "_samples")) else "d", 0, lock=False
            )
            for key in (
                "peak_buffered_samples",
                "producer_wait_count",
                "producer_wait_seconds",
                "producer_wait_started",
                "consumer_wait_count",
                "consumer_wait_seconds",
                "consumer_wait_started",
            )
        }
        self._stopped, self._ready, self._done = ctx.Event(), ctx.Event(), ctx.Event()
        self._errors, child_errors = ctx.Pipe(duplex=False)
        options = {
            "seed": self.seed,
            "repeat": self.repeat,
            "shuffle": self.shuffle,
            "buffer_size": self.buffer_size,
        }
        process = ctx.Process(
            target=_produce,
            args=(
                payload,
                self.cache_path,
                self._run_id,
                self._next_sequence,
                _serialize(self._cursor) if self._cursor is not None else None,
                options,
                self._condition,
                self._consumed,
                self._published,
                self._stopped,
                self._ready,
                self._done,
                child_errors,
                self._metrics,
            ),
        )
        try:
            process.start()
        except BaseException:
            child_errors.close()
            self._errors.close()
            self._errors = None
            raise
        child_errors.close()
        self._process = process
        # Finalize runs before multiprocessing joins non-daemon children at
        # interpreter shutdown, including when the producer is buffer-blocked.
        self._finalizer = Finalize(
            self,
            _shutdown_producer,
            args=(process, self._stopped, self._condition),
            exitpriority=10,
        )
        self._reader = SQLiteCache(self.cache_path, mode="read")

    def __iter__(self):
        if self._exhausted:
            # Standard LoopedLoader restarts finite datasets by calling iter()
            # again. Preserve counters, but begin the next source traversal.
            self._run_id = uuid.uuid4().hex
            if self._cursor is not None:
                self._cursor = copy.deepcopy(self._cursor)
                self._cursor.update(
                    cycle=self._cursor["cycle"] + 1, offset=0, new_pass=True
                )
            self._exhausted = False
        self.start()
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        self.start()
        uid = f"{self._run_id}:{self._next_sequence:020d}"
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        try:
            while True:
                if self._ready.is_set() and self._reader.contains(uid):
                    record = self._reader.get(uid)
                    self._end_consumer_wait()
                    self._cursor = record["cursor"]
                    self._next_sequence += 1
                    with self._condition:
                        # A committed row can be read just before the producer
                        # updates its shared counter. Account for that row so
                        # published >= consumed and the peak cannot miss it.
                        self._published.value = max(
                            self._published.value, self._next_sequence
                        )
                        self._metrics["peak_buffered_samples"].value = max(
                            self._metrics["peak_buffered_samples"].value,
                            self._published.value - self._consumed.value,
                        )
                        self._consumed.value = self._next_sequence
                        self._condition.notify_all()
                    return record["item"]
                # Check errors only after draining committed records. The
                # producer sends no tensor objects over multiprocessing IPC.
                if self._errors.poll():
                    try:
                        error = self._errors.recv()
                    except EOFError:
                        error = None
                    if error is not None:
                        raise CacheWriterError(error)
                if self._done.is_set() or not self._process.is_alive():
                    if self._ready.is_set() and self._reader.contains(uid):
                        continue
                    self._process.join(1)
                    if self._errors.poll():
                        try:
                            error = self._errors.recv()
                        except EOFError:
                            error = None
                        if error is not None:
                            raise CacheWriterError(error)
                    if not self._done.is_set() or self._process.exitcode not in (
                        None,
                        0,
                    ):
                        raise CacheWriterError(
                            f"Caching producer exited with code {self._process.exitcode}"
                        )
                    self._exhausted = True
                    self.close()
                    raise StopIteration
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for produced sequence {self._next_sequence}"
                    )
                with self._condition:
                    # Recheck under the condition to avoid a lost wakeup.
                    if (
                        self._published.value <= self._next_sequence
                        and not self._done.is_set()
                    ):
                        if not self._metrics["consumer_wait_started"].value:
                            self._metrics["consumer_wait_count"].value += 1
                            self._metrics[
                                "consumer_wait_started"
                            ].value = time.monotonic()
                        delay = (
                            self.poll_interval
                            if deadline is None
                            else min(
                                self.poll_interval, max(0, deadline - time.monotonic())
                            )
                        )
                        self._condition.wait(delay)
        except StopIteration:
            raise
        except BaseException:
            self.close()
            raise

    def state_dict(self):
        """Checkpoint the consumed position, never the producer's cursor."""
        return {
            "version": 1,
            "run_id": self._run_id,
            "next_sequence": self._next_sequence,
            "cursor": copy.deepcopy(self._cursor),
            "exhausted": self._exhausted,
            "options": {
                "repeat": self.repeat,
                "shuffle": self.shuffle,
                "seed": self.seed,
            },
        }

    def load_state_dict(self, state):
        """Stop prefetch and recover a checkpoint; retained read-ahead is reused."""
        if state.get("version") != 1:
            raise ValueError("Unsupported dataset checkpoint version")
        if state.get("options") != {
            "repeat": self.repeat,
            "shuffle": self.shuffle,
            "seed": self.seed,
        }:
            raise ValueError("Checkpoint repeat/shuffle/seed options differ")
        if type(state.get("next_sequence")) is not int or state["next_sequence"] < 0:
            raise ValueError("Invalid checkpoint sequence")
        if not isinstance(state.get("run_id"), str) or len(state["run_id"]) != 32:
            raise ValueError("Invalid checkpoint sequence namespace")
        if state["next_sequence"] > 0 and state.get("cursor") is None:
            raise ValueError("A consumed checkpoint requires a source cursor")
        self.close()
        self._run_id = state["run_id"]
        self._next_sequence = state["next_sequence"]
        self._cursor = copy.deepcopy(state["cursor"])
        self._exhausted = state["exhausted"]
        self._last_stats = None

    @mark_as_saver
    def save(self, path):
        """SpeechBrain checkpoint hook for the consumed source cursor."""
        torch.save(self.state_dict(), path)

    @mark_as_loader
    def load(self, path, end_of_epoch=True):
        """Restore alongside SpeechBrain's LoopedLoader counters."""
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def close(self):
        """Cancel producer waits, then join it; retain a restartable cursor."""
        if self._process is not None:
            self._end_consumer_wait()
            self._finalizer()
            # The process handle is now closed, so capture counters without
            # consulting is_alive() on that closed handle.
            self._process = None
            self._last_stats = self.stats()
            self._last_stats.update(producer_alive=False, producer_waiting=False)
            self._finalizer = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._errors is not None:
            self._errors.close()
            self._errors = None
        self._condition = None

    def __enter__(self):
        # Keep checkpoint recovery inside a context side-effect free. Iteration
        # starts prefetch; callers may request earlier prefetch with start().
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
