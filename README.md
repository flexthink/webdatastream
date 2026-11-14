# webdatastream

Checkpointable PyTorch streaming of WebDataset tar shards. The stream downloads a
bounded number of shards to a temporary local cache. While training reads the
current shard, background threads fetch the next shards. Samples contain raw
field bytes, so decoding and model-specific transforms stay in your training
pipeline.

```bash
pip install -e .
```

## Sample audio CLI

Install the CLI and audio extras, then create eight synthetic audio shards by
default. Their tar members use the same `.flac.flac` naming pattern as
`~/data/librilight-fake-wds/`:

```bash
pip install -e '.[cli]'
webdatastream seed --output ./sample-wds --shards 8 --samples-per-shard 4
```

Simulate training on batches while the next shards are prefetched. The optional
checkpoint stores the next unconsumed sample; run again with `--resume` to
continue from it:

```bash
webdatastream simulate ./sample-wds --prefetch-shards 3 --batch-size 2 --steps 8 --train-seconds 0.1 --checkpoint ./stream-state.json
webdatastream simulate ./sample-wds --prefetch-shards 3 --batch-size 2 --steps 8 --train-seconds 0.1 --checkpoint ./stream-state.json --resume
```

`seed` also accepts `--duration` and `--sample-rate`. It refuses to write into a
nonempty output directory. `simulate` accepts `--pattern` for shard filenames
and reads either generated shards or an existing local WebDataset directory.

For shards on a shared project store, opt into direct reads to avoid copying
them to local scratch space:

```bash
webdatastream simulate /project/store/audio-wds --local-access direct
```

The default `--local-access stage` still copies local shards into a temporary
cache (such as `$SLURM_TMPDIR`) and prefetches upcoming shards. In the Python
API, pass `local_access="direct"` or `local_access="stage"`. Direct mode uses
local paths and `file://` URLs in place; other fsspec URLs still stage. The
checkpoint cursor is compatible across the two modes when the ordered shard
URLs are the same.

## SSH shards

SSH shard URLs use `rsync` over the `ssh` command by default. This lets SSH
read your `~/.ssh/config`, including ControlMaster settings for MFA-backed
clusters. The library does not create or close master connections. Shard
prefetch remains bounded by `prefetch_shards`.

```python
stream = WebDataStream([
    "ssh://cluster/project/data/shard-000000.tar",
    "ssh://cluster/project/data/shard-000001.tar",
])
```

SCP-style URLs such as `cluster:/project/data/shard-000000.tar` also work.
For a nonstandard port, use `ssh://cluster:2222/project/data/shard.tar` or set
the port in SSH config. Set `ssh_rsync=False` to pass SSH URLs to fsspec
instead. The simulation CLI accepts ordered `--shard-url` options and
`--no-ssh-rsync`:

```bash
webdatastream simulate --shard-url ssh://cluster/project/data/shard-000000.tar --shard-url ssh://cluster/project/data/shard-000001.tar
```

## Python API

```python
import torch
from torch.utils.data import DataLoader
from webdatastream import WebDataStream

shards = [f"https://example.org/data/{index:05d}.tar" for index in range(100)]
stream = WebDataStream(shards, prefetch_shards=3, repeat=True)
loader = DataLoader(stream, batch_size=32, num_workers=0)

for batch in loader:
    train_step(batch)
    if should_checkpoint():
        torch.save({"model": model.state_dict(), "data": stream.state_dict()}, "checkpoint.pt")

# Construct a fresh stream with the same ordered shards before resuming.
stream = WebDataStream(shards, prefetch_shards=3, repeat=True)
checkpoint = torch.load("checkpoint.pt", weights_only=False)
stream.load_state_dict(checkpoint["data"])
loader = DataLoader(stream, batch_size=32, num_workers=0)
```

To start at a specific shard and sample without a checkpoint, call
`stream.seek(shards[10], 0)` before creating the iterator. The first argument
can also be a zero-based shard ordinal. Earlier shards are not opened; a
nonzero sample index is reached by sequentially counting samples within the
selected shard.

`shards` accepts local paths and URLs supported by fsspec. Object-store URLs
may need the corresponding fsspec backend package and credentials. Shards must
contain sample members consecutively, as in standard WebDataset tar files.
The checkpoint stores the next sample coordinate, including cycle, shard, and
sample index. Downloaded but unconsumed samples are fetched again after resume.
Use `num_workers=0` so the training process owns the checkpoint cursor. With
the default PyTorch collation, each field becomes a list of bytes. Supply a
custom `collate_fn` if your model needs decoded tensors.

## Parallel SQLite pipeline cache

Install the optional SpeechBrain integration:

```bash
pip install -e '.[cache]'
```

`SQLiteCachedDynamicItem` lets one producer transform a dataset while one or
more consumers read completed samples. The producer can run arbitrarily far
ahead, subject to disk space. The cache stores its index and tensors on disk;
it does not accumulate a Python index or queue of unconsumed outputs.

The producer wraps a regular SpeechBrain `@takes`/`@provides` dynamic item.
Its first input is `id`, either one string or a batch of strings. Consumers
use an independent reader item that takes only `id` and blocks until that
sample is committed. They do not need to load audio or execute the transform.

This example runs a batched transform in a spawned process. Replace the
example transform and input construction with your dataset and model code;
the library does not implement an audio codec.

```python
from pathlib import Path

import torch
import torch.multiprocessing as mp
from speechbrain.dataio.batch import PaddedData
from speechbrain.utils.data_pipeline import DataPipeline, provides, takes
from webdatastream import SQLiteCachedDynamicItem


def produce(path):
    # Construct transforms and any accelerator model inside the producer.
    @SQLiteCachedDynamicItem.cache(path)
    @takes("id", "signal")
    @provides("features")
    def transform(ids, signal):
        return PaddedData(signal.data.square(), signal.lengths)

    # The context publishes completion, or an error if processing raises.
    with transform:
        pipeline = DataPipeline(["id", "signal"], [transform], ["features"])
        pipeline.compute_outputs({
            "id": ["sample-a", "sample-b", "sample-c"],
            "signal": PaddedData(
                torch.arange(12.).reshape(3, 4),
                torch.tensor([1., .5, .75]),
            ),
        })


if __name__ == "__main__":
    path = Path("features.sqlite3")
    process = mp.get_context("spawn").Process(target=produce, args=(path,))
    process.start()
    try:
        # A reader may be constructed before the database exists.
        with SQLiteCachedDynamicItem(
            path, provides=["features"], timeout=120,
        ) as reader:
            pipeline = DataPipeline(["id"], [reader], ["features"])
            for uid in ["sample-a", "sample-b", "sample-c"]:
                sample = pipeline.compute_outputs({"id": uid})
                print(uid, sample["features"].shape)  # [4], [2], [3]
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Producer exited with code {process.exitcode}")
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
```

Each sample's complete outputs are serialized directly with `torch.save` into
one SQLite BLOB and read with `torch.load(..., weights_only=True)`. Tensor
shapes and dtypes are preserved; reads return CPU tensors. Sliced tensors are
copied before serialization so one sample does not retain its batch's backing
storage. `PaddedData` is unbatched and trimmed using its relative lengths.
TensorDict batches are also unbatched, including a TensorDict returned as a
single provided output. A TensorDict or mapping returned for multiple
`@provides` keys is matched by key, then adapted to SpeechBrain's positional
output convention.

The database uses WAL, parameterized SQL, and cached prepared statements.
One submitted batch commits atomically, including all outputs of every
sample. A single writer is enforced per database. Connections open lazily
inside their process; use `spawn` and construct model transforms inside the
producer. `writer.reader()` is another way to create an ID-only consumer.

Use a local filesystem shared by the processes, such as node-local scratch;
SQLite WAL is unsuitable for a database shared across hosts over a network
filesystem. Use a separate database for each transform and a new path when
changing a transform or its configuration. Existing IDs are immutable.
Scalar producer calls reuse cache hits. Batched producer calls run the whole
transform for vectorized processing and preserve existing stored IDs; use
`writer.store.contains(id)` to filter already cached inputs when resuming.
Reader calls request one sample at a time, so consumers can choose their own
batch sizes and padding after loading.

Readers wait indefinitely by default; `timeout` bounds that wait and
`poll_interval` controls how often readiness is checked. Finishing the writer
makes missing IDs raise `KeyError`; producer failures raise `CacheWriterError`.
Already committed samples remain readable after either outcome. Outside a
context manager, call `finish()` or `fail(error)`, then `close()` explicitly.

For integrations that do not use SpeechBrain, `SQLiteCache` provides the same
storage and coordination directly through `put`, `put_many`, `contains`, and
`get`. It supports tensors, TensorDicts, nested dictionaries, tuples, lists,
and basic Python scalar values.

## Spawned caching dataset

`ParallelCachedDataset` wraps a dataset and manages the producer process:

```python
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import make_dataloader
from speechbrain.utils.checkpoints import Checkpointer
from webdatastream import ParallelCachedDataset

# source_dataset is a DynamicItemDataset, another map-style dataset, or an
# iterable such as WebDataStream. Its pipeline runs only in the producer.
with ParallelCachedDataset(
    source_dataset,
    "training-output.sqlite3",
    buffer_size=1024,  # maximum produced samples ahead of consumption
    repeat=True,
    shuffle=False,
    seed=42,
    timeout=120,
) as dataset:
    loader = make_dataloader(
        dataset,
        batch_size=32,
        collate_fn=PaddedBatch,
        num_workers=0,
        looped_nominal_epoch=1000,
    )
    checkpointer = Checkpointer(
        "checkpoints",
        recoverables={"cached_dataset": dataset, "loader": loader},
    )
    checkpointer.recover_if_possible()
    for batch in loader:
        train_step(batch)
```

Put process-starting code under `if __name__ == "__main__":` in a script.
Install the `cache` extra described above. Local functions and lambdas in a
SpeechBrain pipeline are supported through cloudpickle. A zero-argument
dataset factory can be passed instead of an instance when models or other
resources should be constructed inside the producer. An optional
`worker_init_fn()` runs in that process before calling the factory.

The producer is the **sole iterator of the source dataset**. It stores an
ordered sequence of outputs and recovery cursors in SQLite, and the main
process consumes that exact sequence through a read-only connection. The
main process never evaluates source dynamic items, including cache writers.
This avoids relying on two independently iterated datasets having identical
shuffles, random transforms, or source positions. Existing
`SQLiteCachedDynamicItem` writers inside the source pipeline remain owned by
the producer and have their handles closed when it stops. Give the wrapper's
output log a different database path from those individual item caches.

At the buffer limit, the producer waits **before requesting another source
sample**. When no sample is ready, the consumer waits. Each returned sample
releases a buffer slot, so a DataLoader batch can be larger than the buffer.
Waiting holds no SQLite transaction. Closing the dataset wakes a producer
waiting on a full buffer, then joins it; a producer stuck in a transform is
terminated after the shutdown grace period. Normal interpreter shutdown also
cleans up a producer waiting on its buffer. Prefer the context manager or
explicit `close()` when stopping training early.

`buffer_size` is a count of samples ahead of consumption, not a database size
limit. The persistent output log retains consumed samples and their order for
checkpoint recovery. Each new wrapper gets a separate sequence namespace;
loading a checkpoint restores the saved namespace and reuses any retained
read-ahead. `buffered_samples` reports ready samples; `total_samples` counts
samples actually returned to the caller. Construction and context entry do
not start a process; iteration does, or call `start()` for earlier prefetch.

Use `dataset.stats()` for a logging-friendly snapshot of cache fullness:

Pass `stats_interval=5` to also print a cache snapshot every five produced
samples and every five consumed samples, independently in each process.
The default `stats_interval=0` disables automatic output. Each line names the
producer or consumer and database path and includes fullness, sample counters,
peak buffering, and wait metrics. This interval counts samples before batching.

```python
stats = dataset.stats()
print(
    f"Cache: {stats['buffered_samples']}/{stats['buffer_capacity']} samples "
    f"({stats['fullness_percent']:.1f}% full); "
    f"peak={stats['peak_buffered_samples']}; "
    f"producer waited {stats['producer_wait_seconds']:.2f}s; "
    f"consumer waited {stats['consumer_wait_seconds']:.2f}s"
)
```

The snapshot includes `fullness` as a fraction, `available_slots`,
`produced_samples` and `consumed_samples` sequence counters, `producer_alive`,
and `producer_waiting`/`consumer_waiting` flags. Wait counts measure distinct
blocking episodes rather than polling wakeups; wait durations include any
current wait. Producer waits indicate backpressure from a full buffer, while
consumer waits indicate waiting for samples, including producer startup.
Statistics describe the current producer session, reset on its next start,
and remain available after `close()`. Reading them does not start a process
or access SQLite, and they are not part of checkpoints. `produced_samples`
includes retained outputs discovered during recovery. Fullness normally lies
between 0 and 100%; recovering a larger retained backlog with a smaller buffer
can temporarily exceed 100% while production waits for that backlog to drain.
These metrics measure unconsumed samples, not the retained database's disk size.

With `repeat=False` (the default), a traversal ends at source exhaustion and
a subsequent `iter(dataset)` starts a new traversal. SpeechBrain's ordinary
`LoopedLoader` therefore preserves its finite-source looping and partial-batch
behavior. With `repeat=True`, the producer loops the source continuously;
downstream batches may span source traversals. An empty source finishes
instead of spinning. Use `shuffle=True` on this wrapper for deterministic
map-style shuffling, rather than on `make_dataloader`. Iterable sources should
perform their own shuffling upstream. Pass `collate_fn=PaddedBatch` explicitly
for SpeechBrain records because the wrapper is an `IterableDataset`.

Checkpoint **both** the dataset and SpeechBrain's loader. The standard
`LoopedLoader` owns `step`, `total_steps`, `total_samples`, and nominal epoch
boundaries; the dataset's registered checkpoint hooks own its consumed sample
position. The source cursor advances only when a sample is returned in the
main process. Read-ahead cannot move the checkpoint past training. If the
SQLite log survives, recovery reuses queued outputs and resumes production
after its last committed entry. If it is lost, recovery rebuilds from the
consumed cursor. Sources implementing `state_dict`/`load_state_dict`, including
`WebDataStream`, recover their native cursor; map-style sources recover their
index and seeded permutation. Other re-iterable sources replay their current
pass to recover the position. Python, NumPy, and Torch RNG states are saved.
Sources with additional mutable state should implement the state protocol.

Use `num_workers=0`: this wrapper already spawns a producer and needs consumption
to occur in the training process for accurate checkpoints. Keep output caches
on a local filesystem, use a different database per simultaneous producer,
and use a new path when changing pipeline semantics. To wrap raw shards:

```python
from webdatastream import WebDataStream, ParallelCachedDataset

source = WebDataStream(shards, prefetch_shards=3)
dataset = ParallelCachedDataset(source, "shard-output.sqlite3", buffer_size=128)
```

Raw WebDataStream samples retain their byte fields and `__key__`/`__url__`
metadata. For transformed audio, wrap a compatible pipeline dataset instead.
