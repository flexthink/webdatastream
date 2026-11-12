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
