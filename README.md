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

`shards` accepts local paths and URLs supported by fsspec. Object-store URLs
may need the corresponding fsspec backend package and credentials. Shards must
contain sample members consecutively, as in standard WebDataset tar files.
The checkpoint stores the next sample coordinate, including cycle, shard, and
sample index. Downloaded but unconsumed samples are fetched again after resume.
Use `num_workers=0` so the training process owns the checkpoint cursor. With
the default PyTorch collation, each field becomes a list of bytes. Supply a
custom `collate_fn` if your model needs decoded tensors.
