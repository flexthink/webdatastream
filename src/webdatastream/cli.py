"""Click commands for synthetic audio shards and a training simulation."""

from __future__ import annotations

import io
import json
import math
import tarfile
import time
from pathlib import Path
from typing import Any

import click
from torch.utils.data import DataLoader

from .stream import WebDataStream


def _soundfile() -> Any:
    """Import the optional audio dependency with an actionable error."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise click.ClickException("Install audio support with: pip install 'webdatastream[cli]'") from exc
    return sf


@click.group()
def main() -> None:
    """Create and simulate checkpointable audio WebDataset streams."""


@main.command("seed")
@click.option("--output", type=click.Path(path_type=Path), default=Path("sample-wds"), show_default=True)
@click.option("--shards", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--samples-per-shard", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--duration", type=click.FloatRange(min=0.01), default=1.0, show_default=True)
@click.option("--sample-rate", type=click.IntRange(min=1), default=16000, show_default=True)
def seed(output: Path, shards: int, samples_per_shard: int, duration: float, sample_rate: int) -> None:
    """Create synthetic FLAC tar shards like librilight-fake-wds.

    Arguments
    ---------
    output : pathlib.Path
        Directory to create.
    shards, samples_per_shard : int
        Dataset dimensions.
    duration : float
        Seconds per generated clip.
    sample_rate : int
        Audio sample rate in Hz.

    Returns
    -------
    None
    """
    import numpy as np

    sf = _soundfile()
    if output.exists() and any(output.iterdir()):
        raise click.ClickException(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames = max(1, round(duration * sample_rate))
    clock = np.arange(frames, dtype=np.float32) / sample_rate
    for shard_index in range(shards):
        path = output / f"librilight-fake-{shard_index:06d}.tar"
        with tarfile.open(path, "w") as archive:
            for sample_index in range(samples_per_shard):
                number = shard_index * samples_per_shard + sample_index
                frequency = 220 + 20 * (number % 24)
                audio = (0.2 * np.sin(2 * math.pi * frequency * clock)).astype(np.float32)
                with io.BytesIO() as buffer:
                    sf.write(buffer, audio, sample_rate, format="FLAC")
                    payload = buffer.getvalue()
                name = f"small__synthetic_{number:06d}.flac.flac"
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        click.echo(f"created {path}")
    click.echo(f"Created {shards} shards and {shards * samples_per_shard} audio samples")


@main.command("simulate")
@click.argument("source", required=False, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--shard-url", multiple=True, help="Ordered remote shard URL; repeat for each shard.")
@click.option("--ssh-rsync/--no-ssh-rsync", default=True, show_default=True)
@click.option("--pattern", default="*.tar", show_default=True)
@click.option("--prefetch-shards", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--local-access", type=click.Choice(["stage", "direct"]), default="stage", show_default=True)
@click.option("--batch-size", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--steps", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--train-seconds", type=click.FloatRange(min=0), default=0.1, show_default=True)
@click.option("--checkpoint", type=click.Path(path_type=Path), default=None)
@click.option("--resume", is_flag=True, help="Load the existing --checkpoint before simulation.")
def simulate(
    source: Path | None, shard_url: tuple[str, ...], ssh_rsync: bool,
    pattern: str, prefetch_shards: int, local_access: str, batch_size: int,
    steps: int, train_seconds: float, checkpoint: Path | None, resume: bool,
) -> None:
    """Decode FLAC batches and pause to imitate model training.

    Arguments
    ---------
    source : pathlib.Path or None
        Directory containing tar shards, omitted when using shard URLs.
    shard_url : tuple of str
        Explicit ordered remote shard URLs.
    ssh_rsync : bool
        Route SSH URLs through rsync using the user's SSH configuration.
    pattern : str
        Shard filename glob.
    prefetch_shards, batch_size, steps : int
        Streaming and simulation dimensions.
    local_access : str
        Stage local shards or read them directly.
    train_seconds : float
        Simulated training time after each batch.
    checkpoint : pathlib.Path or None
        Optional JSON checkpoint path.
    resume : bool
        Restore the saved cursor before loading.

    Returns
    -------
    None
    """
    sf = _soundfile()
    if (source is None) == (not shard_url):
        raise click.UsageError("Provide either SOURCE or one or more --shard-url options")
    if source is not None:
        paths = sorted(source.glob(pattern))
        urls = [str(path.resolve()) for path in paths if path.is_file()]
        if not urls:
            raise click.ClickException(f"No tar shards match {source / pattern}")
    else:
        urls = list(shard_url)
    if resume and checkpoint is None:
        raise click.UsageError("--resume requires --checkpoint")
    dataset = WebDataStream(
        urls, prefetch_shards=prefetch_shards,
        local_access=local_access, ssh_rsync=ssh_rsync,
    )
    if resume:
        assert checkpoint is not None
        try:
            dataset.load_state_dict(json.loads(checkpoint.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError) as exc:
            raise click.ClickException(f"Cannot load checkpoint: {exc}") from exc
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    started = time.monotonic()
    count = 0
    for step, batch in enumerate(loader, start=1):
        audio_fields = [key for key in batch if key.lower().endswith("flac")]
        if len(audio_fields) != 1:
            raise click.ClickException(f"Expected one FLAC field, found {audio_fields}")
        frames = 0
        for payload in batch[audio_fields[0]]:
            with sf.SoundFile(io.BytesIO(payload)) as audio:
                frames += len(audio)
        count += len(batch["__key__"])
        click.echo(f"step={step} samples={count} frames={frames} next={dataset.state_dict()['position']}")
        time.sleep(train_seconds)
        if step >= steps:
            break
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps(dataset.state_dict(), indent=2) + "\n", encoding="utf-8")
        click.echo(f"saved checkpoint {checkpoint}")
    click.echo(f"Loaded {count} samples in {time.monotonic() - started:.2f}s")
