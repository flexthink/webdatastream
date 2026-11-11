"""End-to-end checks for the sample CLI."""

import tarfile
from pathlib import Path

from click.testing import CliRunner

from webdatastream.cli import main


def test_seed_simulate_and_resume(tmp_path: Path) -> None:
    """Generated FLAC shards decode and resume from the saved cursor."""
    runner = CliRunner()
    output = tmp_path / "audio"
    checkpoint = tmp_path / "cursor.json"
    seeded = runner.invoke(main, ["seed", "--output", str(output), "--shards", "3",
                                  "--samples-per-shard", "2", "--duration", "0.1"])
    assert seeded.exit_code == 0, seeded.output
    paths = sorted(output.glob("*.tar"))
    assert len(paths) == 3
    with tarfile.open(paths[0]) as archive:
        assert all(member.name.endswith(".flac.flac") for member in archive)
    first = runner.invoke(main, ["simulate", str(output), "--steps", "1",
                                 "--train-seconds", "0", "--checkpoint", str(checkpoint)])
    assert first.exit_code == 0, first.output
    assert "samples=2" in first.output
    resumed = runner.invoke(main, ["simulate", str(output), "--steps", "1",
                                   "--train-seconds", "0", "--checkpoint", str(checkpoint), "--resume"])
    assert resumed.exit_code == 0, resumed.output
    assert "next=[0, 1, 2]" in resumed.output


def test_seed_refuses_nonempty_output(tmp_path: Path) -> None:
    """Seeding never overwrites an existing shard directory."""
    output = tmp_path / "audio"
    output.mkdir()
    (output / "existing.tar").touch()
    result = CliRunner().invoke(main, ["seed", "--output", str(output)])
    assert result.exit_code != 0
    assert (output / "existing.tar").exists()
