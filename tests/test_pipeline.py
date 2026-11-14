"""SpeechBrain compatibility and per-sample cache batching semantics."""

import pickle

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("speechbrain")
pytest.importorskip("tensordict")

from speechbrain.dataio.batch import PaddedData
from speechbrain.utils.data_pipeline import DataPipeline, provides, takes
from tensordict import TensorDict

from webdatastream import SQLiteCachedDynamicItem, BatchedPipelineDataset, ParallelCachedDataset


class _Waveforms(torch.utils.data.IterableDataset):
    """Finite stateful source with unequal waveform lengths."""

    def __init__(self):
        self.position = 0

    def __iter__(self):
        while self.position < 5:
            index = self.position
            self.position += 1
            yield {"id": str(index), "signal": torch.arange(index + 2).float()}

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


def _batched_waveforms(calls):
    @takes("signal")
    @provides("audio")
    def decode(signal):
        return signal + 1

    @takes("audio")
    @provides("latent")
    def encode(signals):
        calls.append(len(signals))
        padded = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True)
        return PaddedData(padded * 2, torch.tensor([len(s) / padded.shape[1] for s in signals]))

    return BatchedPipelineDataset(
        _Waveforms(),
        DataPipeline(["id", "signal"], [decode], ["id", "audio"]),
        DataPipeline(["id", "audio"], [encode], ["latent"]),
        batch_size=3,
    )


def test_two_stage_pipeline_partial_batch_and_checkpoint():
    """Trim PaddedData and recover a batch suffix without encoding it twice."""
    calls = []
    source = _batched_waveforms(calls)
    iterator = iter(source)
    first = next(iterator)
    state = source.state_dict()
    expected = list(iterator)
    assert calls == [3, 2]
    assert first["id"] == "0"
    restored_calls = []
    restored = _batched_waveforms(restored_calls)
    restored.load_state_dict(state)
    actual = list(restored)
    assert restored_calls == [2]
    assert [r["id"] for r in actual] == [r["id"] for r in expected]
    for index, record in enumerate([first, *actual]):
        assert record["latent"].shape == (index + 2,)
        torch.testing.assert_close(record["latent"], (torch.arange(index + 2).float() + 1) * 2)


def test_two_stage_pipeline_spawned_cache(tmp_path):
    """Keep sample order and variable lengths with a buffer below codec batch size."""
    with ParallelCachedDataset(_batched_waveforms([]), tmp_path / "batch.db",
                               buffer_size=1, timeout=30) as cache:
        records = list(cache)
    assert [r["id"] for r in records] == [str(i) for i in range(5)]
    assert [len(r["latent"]) for r in records] == [2, 3, 4, 5, 6]


def test_serialized_padded_data_keeps_its_named_fields(tmp_path):
    from webdatastream import SQLiteCache

    original = PaddedData(torch.ones(2, 4), torch.tensor([1.0, 0.5]))
    with SQLiteCache(tmp_path / "cache.db", mode="write") as writer:
        writer.put("sample", original)
    with SQLiteCache(tmp_path / "cache.db") as reader:
        restored = reader.get("sample")
        assert isinstance(restored, PaddedData)
        assert torch.equal(restored.data, original.data)
        assert torch.equal(restored.lengths, original.lengths)


def _spawned_pipeline_writer(path):
    @SQLiteCachedDynamicItem.cache(path)
    @takes("id")
    @provides("tokens", "features")
    def encode(ids):
        tokens = torch.arange(12).reshape(3, 4)
        return (
            PaddedData(tokens, torch.tensor([1.0, 0.5, 0.75])),
            PaddedData(tokens.float()[:, :, None], torch.tensor([1.0, 0.5, 0.75])),
        )

    with encode:
        pipeline = DataPipeline(["id"], [encode], ["tokens", "features"])
        pipeline.compute_outputs({"id": ["a", "b", "c"]})


def test_spawned_producer_with_id_only_consumer_pipeline(tmp_path):
    path = tmp_path / "cache.db"
    process = mp.get_context("spawn").Process(
        target=_spawned_pipeline_writer, args=(path,)
    )
    reader = SQLiteCachedDynamicItem(path, provides=["tokens", "features"], timeout=20)
    process.start()
    try:
        with reader:
            pipeline = DataPipeline(["id"], [reader], ["tokens", "features"])
            output = pipeline.compute_outputs({"id": "c"})
            assert torch.equal(output["tokens"], torch.tensor([8, 9, 10]))
            assert output["features"].shape == (3, 1)
        process.join(20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_scalar_cache_hit_and_id_only_reader_pipeline(tmp_path):
    calls = []

    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id", "signal")
    @provides("tokens", "features")
    def encode(uid, signal):
        calls.append(uid)
        return signal.to(torch.int64), signal[:, None]

    with encode:
        pipeline = DataPipeline(["id", "signal"], [encode], ["tokens", "features"])
        first = pipeline.compute_outputs({"id": "a", "signal": torch.arange(3.0)})
        again = encode("a", torch.zeros(9))
        assert torch.equal(first["tokens"], again[0])
        assert calls == ["a"]
    with encode.reader(timeout=0.1) as reader:
        pipeline = DataPipeline(["id"], [reader], ["features"])
        output = pipeline.compute_outputs({"id": "a"})
        assert torch.equal(output["features"], torch.arange(3.0)[:, None])


def test_padded_multi_outputs_are_trimmed_individually(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id", "signal")
    @provides("tokens", "features")
    def encode(ids, signal):
        return (
            PaddedData(signal.long(), torch.tensor([1.0, 0.5, 0.0])),
            PaddedData(
                signal[:, :, None].expand(-1, -1, 2), torch.tensor([0.5, 1.0, 0.0])
            ),
        )

    with encode:
        encode(["a", "b", "c"], torch.arange(12.0).reshape(3, 4))
    with encode.reader() as reader:
        for uid, token_shape, feature_shape in [
            ("a", (4,), (2, 2)),
            ("b", (2,), (4, 2)),
            ("c", (0,), (0, 2)),
        ]:
            tokens, features = reader(uid)
            assert tokens.shape == token_shape
            assert features.shape == feature_shape
            assert tokens.dtype == torch.int64
        assert torch.equal(reader("b")[0], torch.tensor([4, 5]))


def test_tensordict_multi_output_is_unbatched_and_pipeline_compatible(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id", "signal")
    @provides("tokens", "features")
    def encode(ids, signal):
        return TensorDict(
            {"tokens": signal.long(), "features": signal[:, :, None]},
            batch_size=[len(ids)],
        )

    with encode:
        pipeline = DataPipeline(["id", "signal"], [encode], ["tokens", "features"])
        output = pipeline.compute_outputs(
            {"id": ["a", "b"], "signal": torch.arange(6.0).reshape(2, 3)}
        )
        assert output["tokens"].shape == (2, 3)
    with encode.reader() as reader:
        tokens, features = reader("b")
        assert torch.equal(tokens, torch.tensor([3, 4, 5]))
        assert features.shape == (3, 1)


def test_tensordict_as_single_output_preserves_structure(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id")
    @provides("codec")
    def encode(ids):
        return TensorDict(
            {
                "tokens": torch.arange(6).reshape(2, 3),
                "nested": {"features": torch.ones(2, 4, 2)},
            },
            batch_size=[2],
        )

    with encode:
        encode(["a", "b"])
    with encode.reader() as reader:
        output = reader("a")
        assert isinstance(output, TensorDict)
        assert output.batch_size == torch.Size([])
        assert torch.equal(output["tokens"], torch.arange(3))
        assert output["nested", "features"].shape == (4, 2)


def test_nested_mapping_unbatches_padded_values(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id")
    @provides("codec")
    def encode(ids):
        return {
            "tokens": PaddedData(torch.ones(2, 4), torch.tensor([0.25, 1.0])),
            "nested": {"value": torch.tensor([3, 4])},
        }

    with encode:
        encode(["a", "b"])
    with encode.reader() as reader:
        assert reader("a")["tokens"].shape == (1,)
        assert reader("b")["nested"]["value"].item() == 4


def test_invalid_batch_does_not_publish_any_sample(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id")
    @provides("tokens", "features")
    def encode(ids):
        return torch.ones(2, 4), torch.ones(3, 4)

    with encode:
        with pytest.raises(ValueError, match="batch size"):
            encode(["a", "b"])
        assert not encode.store.contains("a")
        assert not encode.store.contains("b")


def test_producer_error_is_published_to_consumers(tmp_path):
    @SQLiteCachedDynamicItem.cache(tmp_path / "cache.db")
    @takes("id")
    @provides("tokens")
    def encode(uid):
        raise ValueError("codec exploded")

    with pytest.raises(ValueError, match="codec exploded"):
        with encode:
            encode("a")
    with encode.reader() as reader:
        from webdatastream import CacheWriterError

        with pytest.raises(CacheWriterError, match="codec exploded"):
            reader("a")


def test_reader_can_be_pickled_after_use(tmp_path):
    writer = SQLiteCachedDynamicItem(
        tmp_path / "cache.db",
        mode="write",
        takes=["id"],
        provides=["tokens"],
        func=lambda uid: torch.ones(2),
    )
    with writer:
        writer("a")
    with writer.reader() as reader:
        reader("a")
        restored = pickle.loads(pickle.dumps(reader))
        with restored:
            assert torch.equal(restored("a"), torch.ones(2))
