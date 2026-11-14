"""SpeechBrain dynamic items backed by the parallel SQLite cache."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from speechbrain.dataio.batch import PaddedData
from speechbrain.utils.data_pipeline import DynamicItem, GeneratorDynamicItem
from tensordict import TensorDictBase

from .cache import SQLiteCache


def _unbatch(value: Any, count: int) -> list[Any]:
    """Split one provided output, removing SpeechBrain's right padding."""
    if isinstance(value, PaddedData):
        data, lengths = value
        if data.ndim < 2 or data.shape[0] != count or lengths.shape != (count,):
            raise ValueError("PaddedData dimensions do not match the ID batch")
        result = []
        for item, relative_length in zip(data, lengths):
            length = float(relative_length)
            if not math.isfinite(length) or not 0 <= length <= 1:
                raise ValueError(
                    "PaddedData lengths must be finite fractions in [0, 1]"
                )
            result.append(item[: round(length * data.shape[1])])
        return result
    if isinstance(value, TensorDictBase):
        if not value.batch_size or value.batch_size[0] != count:
            raise ValueError("TensorDict batch size does not match the ID batch")
        return [value[index] for index in range(count)]
    if isinstance(value, Mapping):
        columns = {key: _unbatch(item, count) for key, item in value.items()}
        return [
            {key: column[index] for key, column in columns.items()}
            for index in range(count)
        ]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError("Tensor batch size does not match the ID batch")
        return list(value.unbind(0))
    if isinstance(value, (tuple, list)) and len(value) == count:
        return list(value)
    raise ValueError("Each batched output must have one value per ID")


class SQLiteCachedDynamicItem(DynamicItem):
    """A producer transform or an ID-only blocking reader for SpeechBrain.

    Decorate a regular ``@takes('id', ...)``/``@provides(...)`` item with
    :meth:`cache`. Producer calls accept a string ID or a sequence of string
    IDs. Batched outputs may be tensors, PaddedData, mappings, or TensorDicts;
    each sample is serialized separately. Multiple outputs are committed
    together in the same sample BLOB.

    :meth:`reader` creates a dynamic item that takes only ``id``. Thus a
    consumer pipeline does not need the producer's audio or transform inputs.
    Reader calls request one ID at a time and never execute the transform.
    Handles are lazy and pickleable for ``torch.multiprocessing`` spawn.
    """

    def __init__(
        self,
        path,
        *,
        mode="read",
        takes=None,
        provides=None,
        func=None,
        timeout=None,
        poll_interval=0.05,
    ):
        takes = list(takes) if takes is not None else ["id"]
        provides = list(provides) if provides is not None else []
        if not takes or takes[0] != "id":
            raise ValueError("The first takes key must be 'id'")
        if not provides or any(not isinstance(key, str) for key in provides):
            raise ValueError("provides must contain output names")
        if len(set(provides)) != len(provides):
            raise ValueError("provides must contain unique names")
        if mode == "write" and func is None:
            raise ValueError("A writer requires a transform function")
        if mode == "read" and takes != ["id"]:
            raise ValueError("A reader takes only 'id'")
        super().__init__(takes=takes, func=func, provides=provides)
        self.store = SQLiteCache(
            path, mode=mode, timeout=timeout, poll_interval=poll_interval
        )

    @classmethod
    def cache(cls, path, *, mode="write", **kwargs):
        """Wrap a regular SpeechBrain dynamic item in a SQLite cache."""

        def decorate(item):
            if not isinstance(item, DynamicItem) or isinstance(
                item, GeneratorDynamicItem
            ):
                raise TypeError(
                    "Decorate a regular SpeechBrain DynamicItem, not a generator"
                )
            return cls(
                path,
                mode=mode,
                takes=item.takes if mode == "write" else ["id"],
                provides=item.provides,
                func=item.func if mode == "write" else None,
                **kwargs,
            )

        return decorate

    def reader(self, *, timeout=None, poll_interval=None):
        """Create an independent, ID-only consumer dynamic item."""
        return type(self)(
            self.store.path,
            provides=self.provides,
            mode="read",
            timeout=self.store.timeout if timeout is None else timeout,
            poll_interval=self.store.poll_interval
            if poll_interval is None
            else poll_interval,
        )

    def _outputs(self, result):
        if len(self.provides) == 1:
            return {self.provides[0]: result}
        if isinstance(result, (Mapping, TensorDictBase)):
            return {key: result[key] for key in self.provides}
        if not isinstance(result, (tuple, list)) or len(result) != len(self.provides):
            raise ValueError("Transform output count does not match provides")
        return dict(zip(self.provides, result))

    def _result(self, outputs):
        if set(outputs) != set(self.provides):
            raise ValueError("Cached outputs do not match this dynamic item's provides")
        values = [outputs[key] for key in self.provides]
        return values[0] if len(values) == 1 else tuple(values)

    def __call__(self, *args):
        if not args:
            raise TypeError("An ID is required")
        uid = args[0]
        if self.store.mode == "read":
            if len(args) != 1 or not isinstance(uid, str):
                raise TypeError("Readers take one string ID")
            return self._result(self.store.get(uid))
        if isinstance(uid, str):
            if self.store.contains(uid):
                return self._result(self.store.get(uid))
            outputs = self._outputs(self.func(*args))
            self.store.put(uid, outputs)
            return self._result(outputs)
        if (
            not isinstance(uid, Sequence)
            or not uid
            or any(not isinstance(key, str) for key in uid)
        ):
            raise TypeError(
                "Producer IDs must be a string or a nonempty sequence of strings"
            )
        if len(set(uid)) != len(uid):
            raise ValueError("Duplicate IDs in a producer batch")
        # Transform a complete input batch so codecs can run vectorized. Existing
        # IDs remain immutable; callers can filter them with store.contains().
        outputs = self._outputs(self.func(*args))
        columns = {key: _unbatch(value, len(uid)) for key, value in outputs.items()}
        self.store.put_many(
            (key, {name: values[index] for name, values in columns.items()})
            for index, key in enumerate(uid)
        )
        return self._result(outputs)

    def finish(self):
        """Mark producer completion."""
        self.store.finish()

    def fail(self, error):
        """Publish a producer error to readers."""
        self.store.fail(error)

    def close(self):
        """Close this process's cache connection."""
        self.store.close()

    def __enter__(self):
        self.store.__enter__()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.store.__exit__(exc_type, exc, traceback)
