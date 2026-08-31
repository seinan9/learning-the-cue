import gzip
import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def timer(name: str):
    """Log start/end and elapsed time around a block, e.g. `with timer("Training"): ...`."""
    logging.info("%s...", name)
    start = time.time()
    yield
    elapsed = time.time() - start
    logging.info("%s completed in %.1fs", name, elapsed)


def _open(path, mode):
    """Open path for text I/O, transparently gzip-compressed if its suffix is .gz."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")  # 't' = text mode
    return open(path, mode, encoding="utf-8")


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write one JSON object per line. Gzipped automatically if path ends in .gz."""
    with _open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON-lines file into a list of dicts. Handles .gz transparently."""
    with _open(path, "r") as f:
        return [json.loads(line) for line in f]


class chunked(Sequence):
    """Consecutive slices of `seq`, each of length `size` (last one may be shorter).

    Lazy: slices are only materialized when indexed, e.g. `for batch in chunked(rows, 64)`.
    """

    def __init__(self, seq, size):
        self.seq = seq
        self.size = size

    def __len__(self):
        return math.ceil(len(self.seq) / self.size)

    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if i < 0 or i >= len(self):
            raise IndexError(i)
        start = i * self.size
        return self.seq[start : start + self.size]
