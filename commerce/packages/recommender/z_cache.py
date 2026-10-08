"""Seller-local cache of frozen text vectors z (model.md §7).

Key = SHA-256 of [text_artifact_hash, preprocessing_version, SHA-256(text)], so
a new encoder artifact, a builder change or an edited product text each miss
the cache and recompute, and nothing else does. The cache holds only derived
vectors of the seller's own catalog; it never goes into an FL export.
"""
import hashlib
from pathlib import Path
import sqlite3
from typing import Sequence

import numpy as np

from commerce.packages.contracts.ids import canonical_json
from commerce.packages.recommender.text_encoder import FrozenTextEncoder, artifact_hash

# The text builder and the relation definitions and constants (model.md §2: preprocessing_version).
PREPROCESSING_SOURCES = (Path(__file__).resolve().parents[1] / "data_adapters" / "text.py",
                         Path(__file__).resolve().parent / "relations.py")


def preprocessing_version(sources: Sequence[Path] = PREPROCESSING_SOURCES) -> str:
    # .gitattributes checks *.py out with LF everywhere, so the bytes match across machines.
    files = [[p.name, hashlib.sha256(p.read_bytes()).hexdigest()] for p in sources]
    return hashlib.sha256(canonical_json(files)).hexdigest()


class ZCache:
    def __init__(self, path: str | Path, encoder: FrozenTextEncoder, model_dir: str | Path, *,
                 preprocessing: str | None = None):
        self.encoder = encoder
        self.text_artifact_hash = artifact_hash(model_dir, encoder.spec)
        self.preprocessing_version = preprocessing or preprocessing_version()
        # Several runs share one cache: a writer waits for the others instead of failing at once.
        self._db = sqlite3.connect(str(path), timeout=300)
        self._db.execute("CREATE TABLE IF NOT EXISTS z (key TEXT PRIMARY KEY, dim INTEGER NOT NULL, "
                         "vector BLOB NOT NULL)")
        self.encoded = 0  # texts encoded by this instance; cache hits do not count

    def close(self) -> None:
        self._db.close()

    def key(self, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return hashlib.sha256(canonical_json(
            [self.text_artifact_hash, self.preprocessing_version, text_hash])).hexdigest()

    def vectors(self, texts: Sequence[str]) -> np.ndarray:
        """z for each text in order; encodes and stores only the misses."""
        keys = [self.key(t) for t in texts]
        found = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's variable limit
            chunk = keys[start:start + 500]
            rows = self._db.execute("SELECT key, dim, vector FROM z WHERE key IN (%s)"
                                    % ",".join("?" * len(chunk)), chunk)
            for key, dim, blob in rows:
                if dim != self.encoder.dim:
                    raise ValueError("a cached vector has another dimension than the encoder")
                found[key] = np.frombuffer(blob, dtype=np.float32)
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in found}.items())
        if missing:
            z = self.encoder.encode([t for _, t in missing])
            with self._db:
                self._db.executemany("INSERT OR REPLACE INTO z VALUES (?, ?, ?)",
                                     [(k, self.encoder.dim, row.tobytes()) for (k, _), row in zip(missing, z)])
            found.update((k, row) for (k, _), row in zip(missing, z))
            self.encoded += len(missing)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, self.encoder.dim), np.float32)
