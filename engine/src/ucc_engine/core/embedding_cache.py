# =============================================================================
# PERSISTENT QUERY EMBEDDING CACHE
# =============================================================================
# A simple disk-backed cache of (model_id, normalized_query_text) -> embedding.
# Lets Stage 4 (advanced_embedding_results) skip the SentenceTransformer encode
# step on warm restarts when the same UCC inputs are reprocessed.
#
# Design choices:
#   - One shard per model (the cache is invalid the moment the model changes,
#     so we key the file by sanitised model id).
#   - JSON manifest: query_text -> row_index in the .npy file. Simple and
#     human-inspectable. We never compact / vacuum -- if the .npy grows beyond
#     a soft limit the user can just delete the cache directory.
#   - Atomic-ish writes: write to a .tmp file then rename. We tolerate cold
#     crashes by truncating to the longest valid prefix on load.
#   - Lock-free: the cache is meant for single-process Dagster runs. If two
#     Stage 4 runs ever race on the same machine, worst case is duplicate
#     entries; embeddings are deterministic so correctness is preserved.
# =============================================================================
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


_MODEL_ID_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_model_id(model_id: str) -> str:
    """Turn a model id / path into something safe to use as a filename."""
    if not model_id:
        return "unknown"
    # Strip parent dirs so a local path like /models/bge-small becomes bge-small
    base = os.path.basename(model_id.rstrip("/").rstrip("\\")) or model_id
    return _MODEL_ID_SANITIZE_RE.sub("_", base)[:128]


class QueryEmbeddingCache:
    """Disk-backed cache of normalised-query -> embedding vector.

    Parameters
    ----------
    cache_dir : str | Path
        Directory that holds one shard per embedding model. Created if missing.
    model_id : str
        Identifier of the embedding model. The cache is sharded by model so
        switching models is safe (old cache is simply unused, not invalidated).
    dim : int
        Embedding dimensionality. Used both to size new arrays and to validate
        any cache file we load.
    """

    def __init__(self, cache_dir: str | Path, model_id: str, dim: int):
        self.cache_dir = Path(cache_dir)
        self.model_id = model_id
        self.dim = int(dim)
        self._shard = _sanitize_model_id(model_id)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._npy_path = self.cache_dir / f"{self._shard}.npy"
        self._manifest_path = self.cache_dir / f"{self._shard}.json"

        self._index: dict[str, int] = {}     # query -> row in vectors
        self._vectors: Optional[np.ndarray] = None
        self._dirty = False

        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        """Load existing shard if present and dimensions match."""
        if not (self._npy_path.exists() and self._manifest_path.exists()):
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            return

        try:
            with open(self._manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            vectors = np.load(self._npy_path, mmap_mode=None)
        except Exception:
            # Corrupt cache -- start over.
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            return

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            # Dim mismatch (likely model changed but old shard kept) -- discard.
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            return

        idx = manifest.get("index") if isinstance(manifest, dict) else None
        if not isinstance(idx, dict):
            self._vectors = np.empty((0, self.dim), dtype=np.float32)
            return

        # Truncate to the shorter of (manifest, vectors) -- defensive against
        # a half-written shard from a crashed prior run.
        max_row = vectors.shape[0]
        clean_index = {k: int(v) for k, v in idx.items() if 0 <= int(v) < max_row}
        if clean_index:
            # Renumber to remove any holes from filtered entries
            kept_rows = sorted(set(clean_index.values()))
            remap = {old: new for new, old in enumerate(kept_rows)}
            vectors = vectors[kept_rows]
            clean_index = {k: remap[v] for k, v in clean_index.items()}

        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._index = clean_index

    # ----------------------------------------------------------------- query
    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, query: str) -> bool:
        return query in self._index

    def get(self, query: str) -> Optional[np.ndarray]:
        row = self._index.get(query)
        if row is None or self._vectors is None:
            return None
        return self._vectors[row]

    def lookup_many(self, queries: Iterable[str]) -> Tuple[np.ndarray, List[str], List[int]]:
        """Return cached embeddings for queries we know, plus a list of misses.

        Returns
        -------
        cached_matrix : np.ndarray
            (n_hits, dim) array of cached embeddings, in the order of `queries`
            for the rows where lookup succeeded.
        misses : list[str]
            Queries we did NOT find in the cache (deduplicated, original order).
        hit_positions : list[int]
            Positions in the input `queries` iterable that were hits. Used so
            the caller can splice cached rows back into the full embedding
            matrix in the right order.
        """
        queries = list(queries)
        n = len(queries)
        if n == 0 or self._vectors is None or len(self._index) == 0:
            return np.empty((0, self.dim), dtype=np.float32), queries, []

        rows: List[int] = []
        hit_positions: List[int] = []
        misses_seen: set[str] = set()
        misses: List[str] = []
        for pos, q in enumerate(queries):
            r = self._index.get(q)
            if r is not None:
                rows.append(r)
                hit_positions.append(pos)
            elif q not in misses_seen:
                misses_seen.add(q)
                misses.append(q)

        if not rows:
            return np.empty((0, self.dim), dtype=np.float32), misses, []

        cached_matrix = np.ascontiguousarray(self._vectors[rows], dtype=np.float32)
        return cached_matrix, misses, hit_positions

    # ----------------------------------------------------------------- write
    def add_many(self, queries: List[str], embeddings: np.ndarray) -> None:
        """Insert new (query, embedding) pairs. Existing keys are left alone."""
        if not queries or embeddings.size == 0:
            return
        if embeddings.shape[0] != len(queries):
            raise ValueError(
                f"add_many: got {len(queries)} queries but {embeddings.shape[0]} embeddings"
            )
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"add_many: embedding dim {embeddings.shape[1]} != cache dim {self.dim}"
            )

        if self._vectors is None:
            self._vectors = np.empty((0, self.dim), dtype=np.float32)

        new_rows = []
        new_queries = []
        for q, vec in zip(queries, embeddings):
            if q in self._index:
                continue  # don't overwrite
            new_queries.append(q)
            new_rows.append(vec)
        if not new_queries:
            return

        start = self._vectors.shape[0]
        new_block = np.ascontiguousarray(np.asarray(new_rows), dtype=np.float32)
        self._vectors = np.concatenate([self._vectors, new_block], axis=0)
        for offset, q in enumerate(new_queries):
            self._index[q] = start + offset
        self._dirty = True

    # ------------------------------------------------------------------ save
    def save(self) -> None:
        """Atomically persist the shard to disk."""
        if not self._dirty:
            return
        if self._vectors is None:
            return

        # numpy.save auto-appends '.npy' if the path doesn't already end in
        # '.npy', so we keep the temporary file ending in '.npy' and rely on a
        # separate '.tmp_<pid>' prefix to avoid collisions with the live file.
        tmp_npy = self._npy_path.with_name(self._npy_path.name + ".tmp")
        tmp_manifest = self._manifest_path.with_name(self._manifest_path.name + ".tmp")

        np.save(tmp_npy, self._vectors, allow_pickle=False)
        # np.save may add a trailing .npy if our chosen name doesn't already
        # end in .npy -- detect that case so the os.replace below targets the
        # actual file numpy wrote.
        if not tmp_npy.exists():
            candidate = tmp_npy.with_name(tmp_npy.name + ".npy")
            if candidate.exists():
                tmp_npy = candidate
        with open(tmp_manifest, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_id": self.model_id,
                    "dim": self.dim,
                    "n_entries": len(self._index),
                    "index": self._index,
                },
                fh,
            )

        os.replace(tmp_npy, self._npy_path)
        os.replace(tmp_manifest, self._manifest_path)
        self._dirty = False
