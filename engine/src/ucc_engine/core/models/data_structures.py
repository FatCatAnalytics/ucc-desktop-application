# =============================================================================
# DATA STRUCTURES
# =============================================================================
import math
import numpy as np
from typing import Set, List, Tuple, Dict
from ..utils import generate_ngrams


class NGramIndex:
    """Inverted index from character n-grams to item ids.

    Build path keeps Python sets for fast insertion. After loading is done,
    call :py:meth:`freeze` to convert posting lists to numpy ``int64`` arrays
    and switch :py:meth:`query` to a vectorised numpy counter (orders of
    magnitude faster than the per-id dict accumulator).
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_to_ids: dict = {}
        self.id_to_name: dict = {}
        self._frozen = False
        self._max_id = -1
        self._postings: dict = {}   # ngram -> np.ndarray[int64]
        self._counter = None        # np.ndarray[int32]

    def add(self, item_id: int, text: str):
        """Add an item to the index."""
        ngrams = generate_ngrams(text, self.n)
        self.add_precomputed(item_id, text, ngrams)

    def add_precomputed(self, item_id: int, text: str, ngrams: set):
        """Add an item with pre-computed n-grams (for parallel processing)."""
        self.id_to_name[item_id] = text
        if item_id > self._max_id:
            self._max_id = item_id
        for ngram in ngrams:
            s = self.ngram_to_ids.get(ngram)
            if s is None:
                self.ngram_to_ids[ngram] = {item_id}
            else:
                s.add(item_id)
        # Any new add invalidates the frozen state
        if self._frozen:
            self._frozen = False
            self._postings = {}
            self._counter = None

    def freeze(self):
        """Pre-convert posting sets to numpy arrays + allocate counter buffer.

        After calling this, :py:meth:`query` runs in vectorised numpy.
        """
        if self._max_id < 0:
            self._postings = {}
            self._counter = np.empty(0, dtype=np.int32)
            self._frozen = True
            return
        self._postings = {
            ng: np.fromiter(ids, dtype=np.int64, count=len(ids))
            for ng, ids in self.ngram_to_ids.items()
        }
        # Reusable counter buffer indexed by item_id
        self._counter = np.zeros(self._max_id + 1, dtype=np.int32)
        self._frozen = True

    def query(self, text: str, min_overlap: int = 1) -> List[int]:
        """Query the index for similar items.

        Frozen path (after :py:meth:`freeze`) uses a numpy counter and
        ``np.add.at`` — O(total_postings) but with C-level loops.
        """
        query_ngrams = generate_ngrams(text, self.n)
        if not query_ngrams:
            return []

        if self._frozen and self._counter is not None:
            counter = self._counter
            counter.fill(0)
            postings = self._postings
            for ng in query_ngrams:
                arr = postings.get(ng)
                if arr is not None:
                    # np.add.at handles repeats safely; postings have unique ids
                    # within an ngram, so plain indexing suffices
                    counter[arr] += 1
            hits = np.flatnonzero(counter >= min_overlap)
            return hits.tolist()

        # Fallback: legacy dict-accumulator path (pre-freeze)
        candidate_counts: Dict[int, int] = {}
        for ngram in query_ngrams:
            ids = self.ngram_to_ids.get(ngram)
            if ids is None:
                continue
            for item_id in ids:
                candidate_counts[item_id] = candidate_counts.get(item_id, 0) + 1
        return [item_id for item_id, count in candidate_counts.items() if count >= min_overlap]


class TFIDFNGramIndex:
    """
    N-gram index with TF-IDF weighting.

    Common n-grams that appear in many documents (e.g. "ban", "ank" from "bank")
    are down-weighted via inverse document frequency, so similarity scores
    emphasise rare, distinguishing n-grams.

    Usage:
        idx = TFIDFNGramIndex(n=3)
        for i, name in enumerate(names):
            idx.add(i, name)
        idx.freeze()                       # computes IDF weights
        sim = idx.similarity(name_a, name_b)
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_to_ids: Dict[str, set] = {}
        self.id_to_ngrams: Dict[int, set] = {}
        self._num_docs = 0
        self._idf: Dict[str, float] = {}
        self._frozen = False

    # -- building --------------------------------------------------------

    def add(self, item_id: int, text: str):
        ngrams = generate_ngrams(text, self.n)
        self.add_precomputed(item_id, ngrams)

    def add_precomputed(self, item_id: int, ngrams: set):
        self.id_to_ngrams[item_id] = ngrams
        self._num_docs += 1
        for ng in ngrams:
            self.ngram_to_ids.setdefault(ng, set()).add(item_id)
        self._frozen = False

    def freeze(self):
        """Pre-compute IDF weights.  Call once after all documents are added."""
        N = self._num_docs
        if N == 0:
            self._frozen = True
            return
        for ng, ids in self.ngram_to_ids.items():
            df = len(ids)
            # smooth IDF: log((N + 1) / (df + 1)) + 1
            self._idf[ng] = math.log((N + 1) / (df + 1)) + 1.0
        self._frozen = True

    def build_vectors(self) -> Tuple[Dict[int, Dict[str, float]], Dict[int, float]]:
        """Pre-compute per-document TF-IDF sparse vectors and their L2 norms.

        Returns ``(id -> {ngram: weight}, id -> l2_norm)`` so scoring loops
        can skip re-vectorising master strings.
        """
        if not self._frozen:
            raise RuntimeError("Call freeze() before build_vectors()")
        vecs: Dict[int, Dict[str, float]] = {}
        norms: Dict[int, float] = {}
        idf = self._idf
        for item_id, ngrams in self.id_to_ngrams.items():
            if not ngrams:
                vecs[item_id] = {}
                norms[item_id] = 0.0
                continue
            tf = 1.0 / len(ngrams)
            v = {ng: tf * idf.get(ng, 1.0) for ng in ngrams}
            vecs[item_id] = v
            norms[item_id] = math.sqrt(sum(x * x for x in v.values()))
        return vecs, norms

    # -- querying --------------------------------------------------------

    def _tfidf_vec(self, ngrams: set) -> Dict[str, float]:
        """Return a sparse TF-IDF vector for the given n-gram set."""
        tf = 1.0 / len(ngrams) if ngrams else 0.0  # uniform TF (binary)
        return {ng: tf * self._idf.get(ng, 1.0) for ng in ngrams}

    def similarity(self, text1: str, text2: str) -> float:
        """
        Cosine similarity of TF-IDF weighted n-gram vectors.
        Returns 0.0 – 1.0.
        """
        if not self._frozen:
            raise RuntimeError("Call freeze() before computing similarity")

        ng1 = generate_ngrams(text1, self.n)
        ng2 = generate_ngrams(text2, self.n)
        if not ng1 or not ng2:
            return 0.0

        v1 = self._tfidf_vec(ng1)
        v2 = self._tfidf_vec(ng2)

        # Dot product (only on shared keys)
        shared = set(v1) & set(v2)
        if not shared:
            return 0.0

        dot = sum(v1[k] * v2[k] for k in shared)
        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


class UnionFind:
    """Union-Find data structure for clustering."""

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def get_clusters(self) -> dict:
        """Return a dict mapping root → list of member indices."""
        clusters = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            clusters.setdefault(root, []).append(i)
        return clusters


