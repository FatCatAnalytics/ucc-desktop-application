"""Engine configuration.

``PipelineConfig`` mirrors the Dagster ``resources.PipelineConfig`` field-for-field
(the generated core reads them by attribute), plus a handful of engine-only
fields that control *where* the local models and caches live.

Inference is always local and pip-installed - there is no network backend.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

APP_NAME = "ucc-engine"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def app_data_dir() -> Path:
    """Per-user writable directory for runs, caches and downloaded models."""
    override = os.environ.get("UCC_ENGINE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_NAME


@dataclass
class PipelineConfig:
    # --- per-run file paths (set by the runner from the DataSource) ---------
    ucc_file: str = ""
    master_file: str = ""
    universe_file: str = ""
    output_dir: str = ""

    # --- models (local only) --------------------------------------------------
    # "auto": fastembed if installed, else sentence-transformers.
    embedding_backend: str = "auto"            # auto | fastembed | sentence-transformers
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # Stages 5-7 (cluster the unmatched rows and suggest a mapping per cluster).
    # Off by default: unmatched rows are still labelled NEW_ENTITY and exported,
    # just without cluster ids / suggestions, and the run is much shorter.
    clustering_enabled: bool = False

    # Offline model folder (local path or network share / UNC path) in the cache
    # layout written by `ucc-engine prefetch-models`; holds both models. When
    # set, files are only read from here and nothing is downloaded.
    model_dir: str = ""
    # Network share holding the models (same cache layout). Before a model is
    # loaded, anything missing or newer on the share is *copied* into
    # ``model_cache_dir`` and the local copy is used - the share is read once
    # per PC, not once per session, and a dropped share does not break a run
    # that already has the files locally. Nothing is ever downloaded when set.
    model_share_dir: str = ""
    model_cache_dir: str = field(default_factory=lambda: os.environ.get("UCC_MODELS_DIR") or str(app_data_dir() / "models"))
    model_threads: int = 0                      # 0 = library default

    # --- matching thresholds --------------------------------------------------
    fuzzy_auto_threshold: int = 90  # see resources.py / tests/test_match_quality.py
    fuzzy_review_threshold: int = 85
    fuzzy_min_threshold: int = 80
    embed_auto_threshold: float = 0.95
    embed_high_threshold: float = 0.90
    embed_medium_threshold: float = 0.85
    embed_top_k: int = 20

    # --- clustering -----------------------------------------------------------
    cluster_sim_threshold: float = 0.88
    cluster_merge_threshold: float = 0.88
    cluster_k_neighbors: int = 20
    cluster_fuzzy_guard: int = 80
    cluster_merge_iterations: int = 1
    cluster_merge_k: int = 10
    cluster_max_size: int = 500

    # --- processing -----------------------------------------------------------
    embed_batch_size: int = 256
    search_batch_size: int = 4096
    fuzzy_batch_size: int = 1024
    fuzzy_workers: int = -1
    cpu_threads: int = -1
    embedding_workers: int = 4
    early_stop_threshold: float = 0.98
    max_candidates_per_query: int = 50

    # --- lead-bank designation (Stage 8) -------------------------------------
    lead_bank_threshold: float = 0.66
    filing_type_column: str = ""
    debtor_company_column: str = ""

    # --- clustering performance ----------------------------------------------
    cluster_memmap_threshold_mb: int = 512
    cluster_hnsw_threshold: int = 200000
    cluster_hnsw_m: int = 32
    cluster_hnsw_ef_search: int = 128
    cluster_hnsw_ef_construction: int = 200

    # --- stage 4 advanced match ----------------------------------------------
    advanced_use_per_state_faiss: bool = True
    advanced_per_state_min_size: int = 64
    query_embed_cache_enabled: bool = True
    query_embed_cache_path: str = field(
        default_factory=lambda: str(app_data_dir() / "cache" / "query_embeddings")
    )
    advanced_cross_encoder_enabled: bool = True
    advanced_cross_encoder_model: str = DEFAULT_CROSS_ENCODER
    advanced_cross_encoder_top_n: int = 10
    advanced_cross_encoder_promote_to_auto: float = 1.01  # review-only; see resources.py
    advanced_cross_encoder_promote_to_high: float = 0.80

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_overrides(cls, overrides: Dict[str, Any] | None = None) -> "PipelineConfig":
        """Defaults + only *known* overrides, coerced to the declared type."""
        type_by_name = {f.name: f.type for f in fields(cls)}
        clean: Dict[str, Any] = {}
        for key, raw in (overrides or {}).items():
            if key not in type_by_name or raw is None or raw == "":
                continue
            clean[key] = _coerce(type_by_name[key], raw)
        return cls(**clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def schema(cls) -> list[dict[str, Any]]:
        """Field metadata for the UI's settings form."""
        defaults = cls()
        return [
            {"name": f.name, "type": str(f.type).replace("<class '", "").replace("'>", ""),
             "default": getattr(defaults, f.name)}
            for f in fields(cls)
        ]


def _coerce(declared: Any, value: Any) -> Any:
    t = str(declared)
    if "bool" in t:
        return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
    if "int" in t:
        return int(value)
    if "float" in t:
        return float(value)
    return str(value)
