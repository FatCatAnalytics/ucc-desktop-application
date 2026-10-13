"""In-process pipeline runner with progress, log streaming, cancellation and results.

A ``PipelineRun`` executes the stages in ``registry.PIPELINE`` on a worker
thread. State is snapshot-able at any time (for the HTTP API) and persisted to
``<run_dir>/run.json`` so finished runs survive an engine restart.
"""
from __future__ import annotations

import gc
import json
import math
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .compat import AssetExecutionContext
from .config import PipelineConfig, app_data_dir
from .datasources import DataSource, SourceSpec, make_datasource, public_spec
from .decisions import DECISIONS, DecisionStore, norm_name
from .registry import KEEP_AFTER_RUN, PIPELINE, Stage, is_enabled, run_stage, total_weight

REVIEW_BANDS = ("REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW")
REVIEW_COLUMNS = [
    "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_NORMALIZED", "SECP_NAME_MAPPED",
    "matched_master_raw", "match_method", "match_score", "confidence",
    "ambiguous_master_candidates", "Firm_ID", "cluster_id", "cluster_label", "cluster_size",
    "suggested_mapping", "suggestion_score", "suggestion_confidence", "alternative_suggestions",
]
# ``_source`` values a reviewer (not the pipeline) puts into the updated master.
REVIEWER_SOURCES = ("reviewed_approved", "mapped_by_hand")


def decision_source(d: Dict[str, Any]) -> str:
    """Provenance tag for an approved decision: a bank the reviewer picked for an
    unmatched filer is ``mapped_by_hand``; approving the pipeline's suggestion
    (incl. decisions inherited from one) is ``reviewed_approved``."""
    return "mapped_by_hand" if str(d.get("match_method") or "").startswith("Manual") else "reviewed_approved"

BUCKETS = ("direct", "fuzzy_high", "review", "unmatched")
BUCKET_LABELS = {
    "direct": "Direct match",
    "fuzzy_high": "Fuzzy match · high probability",
    "review": "Review",
    "unmatched": "Unmatched (too low)",
}
BUCKET_COLUMNS = [
    "DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_NORMALIZED", "SECP_NAME_MAPPED",
    "matched_master_raw", "match_method", "match_score", "confidence", "Firm_ID",
]
MAX_LOG_LINES = 5000
MASTER_KEY = ("DEBT_STATE", "SECP_STATE", "SECP_NAME")
MASTER_COLUMNS = ("DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED", "Firm ID")
DECISION_COLUMNS = ("key", "decision", "decided_mapped_name", "decided_firm_id", "previously_rejected", "decided_in_run",
                    "decision_source", "filings")


def master_key(debt_state: Any, secp_state: Any, secp_name: Any) -> str:
    """Stable id for a review row = the updated master's own key, normalised like the core does."""
    def st(v: Any) -> str:
        return "" if v is None or (isinstance(v, float) and v != v) else str(v).strip().upper()
    return "|".join([st(debt_state), st(secp_state), "" if secp_name is None else str(secp_name)])


def bucket_series(df: pd.DataFrame) -> pd.Series:
    """Classify every result row into one of ``BUCKETS``.

    * ``direct``     - an exact match (raw or normalised) the pipeline accepted (AUTO)
    * ``fuzzy_high`` - a fuzzy / embedding match above the auto threshold (AUTO); added
                       to the updated master without review
    * ``review``     - any REVIEW_* band (fuzzy or embedding below auto, or an
                       ambiguous exact match): a suggestion for a human to decide
    * ``unmatched``  - nothing above the floor; left as-is (NEW_ENTITY / no state pair)
    """
    conf = df["confidence"].astype("string").fillna("")
    method = df["match_method"].astype("string").fillna("") if "match_method" in df.columns else pd.Series("", index=df.index, dtype="string")
    out = pd.Series("unmatched", index=df.index, dtype="object")
    auto = conf.eq("AUTO")
    out[auto & method.str.startswith("Exact Match")] = "direct"
    out[auto & ~method.str.startswith("Exact Match")] = "fuzzy_high"
    out[conf.isin(REVIEW_BANDS)] = "review"
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Make pandas/numpy scalars JSON-serialisable; NaN -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            v = value.item()
            return None if isinstance(v, float) and math.isnan(v) else v
        except Exception:  # noqa: BLE001
            return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def frame_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{k: _json_safe(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


@dataclass
class StageState:
    name: str
    label: str
    group: str
    status: str = "pending"  # pending | running | done | failed | skipped
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    seconds: Optional[float] = None
    rows: Optional[int] = None


@dataclass
class LogLine:
    seq: int
    ts: str
    level: str
    stage: str
    message: str


class RunCancelled(Exception):
    pass


class PipelineRun:
    def __init__(
        self,
        source_spec: SourceSpec,
        config: PipelineConfig,
        *,
        runs_root: Path,
        sample_rows: int = 0,
        run_id: Optional[str] = None,
        name: str = "",
        decisions: Optional[DecisionStore] = None,
    ):
        self.id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        self.name = name or self.id
        self.created_at = _now()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.status = "queued"  # queued | running | succeeded | failed | cancelled
        self.error: Optional[str] = None
        self.sample_rows = int(sample_rows or 0)

        self.run_dir = runs_root / self.id
        self.output_dir = self.run_dir / "outputs"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.source_spec = source_spec
        self.source: DataSource = make_datasource(source_spec)
        self.config = config
        self.config.output_dir = str(self.output_dir)
        self.input_rows: Dict[str, int] = {}

        self.stages: List[StageState] = [StageState(s.name, s.label, s.group) for s in PIPELINE]
        self.current_stage: Optional[str] = None
        self._logs: List[LogLine] = []
        self._seq = 0
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cache: Dict[str, Any] = {}
        self.summary: Optional[Dict[str, Any]] = None
        # Reviewer decisions live in one SQLite store shared by every run, so a key
        # approved or rejected once is never asked about again.
        self.decisions = decisions or DecisionStore.at(runs_root.parent / "decisions.sqlite")
        self._legacy_decisions_path = self.run_dir / "decisions.json"
        self._annotated: Optional[tuple[int, int, pd.DataFrame]] = None  # (store.version, id(enriched), frame)
        self._universe_lookup: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ logging
    def _log(self, level: str, message: str, stage: str = "engine") -> None:
        with self._lock:
            self._seq += 1
            self._logs.append(LogLine(self._seq, _now(), level, stage, message))
            if len(self._logs) > MAX_LOG_LINES:
                del self._logs[: len(self._logs) - MAX_LOG_LINES]

    def logs(self, after: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            return [asdict(l) for l in self._logs if l.seq > after][:limit]

    # ------------------------------------------------------------------ control
    def start(self) -> "PipelineRun":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"ucc-run-{self.id}", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()
        self._log("WARNING", "Cancellation requested; the run stops after the current stage.")

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # --------------------------------------------------------------------- run
    def _run(self) -> None:
        self.status, self.started_at = "running", _now()
        self._persist()
        try:
            self._log("INFO", f"Materialising inputs from {self.source.kind} source"
                              + (f" (sample {self.sample_rows:,} UCC rows)" if self.sample_rows else ""))
            inputs = self.source.materialize(self.run_dir, self.sample_rows)
            self.config.ucc_file = inputs.ucc_file
            self.config.master_file = inputs.master_file
            self.config.universe_file = inputs.universe_file or str(self.run_dir / "inputs" / "__no_universe__.csv")
            self.input_rows = inputs.rows
            self._log("INFO", "Inputs: " + ", ".join(f"{k}={v:,}" for k, v in inputs.rows.items()))

            for stage, state in zip(PIPELINE, self.stages):
                if self._cancel.is_set():
                    raise RunCancelled()
                self._exec_stage(stage, state)

            self.summary = self._build_summary()
            self.status = "succeeded"
            self._carry_forward_approvals()
            self._log("INFO", "Run complete.")
        except RunCancelled:
            self.status = "cancelled"
            for st in self.stages:
                if st.status == "pending":
                    st.status = "skipped"
        except Exception as e:  # noqa: BLE001
            self.status, self.error = "failed", f"{type(e).__name__}: {e}"
            self._log("ERROR", self.error)
            self._log("DEBUG", traceback.format_exc())
            for st in self.stages:
                if st.status == "running":
                    st.status = "failed"
                elif st.status == "pending":
                    st.status = "skipped"
        finally:
            self.finished_at = _now()
            self.current_stage = None
            self._release_memory()
            self._persist()

    def _exec_stage(self, stage: Stage, state: StageState) -> None:
        if not is_enabled(stage, self.config):
            run_stage(stage, self.config, None, self._cache)  # pass-through, keeps the DAG wiring intact
            state.status = "skipped"
            self._log("INFO", f"– {stage.label} skipped ({stage.enabled_by} = false)", stage.name)
            return
        state.status, state.started_at = "running", _now()
        self.current_stage = stage.name
        t0 = time.perf_counter()
        ctx = AssetExecutionContext(stage.name, sink=lambda lvl, msg: self._log(lvl, msg, stage.name))
        self._log("INFO", f"▶ {stage.label}", stage.name)
        try:
            result = run_stage(stage, self.config, ctx, self._cache)
        except Exception:
            state.status, state.finished_at = "failed", _now()
            state.seconds = round(time.perf_counter() - t0, 2)
            raise
        state.seconds = round(time.perf_counter() - t0, 2)
        state.finished_at, state.status = _now(), "done"
        if isinstance(result, pd.DataFrame):
            state.rows = int(len(result))
        self._log("INFO", f"✓ {stage.label} in {state.seconds:.1f}s"
                          + (f" · {state.rows:,} rows" if state.rows is not None else ""), stage.name)
        self._persist()

    def _carry_forward_approvals(self) -> None:
        try:
            self.propagate_decisions()
        except Exception as e:  # noqa: BLE001
            self._log("WARNING", f"Could not propagate decisions: {e}")
        """Approvals from earlier runs that the master still lacks go straight into this run's updated master."""
        try:
            n = self.decision_counts()["unapplied_approved"]
            if n:
                self._log("INFO", f"Carrying forward {n:,} previously approved review rows into the updated master")
                self.apply_decisions()
        except Exception as e:  # noqa: BLE001 - never fail a finished run over this
            self._log("WARNING", f"Could not carry forward approved decisions: {e}")

    def _release_memory(self) -> None:
        self._cache = {k: v for k, v in self._cache.items() if k in KEEP_AFTER_RUN}
        gc.collect()

    # ---------------------------------------------------------------- snapshot
    def progress(self) -> float:
        done = sum(s.weight for s, st in zip(PIPELINE, self.stages) if st.status == "done")
        return round(done / total_weight(self.config), 4)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "progress": self.progress(),
                "current_stage": self.current_stage,
                "sample_rows": self.sample_rows,
                "input_rows": self.input_rows,
                "source": public_spec(self.source_spec),
                "run_dir": str(self.run_dir),
                "output_dir": str(self.output_dir),
                "stages": [asdict(s) for s in self.stages],
                "summary": self.summary,
                "config": self.config.to_dict(),
            }

    def _persist(self) -> None:
        data = self.snapshot()
        data["log_tail"] = self.logs(after=max(0, self._seq - 300))
        tmp = self.run_dir / "run.json.tmp"
        tmp.write_text(json.dumps(data, indent=1, default=str))
        tmp.replace(self.run_dir / "run.json")

    # ----------------------------------------------------------------- results
    def _enriched(self) -> Optional[pd.DataFrame]:
        df = self._cache.get("universe_enriched_results")
        if isinstance(df, pd.DataFrame):
            if df.attrs:
                # The core parks the model + cluster embeddings in df.attrs, and pandas
                # deep-copies attrs into every derived Series - a groupby over a few
                # thousand clusters took >10 s because of it. Serve a view without attrs.
                df = df.copy(deep=False)
                df.attrs = {}
                self._cache["universe_enriched_results"] = df
            return df
        # After a restart: rebuild from the full_results export.
        files = sorted(self.output_dir.glob("full_results_*.csv"))
        if files:
            df = pd.read_csv(files[-1], low_memory=False)
            self._cache["universe_enriched_results"] = df
            return df
        return None

    def _build_summary(self) -> Dict[str, Any]:
        df = self._enriched()
        if df is None:
            return {}
        conf = df["confidence"].fillna("UNKNOWN")
        method = df["match_method"].fillna("Unmatched")
        new_mask = conf.eq("NEW_ENTITY")
        clusters = df.loc[new_mask, "cluster_id"].dropna() if "cluster_id" in df.columns else pd.Series(dtype=float)
        out: Dict[str, Any] = {
            "total_records": int(len(df)),
            "auto_mapped": int(conf.eq("AUTO").sum()),
            "review_needed": int(conf.isin(REVIEW_BANDS).sum()),
            "new_entities": int(new_mask.sum()),
            "no_master_state_pair": int(conf.eq("NO_MASTER_STATE_PAIR").sum()),
            "by_match_method": {str(k): int(v) for k, v in method.value_counts().items()},
            "by_confidence": {str(k): int(v) for k, v in conf.value_counts().items()},
            "clusters": int(clusters.nunique()),
            "avg_cluster_size": round(float(new_mask.sum() / clusters.nunique()), 1) if clusters.nunique() else 0.0,
            "records_with_firm_id": int(df["Firm_ID"].notna().sum()) if "Firm_ID" in df.columns else 0,
            "unique_firm_ids": int(df["Firm_ID"].nunique()) if "Firm_ID" in df.columns else 0,
        }
        bvc = bucket_series(df).value_counts()
        out["buckets"] = {b: int(bvc.get(b, 0)) for b in BUCKETS}
        # Review split by band, so the UI can show high / medium separately.
        out["review_bands"] = {b: int(conf.eq(b).sum()) for b in REVIEW_BANDS}
        um = self._cache.get("updated_master_data")
        if isinstance(um, pd.DataFrame):
            src = um["_source"].astype(str) if "_source" in um.columns else pd.Series(dtype=str)
            out["updated_master"] = {
                "rows": int(len(um)),
                "original_rows": int(src.eq("original_master").sum()),
                "added_rows": int(src.str.startswith("pipeline_").sum()),
                "approved_rows": int(src.isin(REVIEWER_SOURCES).sum()),
                "by_source": {str(k): int(v) for k, v in src.value_counts().items()},
                "file": self._latest("updated_master_*.csv", exclude="_with_source"),
            }
        lb = self._cache.get("lead_bank_designation")
        if isinstance(lb, pd.DataFrame):
            out["lead_bank"] = {"rows": int(len(lb)), "columns": [str(c) for c in lb.columns][:12]}
        out["stage_seconds"] = {s.name: s.seconds for s in self.stages if s.seconds is not None}
        out["total_seconds"] = round(sum(s.seconds or 0 for s in self.stages), 1)
        return out

    def _latest(self, pattern: str, exclude: str = "") -> Optional[str]:
        files = [p for p in sorted(self.output_dir.glob(pattern)) if not exclude or exclude not in p.name]
        return str(files[-1]) if files else None

    def review(
        self,
        *,
        confidence: Optional[Iterable[str]] = None,
        search: str = "",
        method: str = "",
        offset: int = 0,
        limit: int = 100,
        sort: str = "match_score",
        descending: bool = True,
        decision: str = "",
        mapped: Optional[str] = None,
    ) -> Dict[str, Any]:
        df = self._annotated_review()
        bands = list(confidence) if confidence else list(REVIEW_BANDS)
        sub = df[df["confidence"].isin(bands)]
        if method:
            sub = sub[sub["match_method"].astype(str).str.contains(method, case=False, na=False)]
        if mapped is not None:
            final = sub["decided_mapped_name"].where(sub["decided_mapped_name"].notna(), sub["SECP_NAME_MAPPED"]).fillna("").astype(str)
            sub = sub[final.eq(mapped)]
        if search:
            s = search.strip()
            mask = sub["SECP_NAME"].astype(str).str.contains(s, case=False, na=False, regex=False)
            if "SECP_NAME_MAPPED" in sub.columns:
                mask |= sub["SECP_NAME_MAPPED"].astype(str).str.contains(s, case=False, na=False, regex=False)
            if "suggested_mapping" in sub.columns:
                mask |= sub["suggested_mapping"].astype(str).str.contains(s, case=False, na=False, regex=False)
            sub = sub[mask]
        if decision:
            sub = sub[sub["decision"].eq(decision)]
        if sort in sub.columns:
            sub = sub.sort_values(sort, ascending=not descending, na_position="last")
        cols = [c for c in REVIEW_COLUMNS if c in sub.columns] + list(DECISION_COLUMNS)
        rows = frame_records(sub.iloc[offset: offset + limit][cols])
        for row in rows:
            if row["decision"] == "pending":
                row["decision"] = None
        return {"total": int(len(sub)), "offset": offset, "limit": limit, "columns": cols,
                "rows": rows, "decision_counts": self.decision_counts()}

    def rows(
        self,
        *,
        bucket: str,
        search: str = "",
        offset: int = 0,
        limit: int = 100,
        sort: str = "match_score",
        descending: bool = True,
        decision: str = "",
    ) -> Dict[str, Any]:
        """Page the result rows of one bucket (see ``bucket_series``) with any decisions
        attached. ``decision`` = ``pending`` (nothing decided) / ``mapped`` (a decision
        exists) narrows the page; the ``review`` bucket has its own endpoint."""
        if bucket not in BUCKETS:
            raise KeyError(f"unknown bucket {bucket!r}; expected one of {BUCKETS}")
        df = self._enriched()
        if df is None:
            return {"total": 0, "offset": offset, "limit": limit, "columns": BUCKET_COLUMNS, "rows": [], "mapped_total": 0}
        sub = self._bucket_rows(df, bucket)
        decided_all = self.decisions.get_many(sub["key"])
        mapped_total = int(sub["key"].isin(decided_all.keys()).sum())
        if decision == "pending":
            sub = sub[~sub["key"].isin(decided_all.keys())]
        elif decision == "mapped":
            sub = sub[sub["key"].isin(decided_all.keys())]
        if search:
            s = search.strip()
            mask = sub["SECP_NAME"].astype(str).str.contains(s, case=False, na=False, regex=False)
            if "SECP_NAME_MAPPED" in sub.columns:
                mask |= sub["SECP_NAME_MAPPED"].astype(str).str.contains(s, case=False, na=False, regex=False)
            sub = sub[mask]
        if sort in sub.columns:
            sub = sub.sort_values(sort, ascending=not descending, na_position="last")
        elif bucket == "unmatched":
            sub = sub.sort_values("filings", ascending=False)
        cols = [c for c in BUCKET_COLUMNS if c in sub.columns] + ["key", "filings"]
        page = sub.iloc[offset: offset + limit]
        rows = frame_records(page[cols])
        # Attach decisions (manual mappings on unmatched rows, approvals on review rows).
        found = self.decisions.get_many(page["key"])
        for row in rows:
            d = found.get(row["key"])
            row["decision"] = d["decision"] if d else None
            row["decided_mapped_name"] = (d.get("mapped_name") or d.get("suggested")) if d else None
            row["decided_firm_id"] = d.get("firm_id") if d else None
            row["decision_source"] = d.get("source") if d else None
            row["decided_in_run"] = d.get("run_id") if d else None
        return {"total": int(len(sub)), "offset": offset, "limit": limit, "columns": cols + ["decision"],
                "rows": rows, "decision_counts": self.decision_counts(), "mapped_total": mapped_total}

    def _bucket_rows(self, df: pd.DataFrame, bucket: str) -> pd.DataFrame:
        """One row per key in a bucket, with ``key`` and ``filings`` (how many filings share it)."""
        sub = df[bucket_series(df).eq(bucket)]
        keys = self._keys(sub)
        filings = keys.map(keys.value_counts())
        return sub.assign(key=keys.values, filings=filings.values).drop_duplicates("key")

    def banks(self) -> List[Dict[str, Any]]:
        """Every bank a filer can be mapped to by hand: the Universe (name + Firm ID)
        plus the banks the master already maps to."""
        self._resolve_firm_id("_")  # loads the universe lookup (returns None for an unknown name)
        out: Dict[str, Dict[str, Any]] = {}
        names = getattr(self, "_universe_names", None) or {}
        for name, fid in (self._universe_lookup or {}).items():
            out[norm_name(name)] = {"name": names.get(name, name), "firm_id": fid, "source": "universe"}
        # prefer the master's own spelling of the bank name where it has one
        df = self._enriched()
        um = self._master_with_source()
        for frame in (um, df):
            if frame is None or "SECP_NAME_MAPPED" not in frame.columns:
                continue
            fid_col = "Firm ID" if "Firm ID" in frame.columns else "Firm_ID" if "Firm_ID" in frame.columns else None
            pairs = frame[["SECP_NAME_MAPPED", fid_col]].dropna(subset=["SECP_NAME_MAPPED"]).drop_duplicates("SECP_NAME_MAPPED") if fid_col else None
            if pairs is None:
                continue
            for name, fid in zip(pairs["SECP_NAME_MAPPED"], pairs[fid_col]):
                k = norm_name(name)
                if not k:
                    continue
                cur = out.get(k)
                out[k] = {"name": str(name).strip(), "firm_id": _json_safe(fid) if fid == fid else (cur or {}).get("firm_id"),
                          "source": "master" if cur is None else cur["source"]}
        return sorted(out.values(), key=lambda b: b["name"].lower())

    def master_rows(self, *, source: str = "added", search: str = "", offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        """Page the updated master: ``source`` = ``original`` (came in as the master),
        ``added`` (everything this run or a reviewer added) or one exact ``_source`` value."""
        um = self._master_with_source()
        if um is None:
            return {"total": 0, "offset": offset, "limit": limit, "columns": [], "rows": [], "by_source": {}}
        src = um["_source"].astype(str) if "_source" in um.columns else pd.Series("original_master", index=um.index)
        by_source = {str(k): int(v) for k, v in src.value_counts().items()}
        if source == "original":
            sub = um[src.eq("original_master")]
        elif source == "added":
            sub = um[~src.eq("original_master")]
        else:
            sub = um[src.eq(source)]
        if search.strip():
            q = search.strip()
            mask = sub["SECP_NAME"].astype(str).str.contains(q, case=False, na=False, regex=False)
            mask |= sub["SECP_NAME_MAPPED"].astype(str).str.contains(q, case=False, na=False, regex=False)
            sub = sub[mask]
        cols = [c for c in (*MASTER_COLUMNS, "_source") if c in sub.columns]
        return {"total": int(len(sub)), "offset": offset, "limit": limit, "columns": cols,
                "rows": frame_records(sub.iloc[offset: offset + limit][cols]), "by_source": by_source}

    # --------------------------------------------------------------- decisions
    @staticmethod
    def _keys(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [master_key(a, b, c) for a, b, c in zip(df["DEBT_STATE"], df["SECP_STATE"], df["SECP_NAME"])],
            index=df.index, dtype="object",
        )

    def _annotated_review(self) -> pd.DataFrame:
        """Review-band rows joined with the decision store.

        ``decision`` is ``pending`` / ``approved`` / ``rejected``. An approval always
        sticks to its key. A rejection only sticks while the pipeline keeps making the
        *same* suggestion; a different suggestion comes back as pending with
        ``previously_rejected`` naming the one that was turned down.
        """
        df = self._enriched()
        if df is None:
            return pd.DataFrame(columns=[*REVIEW_COLUMNS, *DECISION_COLUMNS])
        if self._annotated and self._annotated[0] == self.decisions.version and self._annotated[1] == id(df):
            return self._annotated[2]
        sub = df[df["confidence"].isin(REVIEW_BANDS)].copy(deep=False)
        # One review row per key: the same filer in the same state pair may appear
        # on many filings, and the reviewer decides once. ``filings`` keeps the volume.
        all_keys = self._keys(sub)
        filings = all_keys.map(all_keys.value_counts())
        sub = sub.assign(_key=all_keys.values, filings=filings.values).drop_duplicates("_key")
        keys = sub.pop("_key")
        found = self.decisions.get_many(keys)
        status, mapped, firm, prev, in_run, source = [], [], [], [], [], []
        suggestions = sub["SECP_NAME_MAPPED"] if "SECP_NAME_MAPPED" in sub.columns else pd.Series(None, index=sub.index)
        for key, suggested in zip(keys, suggestions):
            d = found.get(key)
            if d is None:
                status.append("pending"); mapped.append(None); firm.append(None); prev.append(None); in_run.append(None); source.append(None)
            elif d["decision"] == "rejected" and d.get("suggested") and norm_name(d["suggested"]) != norm_name(suggested):
                status.append("pending"); mapped.append(None); firm.append(None)
                prev.append(d["suggested"]); in_run.append(d.get("run_id")); source.append(None)
            else:
                status.append(d["decision"]); mapped.append(d.get("mapped_name")); firm.append(d.get("firm_id"))
                prev.append(None); in_run.append(d.get("run_id")); source.append(d.get("source"))
        sub["key"], sub["decision"], sub["decided_mapped_name"] = keys.values, status, mapped
        sub["decided_firm_id"], sub["previously_rejected"], sub["decided_in_run"] = firm, prev, in_run
        sub["decision_source"] = source
        self._annotated = (self.decisions.version, id(df), sub)
        return sub

    # ------------------------------------------------------------ propagation
    @staticmethod
    def _norm(name: Any) -> str:
        from .core.utils import normalize_name

        return normalize_name("" if name is None else str(name))

    def _final_name(self, d: Dict[str, Any]) -> str:
        return norm_name(d.get("mapped_name") or d.get("suggested"))

    def propagate_decisions(self) -> Dict[str, int]:
        """Inherit decisions across state pairs and spellings of the same filer.

        A pending review row whose *normalised* filer name was already decided
        elsewhere - another state pair, a typo variant, an earlier run - and
        whose suggested bank is the same bank, gets that decision written for
        its own key with ``source = 'inherited:<key>'`` (audited like any other
        decision; undo works per key). Rows suggesting a *different* bank stay
        pending: that is how PINNACLE BANK (GA) does not inherit an approval
        meant for Pinnacle Bank (NE). Reviewer decisions outrank inherited ones.
        """
        if self.status in ("queued", "running") and threading.current_thread() is not self._thread:
            raise RuntimeError("The run is still in progress")
        # Backfill normalised names for decisions written before the column existed.
        missing = self.decisions.missing_norm_names()
        if missing:
            self.decisions.set_norm_names((m["key"], self._norm(m["secp_name"])) for m in missing)
        review = self._annotated_review()
        cleared = self.decisions.cleared_keys()
        pending = review[review["decision"].eq("pending") & ~review["key"].isin(cleared)]
        # Unmatched rows inherit only from manual mappings made "for all states".
        enriched = self._enriched()
        unmatched = self._bucket_rows(enriched, "unmatched") if enriched is not None else None
        if unmatched is not None and len(unmatched):
            decided = self.decisions.get_many(unmatched["key"])
            unmatched = unmatched[~unmatched["key"].isin(decided.keys()) & ~unmatched["key"].isin(cleared)]
        else:
            unmatched = None
        if pending.empty and (unmatched is None or unmatched.empty):
            return {"approved": 0, "rejected": 0, "candidates": 0}
        norms = pending["SECP_NAME"].map(self._norm)
        unorms = unmatched["SECP_NAME"].map(self._norm) if unmatched is not None else pd.Series(dtype=object)
        pool = self.decisions.by_norm_name(list(norms.unique()) + list(unorms.unique()))
        out = {"approved": 0, "rejected": 0, "candidates": int(sum(1 for n in norms if n in pool) + sum(1 for n in unorms if n in pool))}
        # pending review rows first ...
        work: List[tuple] = [(row, nrm, False) for (_, row), nrm in zip(pending.iterrows(), norms)]
        if unmatched is not None:
            work += [(row, nrm, True) for (_, row), nrm in zip(unmatched.iterrows(), unorms)]
        for row, nrm, is_unmatched in work:
            if is_unmatched:
                matches = [d for d in pool.get(nrm, []) if d["decision"] == "approved"
                           and str(d.get("match_method") or "").startswith("Manual (all states)")]
                suggested = None
            else:
                suggested = _json_safe(row.get("SECP_NAME_MAPPED"))
                want = norm_name(suggested)
                if not want:
                    continue
                # reviewer-made first, then most recent (by_norm_name is already newest-first)
                matches = sorted((d for d in pool.get(nrm, []) if self._final_name(d) == want),
                                 key=lambda d: 0 if not d.get("source") else 1)
            if not matches:
                continue
            d = matches[0]
            origin = d["key"] if not d.get("source") else str(d["source"]).split(":", 1)[-1]
            firm_id = d.get("firm_id")
            if firm_id is None:
                firm_id = _json_safe(row.get("Firm_ID"))
            self.decisions.set(
                str(row["key"]), decision=d["decision"],
                mapped_name=d.get("mapped_name") or (d.get("suggested") if is_unmatched else None), firm_id=firm_id,
                suggested=suggested,
                match_method="Manual (all states)" if is_unmatched else _json_safe(row.get("match_method")),
                match_score=None if is_unmatched else _json_safe(row.get("match_score")),
                confidence=None if is_unmatched else _json_safe(row.get("confidence")),
                run_id=self.id, norm_name=nrm, source=f"inherited:{origin}",
            )
            out[d["decision"]] += 1
        if out["approved"] or out["rejected"]:
            self._log("INFO", f"Inherited {out['approved']:,} approvals and {out['rejected']:,} rejections "
                              f"from decisions on the same filer name elsewhere")
        return out

    def review_groups(self, *, decision: str = "pending", search: str = "") -> List[Dict[str, Any]]:
        """The review queue rolled up by suggested bank, biggest first - so a
        reviewer can clear "41 filers -> JPMorgan" in one go."""
        df = self._annotated_review()
        if df.empty:
            return []
        if decision:
            df = df[df["decision"].eq(decision)]
        if search.strip():
            s = search.strip()
            df = df[df["SECP_NAME"].astype(str).str.contains(s, case=False, na=False, regex=False)
                    | df["SECP_NAME_MAPPED"].astype(str).str.contains(s, case=False, na=False, regex=False)]
        bank = df["decided_mapped_name"].where(df["decided_mapped_name"].notna(), df["SECP_NAME_MAPPED"]).fillna("").astype(str)
        g = df.assign(_bank=bank).groupby("_bank", sort=False)
        rows = []
        for name, grp in g:
            fid = grp["decided_firm_id"].dropna()
            if fid.empty and "Firm_ID" in grp.columns:
                fid = grp["Firm_ID"].dropna()
            vc = grp["decision"].value_counts()
            rows.append({
                "mapped_name": name, "firm_id": _json_safe(fid.iloc[0]) if len(fid) else None,
                "keys": int(len(grp)), "filings": int(grp["filings"].sum()),
                "pending": int(vc.get("pending", 0)), "approved": int(vc.get("approved", 0)), "rejected": int(vc.get("rejected", 0)),
                "methods": {str(k): int(v) for k, v in grp["match_method"].astype(str).value_counts().items()},
                "min_score": _json_safe(grp["match_score"].min()), "max_score": _json_safe(grp["match_score"].max()),
            })
        rows.sort(key=lambda r: (-r["filings"], -r["keys"], r["mapped_name"]))
        return rows

    def _master_keys(self) -> set:
        um = self._master_with_source()
        if um is None or um.empty:
            return set()
        cached = self._cache.get("updated_master_keys")
        if cached and cached[0] == id(um):
            return cached[1]
        keys = set(self._keys(um))
        self._cache["updated_master_keys"] = (id(um), keys)
        return keys

    def decision_counts(self) -> Dict[str, int]:
        """Decision status of *this run's* review rows, plus how many approvals the master still lacks."""
        annotated = self._annotated_review()
        vc = annotated["decision"].value_counts() if len(annotated) else {}
        counts = {k: int(vc.get(k, 0)) for k in ("approved", "rejected", "pending")}
        counts["total"] = int(len(annotated))
        master_keys = self._master_keys()
        counts["unapplied_approved"] = sum(1 for d in self.decisions.approved() if d["key"] not in master_keys)
        counts["store"] = self.decisions.counts()
        return counts

    def set_decisions(self, items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """Record approve / reject (or clear) for review rows of this run.

        Each item: ``{"key", "decision": "approved"|"rejected"|None, "mapped_name"?, "firm_id"?}``.
        ``mapped_name`` overrides the pipeline's suggestion; Firm ID is then re-resolved
        from the Universe (case-insensitive bank-name lookup, as in the pipeline) unless
        given explicitly. Decisions are global: they apply to every later run too.
        """
        if self.status in ("queued", "running"):
            raise RuntimeError("The run is still in progress")
        review = self._annotated_review()
        known = dict(zip(review["key"], review.index))
        enriched = self._enriched()
        unmatched = self._bucket_rows(enriched, "unmatched") if enriched is not None else None
        known_unmatched = dict(zip(unmatched["key"], unmatched.index)) if unmatched is not None else {}
        changed = 0
        manual_name_scope = 0
        items_list = list(items)
        for item in items_list:
            key = str(item.get("key", ""))
            is_manual = key not in known and key in known_unmatched
            if key not in known and not is_manual:
                raise KeyError(f"{key!r} is not a review-queue or unmatched row of this run")
            decision = item.get("decision")
            if decision is None:
                changed += self.decisions.clear(key, self.id)
                continue
            if decision not in DECISIONS:
                raise ValueError(f"decision must be one of {DECISIONS} or null, got {decision!r}")
            row = review.loc[known[key]] if not is_manual else unmatched.loc[known_unmatched[key]]
            suggested = None if is_manual else _json_safe(row.get("SECP_NAME_MAPPED"))
            mapped = (item.get("mapped_name") or "").strip() or None
            if is_manual and decision != "approved":
                raise ValueError(f"{key!r}: an unmatched row can only be mapped (approved with a bank), not rejected")
            if mapped is not None and mapped == (suggested or "").strip():
                mapped = None  # same as the suggestion -> not an override
            final_name = mapped or suggested
            if decision == "approved" and not final_name:
                raise ValueError(f"{key!r}: an approved row needs a mapped name")
            firm_id = item.get("firm_id")
            if firm_id in ("", None):  # universe lookup by final name; else keep the pipeline's value
                firm_id = self._resolve_firm_id(final_name)
                if firm_id is None and mapped is None:
                    firm_id = _json_safe(row.get("Firm_ID"))
            # A manual mapping made "for all states" propagates to the same filer name
            # wherever it is unmatched (also in later runs); a plain one sticks to its key.
            all_states = bool(item.get("all_states")) if is_manual else False
            method = ("Manual (all states)" if all_states else "Manual") if is_manual else _json_safe(row.get("match_method"))
            self.decisions.set(
                key, decision=decision, mapped_name=mapped, firm_id=_json_safe(firm_id), suggested=suggested,
                match_method=method, match_score=None if is_manual else _json_safe(row.get("match_score")),
                confidence=None if is_manual else _json_safe(row.get("confidence")), run_id=self.id,
                norm_name=self._norm(row.get("SECP_NAME")), source=None,
            )
            changed += 1
            manual_name_scope += int(all_states)
        decided = any(item.get("decision") in DECISIONS for item in items_list)
        inherited = self.propagate_decisions() if decided else {"approved": 0, "rejected": 0, "candidates": 0}
        return {"changed": changed, "inherited": inherited, "counts": self.decision_counts()}

    def _resolve_firm_id(self, mapped_name: Optional[str]) -> Any:
        """Universe lookup by (upper-cased) bank name - the pipeline's Pass 1 rule."""
        if not mapped_name:
            return None
        if self._universe_lookup is None:
            self._universe_lookup = {}
            path = Path(self.config.universe_file or "")
            if path.is_file():
                try:
                    from .datasources import detect_delimiter

                    uni = pd.read_csv(path, sep=detect_delimiter(str(path)), encoding="latin-1", low_memory=False)
                    name_col = next((c for c in ("Bank Name", "SECP_NAME") if c in uni.columns), None)
                    id_col = next((c for c in ("Firm ID", "Firm_ID") if c in uni.columns), None)
                    if name_col and id_col:
                        self._universe_names = {}
                        for name, fid in zip(uni[name_col], uni[id_col]):
                            if isinstance(name, str) and name.strip() and pd.notna(fid):
                                self._universe_lookup.setdefault(name.strip().upper(), _json_safe(fid))
                                self._universe_names.setdefault(name.strip().upper(), name.strip())
                except Exception as e:  # noqa: BLE001
                    self._log("WARNING", f"Could not read universe for Firm ID lookup: {e}")
        return self._universe_lookup.get(mapped_name.strip().upper())

    def _master_with_source(self) -> Optional[pd.DataFrame]:
        """The run's updated master incl. provenance, from memory or its CSV."""
        um = self._cache.get("updated_master_data")
        if isinstance(um, pd.DataFrame):
            return um
        path = self._latest("updated_master_*_with_source.csv")
        if path:
            um = pd.read_csv(path, encoding="latin-1", low_memory=False)
            self._cache["updated_master_data"] = um
            return um
        return None

    def apply_decisions(self) -> Dict[str, Any]:
        """Append every approved review row (from any run) to this run's updated master.

        Writes a fresh ``updated_master_<ts>.csv`` (+ ``_with_source``) so the
        newest file always reflects the reviewer's decisions; earlier files stay
        for audit. Original master and pipeline rows still win on a key clash.
        """
        if self.status in ("queued", "running") and threading.current_thread() is not self._thread:
            raise RuntimeError("The run is still in progress")
        base = self._master_with_source()
        if base is None:
            raise FileNotFoundError("This run has no updated master to extend")
        base = base[~base.get("_source", pd.Series("", index=base.index)).astype(str).isin(REVIEWER_SOURCES)]
        approved = self.decisions.approved()
        new = pd.DataFrame([{
            "DEBT_STATE": d["debt_state"], "SECP_STATE": d["secp_state"], "SECP_NAME": d["secp_name"],
            "SECP_NAME_MAPPED": d["mapped_name"] or d["suggested"], "Firm ID": d["firm_id"],
            "_source": decision_source(d),
        } for d in approved], columns=list(MASTER_COLUMNS) + ["_source"])
        combined = pd.concat([base, new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=list(MASTER_KEY), keep="first").reset_index(drop=True)
        fid = pd.to_numeric(combined["Firm ID"], errors="coerce")  # keep the master's integer formatting
        if (fid.dropna() % 1 == 0).all():
            combined["Firm ID"] = fid.astype("Int64")
        src = combined["_source"].astype(str)
        added = int(src.isin(REVIEWER_SOURCES).sum())
        by_hand = int(src.eq("mapped_by_hand").sum())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.output_dir / f"updated_master_{ts}.csv"
        if out.exists():  # same second as the pipeline's own file
            ts += "_reviewed"
            out = self.output_dir / f"updated_master_{ts}.csv"
        canonical = [c for c in MASTER_COLUMNS if c in combined.columns]
        combined[canonical].to_csv(out, index=False)
        combined.to_csv(self.output_dir / f"updated_master_{ts}_with_source.csv", index=False)
        self._cache["updated_master_data"] = combined
        self._cache.pop("updated_master_keys", None)
        self._log("INFO", f"Applied review decisions: {added:,} rows added -> {out.name} "
                          f"({added - by_hand:,} approved in review, {by_hand:,} mapped by hand; "
                          f"{len(approved) - added:,} already present under the same key)")
        if self.summary is not None:
            self.summary["updated_master"] = {
                "rows": int(len(combined)),
                "original_rows": int(src.eq("original_master").sum()),
                "added_rows": int(src.str.startswith("pipeline_").sum()),
                "approved_rows": added,
                "by_source": {str(k): int(v) for k, v in src.value_counts().items()},
                "file": str(out),
            }
        self._persist()
        return {"approved": len(approved), "added": added, "by_hand": by_hand, "file": str(out),
                "rows": int(len(combined)), "counts": self.decision_counts()}

    def clusters(self, *, search: str = "", min_size: int = 1, offset: int = 0, limit: int = 100) -> Dict[str, Any]:
        """New-entity clusters (Stage 5-7) summarised one row per cluster."""
        df = self._enriched()
        if df is None or "cluster_id" not in df.columns:
            return {"total": 0, "rows": [], "offset": offset, "limit": limit}
        sub = df[df["confidence"].eq("NEW_ENTITY") & df["cluster_id"].notna()]
        agg = (
            sub.groupby("cluster_id", sort=False)
            .agg(
                cluster_label=("cluster_label", "first"),
                cluster_size=("SECP_NAME", "size"),
                suggested_mapping=("suggested_mapping", "first") if "suggested_mapping" in sub.columns else ("SECP_NAME", "first"),
                suggestion_score=("suggestion_score", "max") if "suggestion_score" in sub.columns else ("SECP_NAME", "size"),
                suggestion_confidence=("suggestion_confidence", "first") if "suggestion_confidence" in sub.columns else ("SECP_NAME", "first"),
                states=("SECP_STATE", lambda s: ", ".join(sorted({str(x) for x in s.dropna()})[:6])),
                sample_names=("SECP_NAME", lambda s: " | ".join(list(dict.fromkeys(s.astype(str)))[:5])),
            )
            .reset_index()
        )
        agg = agg[agg["cluster_size"] >= max(1, min_size)]
        if search:
            s = search.strip()
            agg = agg[
                agg["cluster_label"].astype(str).str.contains(s, case=False, na=False, regex=False)
                | agg["sample_names"].astype(str).str.contains(s, case=False, na=False, regex=False)
                | agg["suggested_mapping"].astype(str).str.contains(s, case=False, na=False, regex=False)
            ]
        agg = agg.sort_values("cluster_size", ascending=False)
        page = agg.iloc[offset: offset + limit]
        return {"total": int(len(agg)), "offset": offset, "limit": limit,
                "columns": [str(c) for c in page.columns], "rows": frame_records(page)}

    def outputs(self) -> List[Dict[str, Any]]:
        files = []
        for p in sorted(self.output_dir.iterdir()) if self.output_dir.exists() else []:
            if p.is_file():
                files.append({"name": p.name, "path": str(p), "size_bytes": p.stat().st_size,
                              "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
        return files

    def promote_master(self, destination: str) -> Dict[str, Any]:
        """Copy this run's updated master over ``destination`` (keeping a timestamped backup).

        Approved-but-not-yet-applied review decisions are applied first, so the
        promoted file always contains everything the reviewer accepted.
        """
        if self.decision_counts()["unapplied_approved"]:
            self.apply_decisions()
        src = self._latest("updated_master_*.csv", exclude="_with_source")
        if not src:
            raise FileNotFoundError("This run produced no updated_master_*.csv")
        dest = Path(destination).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if dest.exists():
            backup = dest.with_name(f"{dest.stem}.backup_{datetime.now():%Y%m%d_%H%M%S}{dest.suffix}")
            shutil.copy2(dest, backup)
        shutil.copy2(src, dest)
        self._log("INFO", f"Promoted {Path(src).name} -> {dest}" + (f" (backup: {backup.name})" if backup else ""))
        self._persist()
        return {"source": src, "destination": str(dest), "backup": str(backup) if backup else None}

    def _raw_ucc_path(self) -> Optional[Path]:
        """The UCC file this run was fed (the sampled copy when a sample was run)."""
        for cand in (self.config.ucc_file, str(self.run_dir / "inputs" / "ucc.csv"),
                     getattr(self.source.spec, "ucc_path", "")):
            if cand and Path(cand).expanduser().is_file():
                return Path(cand).expanduser()
        return None

    def finalize(self, *, period: str, folder: str) -> Dict[str, Any]:
        """Close the period: write the reviewed mapping as ``Master Mapping_<period>.csv``
        and the UCC file with ``SECP_NAME_MAPPED`` / ``Firm_ID`` appended as
        ``ucc_<period>_enriched.csv``, both into ``folder``.

        Unapplied approvals are applied first so the files carry every review
        decision. The enrichment joins on the same key the matcher uses
        (state pair + upper-cased, stripped filer name), one mapping per key,
        and keeps every raw row and column; unmapped filings stay in the file
        with the new columns blank.
        """
        period = re.sub(r"[^A-Za-z0-9_-]+", "", period.strip())
        if not period:
            raise ValueError("A period label is required, e.g. Aug26")
        if self.decision_counts()["unapplied_approved"]:
            self.apply_decisions()
        src = self._latest("updated_master_*.csv", exclude="_with_source")
        if not src:
            raise FileNotFoundError("This run produced no updated_master_*.csv")
        raw_path = self._raw_ucc_path()
        if raw_path is None:
            raise FileNotFoundError("The run's UCC file is no longer where it was; re-run with the current file")
        from .datasources import detect_delimiter
        from .core.pipeline_logic import _standardize_state_columns

        dest = Path(folder).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        master_out = dest / f"Master Mapping_{period}.csv"
        ucc_out = dest / f"ucc_{period}_enriched.csv"
        if master_out.exists():
            shutil.copy2(master_out, master_out.with_name(f"{master_out.stem}.backup_{datetime.now():%Y%m%d_%H%M%S}.csv"))
        shutil.copy2(src, master_out)

        master = pd.read_csv(master_out, encoding="latin-1", low_memory=False)
        master = _standardize_state_columns(master)
        with_src = self._master_with_source()
        if with_src is not None and "_source" in with_src.columns and len(with_src) == len(master):
            master["mapping_source"] = with_src["_source"].astype(str).values
        key = lambda df: (df["DEBT_STATE"].astype("string").fillna("") + "|" + df["SECP_STATE"].astype("string").fillna("")
                          + "|" + df["SECP_NAME"].astype("string").fillna("").str.strip().str.upper())
        master["_key"] = key(master)
        rhs_cols = ["_key", "SECP_NAME_MAPPED", "Firm ID"] + (["mapping_source"] if "mapping_source" in master.columns else [])
        rhs = master[rhs_cols].drop_duplicates("_key", keep="first").rename(columns={"Firm ID": "Firm_ID"})

        raw = pd.read_csv(raw_path, sep=detect_delimiter(str(raw_path)), encoding="latin-1", low_memory=False)
        raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed:")]  # a saved pandas index
        raw = raw.drop(columns=[c for c in ("SECP_NAME_MAPPED", "Firm_ID", "Firm ID", "mapping_source") if c in raw.columns])
        missing = [c for c in ("DEBT_STATE", "SECP_STATE", "SECP_NAME") if c not in raw.columns]
        if missing:
            raise ValueError(f"UCC file is missing join column(s) {missing}")
        raw_std = _standardize_state_columns(raw[["DEBT_STATE", "SECP_STATE", "SECP_NAME"]])
        raw["_key"] = key(raw_std)
        enriched = raw.merge(rhs, on="_key", how="left").drop(columns="_key")
        fid = pd.to_numeric(enriched["Firm_ID"], errors="coerce")
        if (fid.dropna() % 1 == 0).all():
            enriched["Firm_ID"] = fid.astype("Int64")
        enriched.to_csv(ucc_out, index=False)

        n = int(len(enriched))
        mapped = int(enriched["SECP_NAME_MAPPED"].notna().sum())
        with_fid = int(enriched["Firm_ID"].notna().sum())
        by_src = ({str(k): int(v) for k, v in enriched["mapping_source"].value_counts().items()}
                  if "mapping_source" in enriched.columns else {})
        result = {"period": period, "master_file": str(master_out), "ucc_file": str(ucc_out), "master_rows": int(len(master)),
                  "ucc_rows": n, "mapped_rows": mapped, "rows_with_firm_id": with_fid, "unmapped_rows": n - mapped,
                  "by_source": by_src, "finalized_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self._log("INFO", f"Finalised {period}: {master_out.name} ({len(master):,} rows); {ucc_out.name}: "
                          f"{mapped:,}/{n:,} filings mapped, {with_fid:,} with Firm ID")
        if self.summary is not None:
            self.summary["finalized"] = result
        self._persist()
        return result

    # --------------------------------------------------------------- restore
    @classmethod
    def load(cls, run_dir: Path, decisions: Optional[DecisionStore] = None) -> Optional["PipelineRun"]:
        meta = run_dir / "run.json"
        if not meta.is_file():
            return None
        data = json.loads(meta.read_text())
        spec = make_datasource(data["source"]).spec
        cfg = PipelineConfig.from_overrides(data.get("config") or {})
        run = cls(spec, cfg, runs_root=run_dir.parent, sample_rows=data.get("sample_rows", 0),
                  run_id=data["id"], name=data.get("name", ""), decisions=decisions)
        run.created_at = data.get("created_at", run.created_at)
        run.started_at, run.finished_at = data.get("started_at"), data.get("finished_at")
        run.status = data.get("status", "unknown")
        if run.status in ("queued", "running"):  # engine died mid-run
            run.status, run.error = "failed", data.get("error") or "Engine stopped before the run finished"
        run.error = run.error or data.get("error")
        run.input_rows = data.get("input_rows", {})
        run.summary = data.get("summary")
        run.stages = [StageState(**s) for s in data.get("stages", [])] or run.stages
        for line in data.get("log_tail", []):
            run._logs.append(LogLine(**line))
            run._seq = max(run._seq, line["seq"])
        legacy = run._legacy_decisions_path
        if legacy.is_file():  # pre-SQLite per-run decisions: fold into the shared store once
            try:
                n = run.decisions.import_legacy(json.loads(legacy.read_text()), run.id)
                legacy.rename(legacy.with_suffix(".json.imported"))
                run._log("INFO", f"Imported {n} decision(s) from decisions.json into the decisions database")
            except Exception as e:  # noqa: BLE001 - a corrupt file must not hide the run
                run._log("WARNING", f"Could not import legacy decisions.json: {e}")
        return run


class RunStore:
    """All runs known to this engine, newest first (sharing one decision store)."""

    def __init__(self, root: Optional[Path] = None, decisions_db: Optional[Path] = None):
        self.root = (root or app_data_dir() / "runs").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.decisions = DecisionStore.at(decisions_db or self.root.parent / "decisions.sqlite")
        self._runs: Dict[str, PipelineRun] = {}
        self._lock = threading.Lock()
        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                run = PipelineRun.load(d, decisions=self.decisions)
                if run is not None:
                    self._runs[run.id] = run

    def create(self, spec: SourceSpec, config: PipelineConfig, *, sample_rows: int = 0, name: str = "") -> PipelineRun:
        run = PipelineRun(spec, config, runs_root=self.root, sample_rows=sample_rows, name=name, decisions=self.decisions)
        with self._lock:
            self._runs[run.id] = run
        return run

    def get(self, run_id: str) -> Optional[PipelineRun]:
        return self._runs.get(run_id)

    def list(self) -> List[Dict[str, Any]]:
        snaps = [r.snapshot() for r in self._runs.values()]
        for s in snaps:
            s.pop("config", None)
        return sorted(snaps, key=lambda s: s["created_at"], reverse=True)

    def active(self) -> Optional[PipelineRun]:
        return next((r for r in self._runs.values() if r.status in ("queued", "running")), None)

    def delete(self, run_id: str) -> bool:
        run = self._runs.get(run_id)
        if run is None or run.status in ("queued", "running"):
            return False
        shutil.rmtree(run.run_dir, ignore_errors=True)
        with self._lock:
            del self._runs[run_id]
        return True
