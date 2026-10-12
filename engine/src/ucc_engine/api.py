"""Local HTTP API consumed by the Tauri desktop app (binds to 127.0.0.1 only)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PipelineConfig, app_data_dir
from .datasources import CsvSourceSpec, DataSourceError, SqlSourceSpec, SourceSpec, make_datasource
from .embeddings import available_backends, prefetch_models
from .runner import BUCKETS, REVIEW_BANDS, RunStore


class RunRequest(BaseModel):
    source: SourceSpec = Field(..., discriminator="type")
    config: Dict[str, Any] = Field(default_factory=dict, description="PipelineConfig overrides")
    sample_rows: int = Field(0, ge=0, description="0 = all UCC rows; N = random sample for a quick run")
    name: str = ""


class PromoteRequest(BaseModel):
    destination: str = Field(..., description="Path the updated master should overwrite (a backup is kept)")


class PreviewRequest(BaseModel):
    source: SourceSpec = Field(..., discriminator="type")


class DecisionItem(BaseModel):
    key: str = Field(..., description="`key` of a review-queue row")
    decision: Optional[Literal["approved", "rejected"]] = Field(None, description="null clears the decision")
    mapped_name: Optional[str] = Field(None, description="override the suggested SECP_NAME_MAPPED (required when mapping an unmatched row by hand)")
    firm_id: Optional[Any] = Field(None, description="override the Firm ID (default: Universe lookup by mapped name)")
    all_states: bool = Field(False, description="manual mapping only: also apply to this filer name wherever it is unmatched, now and in later runs")


class DecisionsRequest(BaseModel):
    items: List[DecisionItem] = Field(..., min_length=1)


class ClearDecisionsRequest(BaseModel):
    keys: List[str] = Field(..., min_length=1)


class ExportRequest(BaseModel):
    destination: str = Field("", description="CSV path to write; empty = the engine's exports folder")


def create_app(store: Optional[RunStore] = None) -> FastAPI:
    store = store or RunStore()
    app = FastAPI(title="UCC Mapping Engine", version=__version__)
    app.add_middleware(  # Tauri webview origins (tauri://localhost, http://tauri.localhost, Vite dev)
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # ---------------------------------------------------------------- meta
    @app.get("/health")
    def health() -> Dict[str, Any]:
        active = store.active()
        return {
            "status": "ok",
            "version": __version__,
            "pid": os.getpid(),
            "data_dir": str(app_data_dir()),
            "decisions_db": str(store.decisions.path),
            "decisions": store.decisions.counts(),
            "backends": available_backends(),
            "active_run": active.id if active else None,
        }

    @app.get("/config/defaults")
    def config_defaults() -> Dict[str, Any]:
        return {"defaults": PipelineConfig().to_dict(), "schema": PipelineConfig.schema()}

    @app.post("/models/prefetch")
    def models_prefetch(overrides: Dict[str, Any] | None = None) -> Dict[str, str]:
        try:
            return prefetch_models(PipelineConfig.from_overrides(overrides or {}))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"Model prefetch failed: {e}") from e

    # ------------------------------------------------------------- sources
    @app.get("/models/status")
    def models_status(model_share_dir: str = "", model_cache_dir: str = "") -> Dict[str, Any]:
        """Is the network model share reachable, what is on it, what is local, when last copied."""
        from .model_sync import sync_status

        cfg = PipelineConfig.from_overrides({"model_share_dir": model_share_dir, "model_cache_dir": model_cache_dir})
        return sync_status(cfg.model_share_dir, cfg.model_cache_dir)

    @app.post("/models/sync")
    def models_sync(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Copy the models from `model_share_dir` into the local model folder now."""
        from .model_sync import sync_models_from_share

        cfg = PipelineConfig.from_overrides(overrides or {})
        if not cfg.model_share_dir:
            raise HTTPException(400, "model_share_dir is not set")
        try:
            return sync_models_from_share(cfg.model_share_dir, cfg.model_cache_dir)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except OSError as e:
            raise HTTPException(500, f"Copying models failed: {e}") from e

    @app.post("/sources/preview")
    def sources_preview(req: PreviewRequest) -> Dict[str, Any]:
        try:
            return make_datasource(req.source).describe()
        except DataSourceError as e:
            raise HTTPException(400, str(e)) from e

    # ---------------------------------------------------------------- runs
    @app.get("/runs")
    def runs_list() -> List[Dict[str, Any]]:
        return store.list()

    @app.post("/runs", status_code=201)
    def runs_create(req: RunRequest) -> Dict[str, Any]:
        if store.active() is not None:
            raise HTTPException(409, "A run is already in progress")
        cfg = PipelineConfig.from_overrides(req.config)
        run = store.create(req.source, cfg, sample_rows=req.sample_rows, name=req.name).start()
        return run.snapshot()

    def _get(run_id: str):
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, f"Unknown run {run_id}")
        return run

    @app.get("/runs/{run_id}")
    def runs_get(run_id: str) -> Dict[str, Any]:
        return _get(run_id).snapshot()

    @app.delete("/runs/{run_id}")
    def runs_delete(run_id: str) -> Dict[str, bool]:
        _get(run_id)
        if not store.delete(run_id):
            raise HTTPException(409, "Cannot delete a run that is in progress")
        return {"deleted": True}

    @app.post("/runs/{run_id}/cancel")
    def runs_cancel(run_id: str) -> Dict[str, Any]:
        run = _get(run_id)
        run.cancel()
        return run.snapshot()

    @app.get("/runs/{run_id}/logs")
    def runs_logs(run_id: str, after: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=5000)) -> Dict[str, Any]:
        run = _get(run_id)
        lines = run.logs(after=after, limit=limit)
        return {"lines": lines, "last_seq": lines[-1]["seq"] if lines else after, "status": run.status}

    @app.get("/runs/{run_id}/summary")
    def runs_summary(run_id: str) -> Dict[str, Any]:
        run = _get(run_id)
        if run.summary is None and run.status == "succeeded":
            run.summary = run._build_summary()
        return run.summary or {}

    @app.get("/runs/{run_id}/review")
    def runs_review(
        run_id: str,
        confidence: Optional[List[str]] = Query(None, description=f"any of {list(REVIEW_BANDS)}"),
        search: str = "",
        method: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        sort: str = "match_score",
        descending: bool = True,
        decision: Literal["", "pending", "approved", "rejected"] = "",
        mapped: Optional[str] = Query(None, description="exact suggested/decided bank name (from /review/groups)"),
    ) -> Dict[str, Any]:
        return _get(run_id).review(confidence=confidence, search=search, method=method, offset=offset,
                                   limit=limit, sort=sort, descending=descending, decision=decision, mapped=mapped)

    @app.get("/runs/{run_id}/banks")
    def runs_banks(run_id: str) -> List[Dict[str, Any]]:
        """Banks a filer can be mapped to by hand: the Universe plus the master's mapped names."""
        return _get(run_id).banks()

    @app.get("/runs/{run_id}/master")
    def runs_master(
        run_id: str, source: str = Query("added", description="original | added | an exact _source value"),
        search: str = "", offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        """The updated master with provenance: what came in, what this run / reviewers added."""
        return _get(run_id).master_rows(source=source, search=search, offset=offset, limit=limit)

    @app.get("/runs/{run_id}/review/groups")
    def runs_review_groups(
        run_id: str, decision: Literal["", "pending", "approved", "rejected"] = "pending", search: str = "",
    ) -> List[Dict[str, Any]]:
        """The review queue rolled up by suggested bank (biggest first)."""
        return _get(run_id).review_groups(decision=decision, search=search)

    @app.post("/runs/{run_id}/propagate-decisions")
    def runs_propagate(run_id: str) -> Dict[str, Any]:
        """Inherit decisions already made on the same filer name (other state pairs / spellings / runs)."""
        run = _get(run_id)
        try:
            return {**run.propagate_decisions(), "counts": run.decision_counts()}
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/runs/{run_id}/rows")
    def runs_rows(
        run_id: str,
        bucket: Literal["direct", "fuzzy_high", "review", "unmatched"] = Query(..., description=f"one of {list(BUCKETS)}"),
        search: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        sort: str = "match_score",
        descending: bool = True,
    ) -> Dict[str, Any]:
        """Read-only page of one result bucket (direct / fuzzy_high / review / unmatched)."""
        return _get(run_id).rows(bucket=bucket, search=search, offset=offset, limit=limit, sort=sort, descending=descending)

    # ----------------------------------------------------- review decisions
    @app.get("/runs/{run_id}/decisions")
    def runs_decisions(run_id: str) -> Dict[str, Any]:
        return {"counts": _get(run_id).decision_counts()}

    @app.put("/runs/{run_id}/decisions")
    def runs_set_decisions(run_id: str, req: DecisionsRequest) -> Dict[str, Any]:
        try:
            return _get(run_id).set_decisions([i.model_dump() for i in req.items])
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except (ValueError, RuntimeError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/runs/{run_id}/apply-decisions")
    def runs_apply_decisions(run_id: str) -> Dict[str, Any]:
        try:
            return _get(run_id).apply_decisions()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from e

    # ------------------------------------------- the decisions database (all runs)
    @app.get("/decisions")
    def decisions_list(
        decision: Literal["", "approved", "rejected"] = "",
        search: str = "",
        run_id: str = "",
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        sort: str = "decided_at",
        descending: bool = True,
    ) -> Dict[str, Any]:
        return store.decisions.list(decision=decision, search=search, run_id=run_id, offset=offset,
                                    limit=limit, sort=sort, descending=descending)

    @app.post("/decisions/clear")
    def decisions_clear(req: ClearDecisionsRequest) -> Dict[str, Any]:
        cleared = sum(store.decisions.clear(k) for k in req.keys)
        return {"cleared": cleared, "counts": store.decisions.counts()}

    @app.get("/decisions/history")
    def decisions_history(key: str) -> List[Dict[str, Any]]:
        return store.decisions.history(key)

    def _export_path(destination: str = "") -> Path:
        path = Path(destination).expanduser() if destination else (
            store.decisions.path.parent / "exports" / f"review_decisions_{datetime.now():%Y%m%d_%H%M%S}.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @app.get("/decisions/export")
    def decisions_export() -> FileResponse:
        path = _export_path()
        store.decisions.export_csv(path)
        return FileResponse(path, filename=path.name, media_type="text/csv")

    @app.post("/decisions/export")
    def decisions_export_to(req: ExportRequest) -> Dict[str, Any]:
        path = _export_path(req.destination)
        return {"path": str(path), "rows": store.decisions.export_csv(path)}

    @app.get("/runs/{run_id}/clusters")
    def runs_clusters(
        run_id: str, search: str = "", min_size: int = Query(1, ge=1),
        offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return _get(run_id).clusters(search=search, min_size=min_size, offset=offset, limit=limit)

    @app.get("/runs/{run_id}/outputs")
    def runs_outputs(run_id: str) -> List[Dict[str, Any]]:
        return _get(run_id).outputs()

    @app.get("/runs/{run_id}/outputs/{name}")
    def runs_output_file(run_id: str, name: str) -> FileResponse:
        run = _get(run_id)
        path = (run.output_dir / Path(name).name).resolve()
        if run.output_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(404, "No such output file")
        return FileResponse(path, filename=path.name)

    @app.post("/runs/{run_id}/promote-master")
    def runs_promote(run_id: str, req: PromoteRequest) -> Dict[str, Any]:
        try:
            return _get(run_id).promote_master(req.destination)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    return app


# `uvicorn ucc_engine.api:app`
app = create_app()
