"""Data sources: where the three inputs come from.

The generated core reads its inputs from CSV paths (``config.ucc_file`` etc.) and
re-reads the UCC file in the reporting stages. Rather than fork that code, every
``DataSource`` *materialises* its inputs as CSV files inside the run directory
(or hands back the original paths when they are already CSV). Swapping CSV for
SQL is therefore a configuration change, not a code change.

Required columns
----------------
UCC filings   : DEBT_STATE, SECP_STATE, SECP_NAME   (+ optional filing-type /
                debtor-name columns used by the lead-bank stage)
Master mapping: DEBT_STATE, SECP_STATE, SECP_NAME, SECP_NAME_MAPPED, "Firm ID"
Tracked banks : "Bank Name", "Firm ID" (or "Firm_ID")
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field

UCC_REQUIRED = ["DEBT_STATE", "SECP_STATE", "SECP_NAME"]
MASTER_REQUIRED = UCC_REQUIRED + ["SECP_NAME_MAPPED", "Firm ID"]
UNIVERSE_REQUIRED = ["Bank Name", "Firm ID"]
# Columns that satisfy a required one under another spelling (tracked_banks.csv says Firm_ID).
COLUMN_ALIASES = {"Firm ID": ("Firm_ID",)}


def _has(cols, c: str) -> bool:
    return c in cols or any(a in cols for a in COLUMN_ALIASES.get(c, ()))

_ENCODING = "latin-1"  # what the core reads with


# ----------------------------------------------------------------------------- specs
class CsvSourceSpec(BaseModel):
    type: Literal["csv"] = "csv"
    ucc_path: str = Field(..., description="UCC filings CSV")
    master_path: str = Field(..., description="Master Mapping CSV")
    universe_path: str = Field("", description="Tracked banks CSV: Bank Name + Firm ID (optional)")


class SqlSourceSpec(BaseModel):
    type: Literal["sql"] = "sql"
    url: str = Field(..., description="SQLAlchemy URL, e.g. mssql+pyodbc://... or postgresql://...")
    ucc_query: str
    master_query: str
    universe_query: str = ""


SourceSpec = Union[CsvSourceSpec, SqlSourceSpec]


# ------------------------------------------------------------------------- results
@dataclass
class MaterializedInputs:
    ucc_file: str
    master_file: str
    universe_file: str  # "" when not provided
    rows: dict[str, int]


class DataSourceError(RuntimeError):
    pass


def _require(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if not _has(df.columns, c)]
    if missing:
        raise DataSourceError(f"{what}: missing required column(s) {missing}; have {list(df.columns)}")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=_ENCODING, errors="replace")


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding=_ENCODING, errors="replace") as fh:
        head = fh.readline()
    return "\t" if head.count("\t") > head.count(",") else ","


def _count_rows(path: str) -> int:
    with open(path, "rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


# --------------------------------------------------------------------------- CSV
class CsvDataSource:
    kind = "csv"

    def __init__(self, spec: CsvSourceSpec):
        self.spec = spec

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "csv", "inputs": {}}
        for name, path, required in (
            ("ucc", self.spec.ucc_path, UCC_REQUIRED),
            ("master", self.spec.master_path, MASTER_REQUIRED),
            ("universe", self.spec.universe_path, UNIVERSE_REQUIRED),
        ):
            if not path:
                out["inputs"][name] = {"present": False}
                continue
            p = Path(path).expanduser()
            if not p.is_file():
                out["inputs"][name] = {"present": False, "error": f"file not found: {p}"}
                continue
            try:
                head = pd.read_csv(p, sep=detect_delimiter(str(p)), encoding=_ENCODING, nrows=8)
                cols = [str(c) for c in head.columns]
                out["inputs"][name] = {
                    "present": True,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "rows": _count_rows(str(p)),
                    "columns": cols,
                    "missing_columns": [c for c in required if not _has(cols, c)],
                    "preview": head.fillna("").astype(str).to_dict(orient="records"),
                }
            except Exception as e:  # noqa: BLE001
                out["inputs"][name] = {"present": True, "path": str(p), "error": repr(e)}
        return out

    def materialize(self, run_dir: Path, sample_rows: int = 0) -> MaterializedInputs:
        ucc = str(Path(self.spec.ucc_path).expanduser())
        master = str(Path(self.spec.master_path).expanduser())
        universe = str(Path(self.spec.universe_path).expanduser()) if self.spec.universe_path else ""
        for label, p in (("UCC", ucc), ("Master Mapping", master)):
            if not Path(p).is_file():
                raise DataSourceError(f"{label} file not found: {p}")
        if universe and not Path(universe).is_file():
            raise DataSourceError(f"Universe file not found: {universe}")

        rows = {"ucc": _count_rows(ucc), "master": _count_rows(master),
                "universe": _count_rows(universe) if universe else 0}

        if sample_rows and sample_rows > 0 and rows["ucc"] > sample_rows:
            df = pd.read_csv(ucc, sep=detect_delimiter(ucc), encoding=_ENCODING, low_memory=False)
            df = df.sample(n=sample_rows, random_state=42).reset_index(drop=True)
            target = run_dir / "inputs" / "ucc_sample.csv"
            _write_csv(df, target)
            ucc, rows["ucc"] = str(target), len(df)

        return MaterializedInputs(ucc, master, universe, rows)


# --------------------------------------------------------------------------- SQL
class SqlDataSource:
    """Runs three queries through SQLAlchemy and stages the results as CSV."""

    kind = "sql"

    def __init__(self, spec: SqlSourceSpec):
        self.spec = spec

    def _engine(self):
        try:
            from sqlalchemy import create_engine
        except ImportError as e:  # pragma: no cover
            raise DataSourceError("SQL source requires `pip install 'ucc-engine[sql]'` plus your DB driver") from e
        return create_engine(self.spec.url)

    def redacted_url(self) -> str:
        try:
            from sqlalchemy.engine import make_url

            return make_url(self.spec.url).render_as_string(hide_password=True)
        except Exception:  # noqa: BLE001
            return "<sql url>"

    def _frame(self, query: str, chunksize: Optional[int] = None) -> pd.DataFrame:
        from sqlalchemy import text

        with self._engine().connect() as con:
            if chunksize:
                return next(iter(pd.read_sql(text(query), con, chunksize=chunksize)))
            return pd.read_sql(text(query), con)

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "sql", "url": self.redacted_url(), "inputs": {}}
        for name, q, required in (
            ("ucc", self.spec.ucc_query, UCC_REQUIRED),
            ("master", self.spec.master_query, MASTER_REQUIRED),
            ("universe", self.spec.universe_query, UNIVERSE_REQUIRED),
        ):
            if not q.strip():
                out["inputs"][name] = {"present": False}
                continue
            try:
                head = self._frame(q, chunksize=8)
                cols = [str(c) for c in head.columns]
                out["inputs"][name] = {
                    "present": True, "columns": cols,
                    "missing_columns": [c for c in required if not _has(cols, c)],
                    "preview": head.fillna("").astype(str).to_dict(orient="records"),
                }
            except Exception as e:  # noqa: BLE001
                out["inputs"][name] = {"present": True, "error": repr(e)}
        return out

    def materialize(self, run_dir: Path, sample_rows: int = 0) -> MaterializedInputs:
        inputs = run_dir / "inputs"
        rows: dict[str, int] = {}

        ucc = self._frame(self.spec.ucc_query)
        _require(ucc, UCC_REQUIRED, "UCC query")
        if sample_rows and 0 < sample_rows < len(ucc):
            ucc = ucc.sample(n=sample_rows, random_state=42).reset_index(drop=True)
        ucc_path = inputs / "ucc.csv"
        _write_csv(ucc, ucc_path)
        rows["ucc"] = len(ucc)

        master = self._frame(self.spec.master_query)
        _require(master, MASTER_REQUIRED, "Master query")
        master_path = inputs / "Master Mapping.csv"
        _write_csv(master, master_path)
        rows["master"] = len(master)

        universe_path = ""
        if self.spec.universe_query.strip():
            uni = self._frame(self.spec.universe_query)
            _require(uni, UNIVERSE_REQUIRED, "Universe query")
            p = inputs / "Universe.csv"
            _write_csv(uni, p)
            universe_path, rows["universe"] = str(p), len(uni)
        else:
            rows["universe"] = 0

        return MaterializedInputs(str(ucc_path), str(master_path), universe_path, rows)


DataSource = Union[CsvDataSource, SqlDataSource]


def make_datasource(spec: SourceSpec | dict[str, Any]) -> DataSource:
    if isinstance(spec, dict):
        spec = CsvSourceSpec(**spec) if spec.get("type", "csv") == "csv" else SqlSourceSpec(**spec)
    return CsvDataSource(spec) if isinstance(spec, CsvSourceSpec) else SqlDataSource(spec)


def public_spec(spec: SourceSpec) -> dict[str, Any]:
    """Spec safe to persist / show in the UI (SQL password hidden)."""
    data = spec.model_dump()
    if isinstance(spec, SqlSourceSpec):
        data["url"] = SqlDataSource(spec).redacted_url()
    return data
