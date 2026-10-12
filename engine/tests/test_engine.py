"""Fast, model-free tests for the engine plumbing (the pipeline itself is exercised by `ucc-engine run`)."""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ENGINE = Path(__file__).resolve().parents[1]


def test_core_imports_without_dagster_or_torch():
    import ucc_engine.core.pipeline_logic as L

    src = Path(L.__file__).read_text()
    assert "from dagster" not in src and "sentence_transformers" not in src
    assert callable(L.fuzzy_match_results) and callable(L.updated_master_data)


def test_config_mirrors_dagster_fields_and_coerces_overrides():
    from ucc_engine.config import PipelineConfig

    cfg = PipelineConfig.from_overrides({"fuzzy_auto_threshold": "95", "embed_auto_threshold": "0.9",
                                         "advanced_cross_encoder_enabled": "false", "unknown_key": 1, "cpu_threads": ""})
    assert cfg.fuzzy_auto_threshold == 95 and cfg.embed_auto_threshold == 0.9
    assert cfg.advanced_cross_encoder_enabled is False
    assert cfg.cpu_threads == -1  # blank override ignored -> default
    assert not hasattr(cfg, "unknown_key")
    names = {f["name"] for f in PipelineConfig.schema()}
    # Every attribute the core *reads* (code, not docstrings) must exist on the engine config.
    import ast

    import ucc_engine.core.pipeline_logic as L

    tree = ast.parse(Path(L.__file__).read_text())
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config"
    }
    assert used <= names, f"core reads config fields missing from PipelineConfig: {sorted(used - names)}"


def test_shared_model_survives_dataframe_copy():
    """pandas deep-copies df.attrs; the core stores the model there."""
    from ucc_engine.embeddings import _SharedModel

    class Fake(_SharedModel):
        pass

    m = Fake()
    df = pd.DataFrame({"a": [1]})
    df.attrs["embedding_model"] = m
    assert df.copy().attrs["embedding_model"] is m
    assert copy.deepcopy({"model": m})["model"] is m


def test_compat_context_forwards_logs():
    from ucc_engine.compat import AssetExecutionContext, MetadataValue, Output

    seen = []
    ctx = AssetExecutionContext("stage", sink=lambda lvl, msg: seen.append((lvl, msg)))
    ctx.log.info("hello %s", "world")
    ctx.log.warning("careful")
    assert seen == [("INFO", "hello world"), ("WARNING", "careful")]
    assert Output(value=42, metadata={"x": MetadataValue.int(1)}) == 42


def test_csv_datasource_validates_columns(tmp_path: Path):
    from ucc_engine.datasources import CsvSourceSpec, DataSourceError, make_datasource

    good = tmp_path / "ucc.csv"
    good.write_text("DEBT_STATE,SECP_STATE,SECP_NAME\nCA,NY,ACME BANK\n")
    master = tmp_path / "master.csv"
    master.write_text('DEBT_STATE,SECP_STATE,SECP_NAME,SECP_NAME_MAPPED,Firm ID\nCA,NY,ACME BANK,Acme Bank,1\n')
    bad = tmp_path / "bad.csv"
    bad.write_text("Bank Name,Firm ID\nAcme,1\n")

    ds = make_datasource(CsvSourceSpec(ucc_path=str(good), master_path=str(bad)))
    desc = ds.describe()
    assert desc["inputs"]["ucc"]["missing_columns"] == []
    assert set(desc["inputs"]["master"]["missing_columns"]) == {"DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED"}
    assert desc["inputs"]["universe"] == {"present": False}

    ds = make_datasource({"type": "csv", "ucc_path": str(good), "master_path": str(master)})
    mat = ds.materialize(tmp_path / "run", sample_rows=0)
    assert mat.ucc_file == str(good) and mat.universe_file == "" and mat.rows == {"ucc": 1, "master": 1, "universe": 0}

    with pytest.raises(DataSourceError):
        make_datasource(CsvSourceSpec(ucc_path=str(tmp_path / "nope.csv"), master_path=str(master))).materialize(tmp_path)


def test_sql_spec_redacts_password():
    from ucc_engine.datasources import SqlSourceSpec, public_spec

    spec = SqlSourceSpec(url="postgresql://user:s3cret@db.local/ucc", ucc_query="select 1", master_query="select 1")
    assert "s3cret" not in public_spec(spec)["url"]


def test_registry_is_topologically_ordered_and_matches_core_signatures():
    import inspect

    import ucc_engine.core.pipeline_logic as L
    from ucc_engine.registry import PIPELINE

    seen: set[str] = set()
    for st in PIPELINE:
        assert set(st.deps) <= seen, f"{st.name} depends on {set(st.deps) - seen} before they run"
        params = [p for p in inspect.signature(getattr(L, st.name)).parameters if p != "context"]
        expected = (["config"] if st.needs_config else []) + st.deps
        assert params == expected, f"{st.name}: core wants {params}, registry passes {expected}"
        seen.add(st.name)


def test_clustering_stages_are_skipped_by_default_but_keep_the_dag_wired():
    import numpy as np

    from ucc_engine.config import PipelineConfig
    from ucc_engine.registry import CLUSTER_COLUMNS, STAGES_BY_NAME, is_enabled, run_stage, total_weight

    cfg = PipelineConfig()
    assert cfg.clustering_enabled is False
    clustering = [STAGES_BY_NAME[n] for n in ("clustered_results", "merged_cluster_results", "suggested_mapping_results")]
    assert all(not is_enabled(s, cfg) for s in clustering)
    assert all(is_enabled(s, PipelineConfig(clustering_enabled=True)) for s in clustering)
    assert total_weight(cfg) < total_weight(PipelineConfig(clustering_enabled=True))

    # Frame as it leaves Stage 4: one matched row, one pending, one NO_MASTER_STATE_PAIR.
    df = pd.DataFrame({
        "SECP_NAME": ["A", "B", "C"], "SECP_NAME_NORMALIZED": ["a", "b", "c"],
        "SECP_NAME_MAPPED": ["Acme", None, None], "confidence": ["AUTO", None, "NO_MASTER_STATE_PAIR"],
    })
    df.attrs["embedding_model"] = object()
    cache = {"advanced_embedding_results": df, "master_embeddings_cache": {}, "normalized_master_data": {}}
    for s in clustering:
        out = run_stage(s, cfg, None, cache)
    assert out is cache["suggested_mapping_results"]
    assert out["confidence"].tolist() == ["AUTO", "NEW_ENTITY", "NO_MASTER_STATE_PAIR"]
    assert set(CLUSTER_COLUMNS) <= set(out.columns) and out["cluster_id"].isna().all()
    assert np.isnan(out["cluster_size"].mean())  # what pipeline_outputs computes for the summary file
    assert out.attrs == {} and df.attrs  # input untouched, output carries no model handle


def _fake_finished_run(tmp_path: Path, name: str = "t", suggestions: dict | None = None):
    """A succeeded run with an in-memory results frame + updated master, no models involved."""
    from ucc_engine.config import PipelineConfig
    from ucc_engine.datasources import CsvSourceSpec
    from ucc_engine.runner import PipelineRun

    universe = tmp_path / "universe.csv"
    universe.write_text("Bank Name,Firm ID\nAcme Bank,101\nZeta Bank,202\n")
    spec = CsvSourceSpec(ucc_path=str(tmp_path / "u.csv"), master_path=str(tmp_path / "m.csv"), universe_path=str(universe))
    run = PipelineRun(spec, PipelineConfig(), runs_root=tmp_path / "runs", name=name)
    run.config.universe_file = str(universe)
    run.status = "succeeded"
    mapped = {"ACME BK NA": "Acme Bank", "ZETA BANK TRUSTEE": "Zeta Bank", "OTHER CO": "Unknown Co", "KNOWN": "Known"}
    mapped.update(suggestions or {})
    results = pd.DataFrame({
        "DEBT_STATE": ["ca", "NY", "TX", "FL"], "SECP_STATE": ["CA", "NY", "TX", "FL"],
        "SECP_NAME": list(mapped), "SECP_NAME_MAPPED": list(mapped.values()),
        "match_method": ["Advanced Match"] * 3 + ["Exact Match (Raw)"],
        "match_score": [0.91, 0.88, 0.86, 1.0],
        "confidence": ["REVIEW_HIGH", "REVIEW_MEDIUM", "REVIEW_LOW", "AUTO"],
        "Firm_ID": [101, None, None, 5],
    })
    run._cache["universe_enriched_results"] = results
    master = pd.DataFrame({"DEBT_STATE": ["FL"], "SECP_STATE": ["FL"], "SECP_NAME": ["KNOWN"],
                           "SECP_NAME_MAPPED": ["Known"], "Firm ID": [5], "_source": ["original_master"]})
    run._cache["updated_master_data"] = master
    run.output_dir.mkdir(parents=True, exist_ok=True)
    master.drop(columns="_source").to_csv(run.output_dir / "updated_master_20260101_000000.csv", index=False)
    master.to_csv(run.output_dir / "updated_master_20260101_000000_with_source.csv", index=False)
    run.summary = run._build_summary()
    return run


def test_review_decisions_flow_into_updated_master(tmp_path: Path):
    from ucc_engine.runner import PipelineRun

    run = _fake_finished_run(tmp_path)
    page = run.review()
    assert page["total"] == 3 and all("key" in r and r["decision"] is None for r in page["rows"])
    k_acme, k_other, k_zeta = (r["key"] for r in sorted(page["rows"], key=lambda r: r["SECP_NAME"]))
    assert k_acme == "CA|CA|ACME BK NA"  # states normalised like the core

    # approve as suggested, approve with a corrected name (Firm ID re-resolved from the Universe), reject one
    out = run.set_decisions([
        {"key": k_acme, "decision": "approved"},
        {"key": k_other, "decision": "approved", "mapped_name": "Zeta Bank"},
        {"key": k_zeta, "decision": "rejected"},
    ])
    counts = {k: v for k, v in out["counts"].items() if k != "store"}
    assert counts == {"approved": 2, "rejected": 1, "pending": 0, "total": 3, "unapplied_approved": 2}
    assert out["counts"]["store"] == {"approved": 2, "rejected": 1, "total": 3}
    assert run.decisions.get(k_other)["firm_id"] == 202 and run.decisions.get(k_acme)["firm_id"] == 101
    assert run.decisions.get(k_other)["run_id"] == run.id
    assert run.review(decision="approved")["total"] == 2 and run.review(decision="pending")["total"] == 0
    with pytest.raises(KeyError):
        run.set_decisions([{"key": "FL|FL|KNOWN", "decision": "approved"}])  # AUTO row is not reviewable
    with pytest.raises(ValueError):
        run.set_decisions([{"key": k_acme, "decision": "maybe"}])

    # promote applies pending approvals first; the promoted file contains exactly master + approved rows
    dest = tmp_path / "Master Mapping.csv"
    dest.write_text("x\n")
    res = run.promote_master(str(dest))
    promoted = pd.read_csv(dest)
    assert list(promoted.columns) == ["DEBT_STATE", "SECP_STATE", "SECP_NAME", "SECP_NAME_MAPPED", "Firm ID"]
    assert promoted["SECP_NAME"].tolist() == ["KNOWN", "ACME BK NA", "OTHER CO"]
    assert promoted.set_index("SECP_NAME").loc["OTHER CO", "SECP_NAME_MAPPED"] == "Zeta Bank"
    assert promoted["Firm ID"].tolist() == [5, 101, 202]
    assert res["backup"] and Path(res["backup"]).read_text() == "x\n"
    assert run.decision_counts()["unapplied_approved"] == 0 and run.summary["updated_master"]["approved_rows"] == 2

    # re-applying is idempotent (no duplicate approved rows), and decisions survive a restart
    run.apply_decisions()
    latest = pd.read_csv(run._latest("updated_master_*.csv", exclude="_with_source"))
    assert len(latest) == 3
    run._persist()
    restored = PipelineRun.load(run.run_dir)  # fresh store object on the same file
    assert restored is not None and restored.decisions.counts()["approved"] == 2

    # clearing a decision
    run.set_decisions([{"key": k_zeta, "decision": None}])
    assert run.decision_counts()["pending"] == 1
    assert [h["action"] for h in run.decisions.history(k_zeta)] == ["cleared", "rejected"]

    # a corrected name that is not in the Universe gets a blank Firm ID (not the suggestion's),
    # and the blank must not demote the column to floats ("101.0") in the CSV
    run.set_decisions([{"key": k_zeta, "decision": "approved", "mapped_name": "Unknown Bank"}])
    assert run.decisions.get(k_zeta)["firm_id"] is None
    run.apply_decisions()
    text = Path(run._latest("updated_master_*.csv", exclude="_with_source")).read_text()
    assert ",101\n" in text and ".0" not in text
    assert any(line.endswith(",Unknown Bank,") for line in text.splitlines())


def test_decisions_carry_across_runs(tmp_path: Path):
    """Decided keys are not reviewed again; approvals flow into later runs' masters; rejections are per suggestion."""
    from ucc_engine.decisions import DecisionStore
    from ucc_engine.runner import PipelineRun

    first = _fake_finished_run(tmp_path, "first")
    assert first.decisions.path == tmp_path / "decisions.sqlite"
    first.set_decisions([
        {"key": "CA|CA|ACME BK NA", "decision": "approved"},
        {"key": "NY|NY|ZETA BANK TRUSTEE", "decision": "rejected"},
    ])
    # nothing applied yet in run 1 - the approval is still "owed" to the master
    assert first.decision_counts()["unapplied_approved"] == 1

    # run 2 (same store): the pipeline makes the same suggestions again, except OTHER CO
    second = _fake_finished_run(tmp_path, "second", suggestions={"OTHER CO": "Other Bank"})
    assert second.decisions is first.decisions
    rows = {r["key"]: r for r in second.review()["rows"]}
    assert rows["CA|CA|ACME BK NA"]["decision"] == "approved"
    assert rows["NY|NY|ZETA BANK TRUSTEE"]["decision"] == "rejected"
    assert rows["TX|TX|OTHER CO"]["decision"] is None
    assert second.review(decision="pending")["total"] == 1  # only the never-reviewed row is left to decide
    assert rows["CA|CA|ACME BK NA"]["decided_in_run"] == first.id

    # the run-end hook carries the earlier approval into run 2's updated master
    second._carry_forward_approvals()
    latest = pd.read_csv(second._latest("updated_master_*_with_source.csv"))
    assert latest.set_index("SECP_NAME").loc["ACME BK NA", "_source"] == "reviewed_approved"
    assert second.decision_counts()["unapplied_approved"] == 0
    assert second.summary["updated_master"]["approved_rows"] == 1

    # a rejection is tied to the suggestion: a *different* suggestion comes back for review, with the history shown
    second.set_decisions([{"key": "TX|TX|OTHER CO", "decision": "rejected"}])
    third = _fake_finished_run(tmp_path, "third", suggestions={"OTHER CO": "Completely Different Bank"})
    other = {r["key"]: r for r in third.review()["rows"]}["TX|TX|OTHER CO"]
    assert other["decision"] is None and other["previously_rejected"] == "Other Bank"
    # ...while the same suggestion (ignoring case/spacing) stays hidden
    fourth = _fake_finished_run(tmp_path, "fourth", suggestions={"OTHER CO": "  other   BANK "})
    assert {r["key"]: r for r in fourth.review()["rows"]}["TX|TX|OTHER CO"]["decision"] == "rejected"

    # global listing / export / clear
    store: DecisionStore = first.decisions
    listing = store.list(decision="rejected", search="other")
    assert listing["total"] == 1 and listing["rows"][0]["key"] == "TX|TX|OTHER CO"
    out = tmp_path / "decisions.csv"
    assert store.export_csv(out) == 3 and out.read_text().startswith("key,debt_state,secp_state,secp_name,decision")
    assert store.clear("TX|TX|OTHER CO") and not store.clear("TX|TX|OTHER CO")
    assert store.counts() == {"approved": 1, "rejected": 1, "total": 2}

    # a pre-SQLite decisions.json next to a run is imported once on load
    legacy_run = _fake_finished_run(tmp_path, "legacy")
    legacy_run._persist()
    (legacy_run.run_dir / "decisions.json").write_text(
        '{"TX|TX|OTHER CO": {"decision": "approved", "suggested": "Unknown Co", "mapped_name": null, "firm_id": 7}}')
    loaded = PipelineRun.load(legacy_run.run_dir, decisions=store)
    assert loaded is not None and store.get("TX|TX|OTHER CO")["firm_id"] == 7
    assert not (legacy_run.run_dir / "decisions.json").exists()


def test_api_health_and_validation(tmp_path: Path):
    from fastapi.testclient import TestClient

    from ucc_engine.api import create_app
    from ucc_engine.runner import RunStore

    client = TestClient(create_app(RunStore(tmp_path / "runs")))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["decisions"] == {"approved": 0, "rejected": 0, "total": 0}
    assert client.get("/decisions").json()["total"] == 0
    assert client.post("/decisions/clear", json={"keys": ["X|X|NOPE"]}).json()["cleared"] == 0
    assert client.get("/decisions/export").headers["content-type"].startswith("text/csv")
    assert client.get("/config/defaults").json()["defaults"]["fuzzy_auto_threshold"] == 90
    assert client.get("/runs/nope").status_code == 404
    r = client.post("/runs", json={"source": {"type": "csv", "ucc_path": "/nope.csv", "master_path": "/nope2.csv"}, "sample_rows": -1})
    assert r.status_code == 422  # sample_rows must be >= 0


def test_bucket_classification_and_rows_endpoint(tmp_path: Path):
    """Every result row lands in exactly one of the four pipeline buckets, and the
    /runs/{id}/rows endpoint pages them."""
    from ucc_engine.runner import BUCKETS, bucket_series

    df = pd.DataFrame({
        "DEBT_STATE": ["CA"] * 6, "SECP_STATE": ["CA"] * 6,
        "SECP_NAME": ["A", "B", "C", "D", "E", "F"],
        "SECP_NAME_MAPPED": ["A", "B", "C", "D", None, None],
        "match_method": ["Exact Match (Raw)", "Exact Match (Normalized)", "Fuzzy Match", "Advanced Match", None, "No Master State Pair"],
        "confidence": ["AUTO", "AUTO", "AUTO", "REVIEW_HIGH", "NEW_ENTITY", None],
        "match_score": [1.0, 1.0, 0.97, 0.91, None, None],
    })
    b = bucket_series(df)
    assert list(b) == ["direct", "direct", "fuzzy_high", "review", "unmatched", "unmatched"]
    assert set(b.unique()) <= set(BUCKETS)

    # Ambiguous exact matches are suggestions, not facts -> review.
    amb = df.iloc[[0]].copy()
    amb["match_method"], amb["confidence"] = "Exact Match (Raw - State Ambiguous)", "REVIEW_HIGH"
    assert list(bucket_series(amb)) == ["review"]

    # summary + paging on a fake finished run
    run = _fake_finished_run(tmp_path)
    assert run.summary["buckets"] == {"direct": 1, "fuzzy_high": 0, "review": 3, "unmatched": 0}
    assert run.summary["review_bands"] == {"REVIEW_HIGH": 1, "REVIEW_MEDIUM": 1, "REVIEW_LOW": 1}
    page = run.rows(bucket="direct")
    assert page["total"] == 1 and page["rows"][0]["SECP_NAME"] == "KNOWN"
    assert run.rows(bucket="review", search="zeta")["total"] == 1
    with pytest.raises(KeyError):
        run.rows(bucket="nope")


def test_models_are_copied_from_the_network_share(tmp_path: Path):
    """model_share_dir: files missing/newer on the share are copied into the local
    model folder; up-to-date files are left alone; an unreachable share is fine
    once the local copy exists."""
    import os
    import time
    from ucc_engine.model_sync import sync_models_from_share, sync_status

    share = tmp_path / "share"
    (share / "models--Qdrant--bge-small-en-v1.5-onnx-Q" / "snapshots" / "abc").mkdir(parents=True)
    (share / "models--Qdrant--bge-small-en-v1.5-onnx-Q" / "snapshots" / "abc" / "model.onnx").write_bytes(b"x" * 1000)
    (share / "models--Xenova--ms-marco-MiniLM-L-6-v2").mkdir()
    (share / "models--Xenova--ms-marco-MiniLM-L-6-v2" / "config.json").write_text("{}")
    (share / "unrelated.txt").write_text("ignored")
    local = tmp_path / "local"

    r = sync_models_from_share(str(share), str(local))
    assert r["status"] == "updated" and r["copied"] == 2 and r["bytes"] == 1002
    assert (local / "models--Qdrant--bge-small-en-v1.5-onnx-Q" / "snapshots" / "abc" / "model.onnx").read_bytes() == b"x" * 1000
    assert not (local / "unrelated.txt").exists()

    # second pass: nothing to do
    assert sync_models_from_share(str(share), str(local))["status"] == "up_to_date"

    # a newer file on the share is re-copied
    time.sleep(0.01)
    f = share / "models--Xenova--ms-marco-MiniLM-L-6-v2" / "config.json"
    f.write_text('{"v": 2}')
    os.utime(f, (time.time() + 5, time.time() + 5))
    r = sync_models_from_share(str(share), str(local))
    assert r["copied"] == 1 and (local / "models--Xenova--ms-marco-MiniLM-L-6-v2" / "config.json").read_text() == '{"v": 2}'

    st = sync_status(str(share), str(local))
    assert st["share_reachable"] and len(st["local_models"]) == 2 and st["last_synced"]

    # share gone: keep going on the local copy; share gone and nothing local: hard error
    assert sync_models_from_share(str(tmp_path / "nope"), str(local))["status"] == "share_unreachable"
    with pytest.raises(FileNotFoundError):
        sync_models_from_share(str(tmp_path / "nope"), str(tmp_path / "empty"))

    # the loader resolves to the local copy in offline mode
    from ucc_engine.config import PipelineConfig
    from ucc_engine.embeddings import model_location
    cfg = PipelineConfig(model_share_dir=str(share), model_cache_dir=str(local))
    assert model_location(cfg) == (str(local), str(local))
    assert model_location(PipelineConfig(model_dir=str(share), model_cache_dir=str(local))) == (str(local), str(share))


def test_decisions_propagate_across_state_pairs_and_spellings(tmp_path: Path):
    """One decision on a filer name covers its other state pairs and spelling variants
    when the suggested bank is the same; a different suggested bank stays pending."""
    from ucc_engine.config import PipelineConfig
    from ucc_engine.datasources import CsvSourceSpec
    from ucc_engine.runner import PipelineRun

    universe = tmp_path / "universe.csv"
    universe.write_text("Bank Name,Firm ID\nWells Fargo,1\nPinnacle Bank (GA),2\nPinnacle Bank (NE),3\n")
    spec = CsvSourceSpec(ucc_path=str(tmp_path / "u.csv"), master_path=str(tmp_path / "m.csv"), universe_path=str(universe))
    run = PipelineRun(spec, PipelineConfig(), runs_root=tmp_path / "runs", name="p")
    run.config.universe_file = str(universe)
    run.status = "succeeded"
    results = pd.DataFrame({
        "DEBT_STATE": ["CA", "TX", "NY", "GA", "NE", "CA"],
        "SECP_STATE": ["SD", "SD", "SD", "GA", "NE", "SD"],
        "SECP_NAME": [
            "WELLS FARGO BANK, N.A., AS ADMINISTRIVE AGENT",   # decided by the reviewer
            "WELLS FARGO BANK, NATIONAL ASSOCIATION, AS AGENT",  # same filer, other state pair -> inherits
            "WELLS FARGO BANK NA",                               # spelling variant -> inherits
            "PINNACLE BANK", "PINNACLE BANK",                    # same name, DIFFERENT suggested banks
            "WELLS FARGO BANK, N.A., AS ADMINISTRIVE AGENT",     # a second filing of the first key (dedup)
        ],
        "SECP_NAME_MAPPED": ["Wells Fargo", "Wells Fargo", "Wells Fargo", "Pinnacle Bank (GA)", "Pinnacle Bank (NE)", "Wells Fargo"],
        "match_method": ["Advanced Match"] * 6, "match_score": [0.9] * 6,
        "confidence": ["REVIEW_HIGH"] * 6, "Firm_ID": [1, 1, 1, 2, 3, 1],
    })
    run._cache["universe_enriched_results"] = results
    master = pd.DataFrame({"DEBT_STATE": ["FL"], "SECP_STATE": ["FL"], "SECP_NAME": ["KNOWN"],
                           "SECP_NAME_MAPPED": ["Known"], "Firm ID": [5], "_source": ["original_master"]})
    run._cache["updated_master_data"] = master
    run.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(run.output_dir / "updated_master_20260101_000000_with_source.csv", index=False)
    run.summary = run._build_summary()

    page = run.review()
    assert page["total"] == 5, "review rows are one per key"
    first = next(r for r in page["rows"] if r["DEBT_STATE"] == "CA")
    assert first["filings"] == 2

    out = run.set_decisions([{"key": first["key"], "decision": "approved"}])
    assert out["changed"] == 1 and out["inherited"] == {"approved": 2, "rejected": 0, "candidates": 2}
    rows = {r["DEBT_STATE"]: r for r in run.review()["rows"]}
    assert rows["TX"]["decision"] == "approved" and rows["TX"]["decision_source"].startswith("inherited:")
    assert rows["NY"]["decision"] == "approved"
    assert rows["GA"]["decision"] is None and rows["NE"]["decision"] is None  # pending is null on the wire

    # the two PINNACLE rows suggest different banks: approving one must not decide the other
    run.set_decisions([{"key": rows["GA"]["key"], "decision": "approved"}])
    rows = {r["DEBT_STATE"]: r for r in run.review()["rows"]}
    assert rows["GA"]["decision"] == "approved" and rows["NE"]["decision"] is None

    # groups roll up by bank, biggest first
    groups = run.review_groups(decision="")
    assert [g["mapped_name"] for g in groups][:1] == ["Wells Fargo"]
    wf = groups[0]
    assert wf["keys"] == 3 and wf["filings"] == 4 and wf["approved"] == 3
    assert run.review(mapped="Wells Fargo")["total"] == 3

    # inherited approvals are real decisions: they reach the updated master and can be undone per key
    res = run.apply_decisions()
    assert res["added"] == 4
    run.set_decisions([{"key": rows["TX"]["key"], "decision": None}])
    assert {r["DEBT_STATE"]: r["decision"] for r in run.review()["rows"]}["TX"] is None
    assert [h["action"] for h in run.decisions.history(rows["NY"]["key"])][-1].startswith("approved (inherited:")


def test_manual_mapping_of_unmatched_rows(tmp_path: Path):
    """An unmatched filer can be mapped by hand to a bank from the Universe; 'all states'
    spreads it to the same filer name elsewhere; it reaches the updated master; the
    provenance view shows what came in vs what was added."""
    from ucc_engine.config import PipelineConfig
    from ucc_engine.datasources import CsvSourceSpec
    from ucc_engine.runner import PipelineRun

    universe = tmp_path / "universe.csv"
    universe.write_text("Bank Name,Firm ID\nJPMorgan,7\nUS Bank,8\n")
    spec = CsvSourceSpec(ucc_path=str(tmp_path / "u.csv"), master_path=str(tmp_path / "m.csv"), universe_path=str(universe))
    run = PipelineRun(spec, PipelineConfig(), runs_root=tmp_path / "runs", name="m")
    run.config.universe_file = str(universe)
    run.status = "succeeded"
    results = pd.DataFrame({
        "DEBT_STATE": ["NY", "TX", "CA", "FL"], "SECP_STATE": ["TX", "TX", "TX", "FL"],
        "SECP_NAME": ["JPMORGAN CHASE BANK, N.A., TORONTO BRANCH", "JPMORGAN CHASE BANK NA TORONTO BRANCH", "ACME WIDGETS LLC", "KNOWN"],
        "SECP_NAME_NORMALIZED": ["JPMORGAN CHASE BANK", "JPMORGAN CHASE BANK", "ACME WIDGETS", "KNOWN"],
        "SECP_NAME_MAPPED": [None, None, None, "Known"],
        "match_method": [None, None, None, "Exact Match (Raw)"], "match_score": [None, None, None, 1.0],
        "confidence": ["NEW_ENTITY", "NEW_ENTITY", "NEW_ENTITY", "AUTO"], "Firm_ID": [None, None, None, 5],
    })
    run._cache["universe_enriched_results"] = results
    master = pd.DataFrame({"DEBT_STATE": ["FL"], "SECP_STATE": ["FL"], "SECP_NAME": ["KNOWN"],
                           "SECP_NAME_MAPPED": ["Known"], "Firm ID": [5], "_source": ["original_master"]})
    run._cache["updated_master_data"] = master
    run.output_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(run.output_dir / "updated_master_20260101_000000_with_source.csv", index=False)
    run.summary = run._build_summary()

    banks = run.banks()
    assert [b["name"] for b in banks] == ["JPMorgan", "Known", "US Bank"] and banks[0]["firm_id"] == 7

    page = run.rows(bucket="unmatched")
    assert page["total"] == 3 and all(r["decision"] is None for r in page["rows"])
    ny = next(r for r in page["rows"] if r["DEBT_STATE"] == "NY")

    # can't reject an unmatched row, and a mapping needs a bank
    with pytest.raises(ValueError):
        run.set_decisions([{"key": ny["key"], "decision": "rejected"}])
    with pytest.raises(ValueError):
        run.set_decisions([{"key": ny["key"], "decision": "approved"}])

    out = run.set_decisions([{"key": ny["key"], "decision": "approved", "mapped_name": "JPMorgan", "all_states": True}])
    assert out["changed"] == 1 and out["inherited"]["approved"] == 1  # the TX spelling variant inherited it
    rows = {r["DEBT_STATE"]: r for r in run.rows(bucket="unmatched")["rows"]}
    assert rows["NY"]["decision"] == "approved" and rows["NY"]["decided_mapped_name"] == "JPMorgan" and rows["NY"]["decided_firm_id"] == 7
    assert rows["TX"]["decision"] == "approved" and str(rows["TX"]["decision_source"]).startswith("inherited:")
    assert rows["CA"]["decision"] is None
    assert run.decisions.get(ny["key"])["match_method"] == "Manual (all states)"

    # a key-only mapping does not spread
    ca = rows["CA"]
    run.set_decisions([{"key": ca["key"], "decision": "approved", "mapped_name": "US Bank"}])
    assert run.decisions.get(ca["key"])["match_method"] == "Manual"

    # manual mappings reach the updated master and the provenance view shows them as added
    res = run.apply_decisions()
    assert res["added"] == 3
    prov = run.master_rows(source="added")
    assert prov["total"] == 3 and prov["by_source"] == {"original_master": 1, "reviewed_approved": 3}
    assert run.master_rows(source="original")["total"] == 1
    assert {r["SECP_NAME_MAPPED"] for r in prov["rows"]} == {"JPMorgan", "US Bank"}


def test_stage_2b_matches_a_national_bank_across_state_pairs_into_review():
    """A normalised name that exists in the master only under OTHER state pairs, and maps
    to a single bank nationally, becomes a review suggestion; names that map to several
    banks (state-dependent) do not."""
    from ucc_engine.compat import AssetExecutionContext
    from ucc_engine.core import pipeline_logic as L

    ctx = AssetExecutionContext("t", sink=lambda *_: None)
    master = pd.DataFrame({
        "DEBT_STATE": ["CA", "NY", "GA", "NE"], "SECP_STATE": ["CA", "NY", "GA", "NE"],
        "SECP_NAME": ["U.S. BANK TRUST COMPANY, NATIONAL ASSOCIATION", "US BANK TRUST COMPANY NA", "PINNACLE BANK", "PINNACLE BANK"],
        "SECP_NAME_MAPPED": ["US Bank", "US Bank", "Pinnacle Bank (GA)", "Pinnacle Bank (NE)"], "Firm ID": [8, 8, 2, 3],
    })
    norm = L.normalized_master_data(ctx, master)
    assert norm["global_norm_to_mapped"]["US BANK TRUST"][0] == "US Bank"
    assert "PINNACLE BANK" not in norm["global_norm_to_mapped"]

    ucc = pd.DataFrame({
        "DEBT_STATE": ["AR", "AR", "TX"], "SECP_STATE": ["IL", "IL", "TX"],
        "SECP_NAME": ["US BANK TRUST COMPANY NA AS COLLATERAL AGENT", "PINNACLE BANK", "SOMETHING ELSE LLC"],
    })
    for c in ("SECP_NAME_MAPPED", "matched_master_raw", "Firm_ID", "match_score", "match_method", "confidence", "ambiguous_master_candidates"):
        ucc[c] = None
    out = L.exact_match_normalized_results(ctx, L.exact_match_raw_results(ctx, ucc, norm), norm)
    us = out.iloc[0]
    assert us["SECP_NAME_MAPPED"] == "US Bank" and us["confidence"] == "REVIEW_HIGH"
    assert us["match_method"] == "Exact Match (Normalized - Other State)" and us["Firm_ID"] == 8
    assert pd.isna(out.iloc[1]["SECP_NAME_MAPPED"]), "state-dependent names must not match across states"
    assert pd.isna(out.iloc[2]["SECP_NAME_MAPPED"])
