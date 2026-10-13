# ucc-engine

Headless version of the UCC → master-mapping pipeline, with a local HTTP API.
It is the backend ("sidecar") of the desktop app in `../desktop`, and also a
plain CLI.

* **`src/ucc_engine/core/` is the matching pipeline** - normalisation, the
  seven stages and the output writers. It began life as a Dagster project; the
  Dagster decorators are now no-op shims in `compat.py` and the code is edited
  here directly. `docs/PIPELINE_DOCUMENTATION.md` describes the stages.
* **No Dagster, no network inference.** The two small models (the
  `bge-small-en-v1.5` embedder and the `ms-marco-MiniLM` cross-encoder) run on
  the local CPU through [`fastembed`](https://github.com/qdrant/fastembed)
  (ONNX Runtime). Install the `st` extra to use the original torch
  `sentence-transformers` models instead. There is no LLM anywhere.
* **Inputs are pluggable.** `CsvDataSource` today; `SqlDataSource` (SQLAlchemy,
  three queries) is implemented and waiting for a connection string.

## Setup

```bash
cd engine
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,sql,build]"
source .venv/bin/activate

ucc-engine prefetch-models            # one-time download into the local model cache
pytest                                # fast plumbing tests (no models needed)
```

## CLI

```bash
# whole pipeline on CSVs, streaming the log; add --sample 3000 for a quick run
ucc-engine run --ucc Inputs/ucc.csv --master "Inputs/Master Mapping.csv" \
               --universe Inputs/Universe.csv --sample 3000 --set fuzzy_auto_threshold=95

# local API for the desktop app (prints UCC_ENGINE_PORT=<n> first)
ucc-engine serve --port 8765
```

Runs live under the per-user app data dir (`~/Library/Application Support/ucc-engine/runs/<id>/`
on macOS, `%LOCALAPPDATA%\ucc-engine` on Windows; override with `UCC_ENGINE_HOME`). In
development the sidecar stub points `UCC_ENGINE_HOME` at `<repo>/output` and
`UCC_MODELS_DIR` at `<repo>/models`, so everything a dev run produces stays in the
repository folder. Each run has `inputs/` (only when
sampled or from SQL), `outputs/` (the same files the Dagster `pipeline_outputs`
asset writes, plus `updated_master_*.csv`, `merged_raw_with_firm_id_*.csv`,
lead-bank files) and `run.json` (status, timings, summary, log tail).

## HTTP API (127.0.0.1 only)

| Method & path | Purpose |
|---|---|
| `GET /health` | version, available backends, active run |
| `GET /config/defaults` | `PipelineConfig` defaults + schema for a settings form |
| `POST /models/prefetch` | download models now (or, with `model_share_dir`, copy them from the share) |
| `GET /models/status` | is the network model share reachable, what is local, when last copied |
| `POST /models/sync` | copy the models from `model_share_dir` into the local model folder now |
| `POST /sources/preview` | validate a CSV/SQL source: columns present/missing, row counts, 5-row preview |
| `GET/POST /runs` | list runs / start one `{source, config, sample_rows, name}` (409 if one is running) |
| `GET /runs/{id}` | status, progress (0-1), per-stage state, summary |
| `GET /runs/{id}/logs?after=N` | incremental log lines |
| `POST /runs/{id}/cancel` | stop after the current stage |
| `GET /runs/{id}/summary` | the numbers from the Dagster summary file, as JSON |
| `GET /runs/{id}/rows?bucket=` | read-only page of one result bucket: `direct` / `fuzzy_high` / `review` / `unmatched` (the summary carries the counts under `buckets`) |
| `GET /runs/{id}/review/groups` | the review queue rolled up by suggested bank (`decision`, `search`) |
| `POST /runs/{id}/propagate-decisions` | inherit decisions made on the same filer name elsewhere |
| `GET /runs/{id}/review` | paged review queue (`confidence`, `search`, `method`, `decision` = `pending`/`approved`/`rejected`, `offset`, `limit`); each row carries its `key` and current decision |
| `GET/PUT /runs/{id}/decisions` | read / record review decisions: `{items: [{key, decision: "approved"\|"rejected"\|null, mapped_name?, firm_id?}]}` |
| `POST /runs/{id}/apply-decisions` | append reviewer rows to a fresh `updated_master_<ts>.csv` (`_source = reviewed_approved` for approved suggestions, `mapped_by_hand` for banks picked for unmatched filers) |
| `GET /decisions` | the decisions database across all runs (`decision`, `search`, `run_id`, `offset`, `limit`) |
| `POST /decisions/clear` | `{keys: [...]}` – undo decisions (they become "to decide" again) |
| `GET /decisions/history?key=` | audit log for one key |
| `GET/POST /decisions/export` | CSV of all decisions (download / `{destination}` to write a file) |
| `GET /runs/{id}/clusters` | new-entity clusters, one row per cluster |
| `GET /runs/{id}/outputs[/{name}]` | list / download output files |
| `POST /runs/{id}/promote-master` | apply any pending approvals, then copy this run's `updated_master` over a destination (timestamped backup kept) |
| `DELETE /runs/{id}` | remove a finished run |

### Reviewing suggestions

Rows in the `REVIEW_*` confidence bands are suggestions, not facts, so they are
*not* added to the updated master automatically. The reviewer approves (optionally
after correcting the mapped name), rejects, or leaves them pending.

Decisions are keyed by `DEBT_STATE|SECP_STATE|SECP_NAME` and stored in one SQLite
database shared by every run, `<engine home>/decisions.sqlite` (tables
`decisions` – current state – and `decision_log` – append-only audit). Stdlib
`sqlite3`, nothing extra to install; open it with any SQLite viewer. Consequences:

- A key decided in any run is never "to decide" again; later runs show it as
  approved / rejected with the run it was decided in.
- A rejection is of a *specific suggestion*. If a later run proposes a different
  mapped name for the same key, that row comes back as pending with
  `previously_rejected` set, so the new suggestion gets looked at once.
- Approved rows are carried into every later run's updated master automatically
  (at run end, and on `apply-decisions` / `promote-master`) until the promoted
  Master Mapping contains them, after which Stage 1 matches them exactly.
- Approving writes `SECP_NAME → mapped name (Firm ID)`; the Firm ID is looked up
  in the Universe by the final mapped name and left blank if that name is
  unknown. Original master and pipeline (Stage 1–3) rows always win over an
  approved row with the same key.
- **Decisions propagate.** A decision on a filer name is inherited by every
  pending row with the same *normalised* name (other state pairs, spelling
  variants, later runs) **when the suggested bank is the same bank**; a row
  suggesting a different bank stays pending (so `PINNACLE BANK` in GA does not
  inherit an approval for Pinnacle Bank (NE)). Inherited decisions are written
  per key with `source = inherited:<origin key>`, show as *approved · inherited*,
  flow into the updated master like any approval, and can be undone per key -
  an undone key is never re-inherited. Propagation runs after every decision,
  at the end of every run, and on `POST /runs/{id}/propagate-decisions`.
- Review rows are one per key (`filings` carries how many filings share it);
  `GET /runs/{id}/review/groups` rolls the queue up by suggested bank, biggest
  first, and `review?mapped=<bank>` pages one bank's rows - the desktop app's
  *By bank* view approves a whole block at once.
- **Unmatched rows can be mapped by hand** (`GET /runs/{id}/banks` lists the Universe
  + the master's banks; `PUT decisions` with `mapped_name` on an unmatched key). With
  `all_states: true` the mapping is inherited by the same filer name wherever it is
  unmatched, now and in later runs (`match_method = "Manual (all states)"`); without
  it, it sticks to its key. Mapped rows reach the updated master like approvals.
- `GET /runs/{id}/master?source=original|added|<_source>` pages the updated master with
  provenance: what came in as the mapping vs what the pipeline / reviewers added.
- A per-run `decisions.json` from older engine versions is imported into the
  database the first time the run is loaded (renamed `.json.imported`).

## Matching quality

`tests/test_match_quality.py` is a hold-out test on the real Master Mapping:
1,500 known rows are removed and their raw names fed back through the Stage 3
logic against the rest. It asserts floors on precision / recall at the auto
threshold, so a change to `normalize_name`, the guardrails or the thresholds
is measured rather than guessed. Baseline (2026-10-12, seed 42):

| | precision @ auto | recall @ auto | recall, any score |
|---|---|---|---|
| before (`fuzzy_auto_threshold=96`, old normaliser) | 0.990 | 0.551 | 0.856 |
| now (`=90`, pre-clean normaliser, order guardrail) | 0.990 | 0.719 | 0.867 |

What changed, and why (all in `core/utils.py` + `core/constants.py`):

* `pre_clean_name()` runs before normalisation: drops parentheticals, picks the
  lender out of "X, A DIVISION OF Y" / "FDIC AS RECEIVER OF Y", keeps the current
  entity in "X, SUCCESSOR BY MERGER TO Y" / "X F/K/A Y" / "X DBA Y", cuts
  capacity tails the role regex missed ("AS ADMINISTRIVE AGENT", ", A TEXAS STATE
  BANKING ASSOCIATION", "NOT IN ITS INDIVIDUAL CAPACITY ..."), strips branch
  qualifiers ("PARK BANK- BROOKFIELD") and mid-string charter words (N.A., FSB).
* **Stage 2b** (in `exact_match_normalized_results`): a normalised name that is in the
  master only under *other* state pairs and maps to one bank nationally becomes a
  review suggestion (`Exact Match (Normalized - Other State)`). Never auto: a name
  that maps to one bank today may only be the first of that name seen. Names that
  map to several banks (PINNACLE BANK) are skipped. Tonight's run: 1,662 filings /
  194 names (HSBC, KeyBank, US Bank Trust, ...).
* Split initialisms are joined after normalisation (`U.S. BANK` = `U S BANK` =
  `US BANK`, `J.P. MORGAN` = `JP MORGAN`); branch qualifiers (`TORONTO BRANCH`),
  `C/O ...` and typo'd `SUCESSOR IN ITEREST` tails are stripped. Hold-out recall at
  auto: 0.719 -> 0.769, precision unchanged.
* The updated master takes **AUTO rows of any method** (it used to take a list of
  match methods, which let unreviewed `Fuzzy Match (Ambiguous)` rows in and kept AUTO
  `Advanced Match` rows out).
* Guardrail 6 in `combined_fuzzy_score`: same words in a different order
  ("NATIONAL CITY BANK" vs "CITY NATIONAL BANK") no longer reach the review floor
  on fuzzy alone.
* `fuzzy_auto_threshold` 96 → 90: precision is flat between the two on the
  hold-out; recall rises.
* `advanced_cross_encoder_promote_to_auto` 0.92 → 1.01: the reranker is
  review-only. In the June 2026 run it had promoted 983 REVIEW_MEDIUM names to
  AUTO and a hand-labelled sample put ~25% of them on the wrong bank. Those rows
  now land in the review queue (decisions are remembered across runs).

## Configuration

`config.PipelineConfig` mirrors the Dagster `resources.PipelineConfig`
field-for-field (thresholds, clustering, performance, lead-bank, rerank), plus:

| Field | Default | Meaning |
|---|---|---|
| `clustering_enabled` | `false` | run Stages 5-7 (cluster unmatched rows, suggest a mapping per cluster). Off: unmatched rows are still labelled `NEW_ENTITY` and exported, without cluster columns; the stages show as *skipped* |
| `embedding_backend` | `auto` | `fastembed` if installed, else `sentence-transformers` |
| `embedding_model` | `BAAI/bge-small-en-v1.5` | 384-dim; keep in sync with any persisted query cache |
| `model_dir` | `""` | read the models *in place* from this folder (local path or network share); when set nothing is downloaded or copied |
| `model_share_dir` | `""` | network share holding both models; anything missing or newer is **copied** into `model_cache_dir` before the first load, then the local copy is used offline (see below) |
| `model_cache_dir` | `<app data>/models` | where models are downloaded when `model_dir` is not set |
| `model_threads` | `0` | ONNX / torch intra-op threads (0 = library default) |

## Packaging the sidecar

```bash
python scripts/build_sidecar.py --dev-stub   # dev: shell stub -> engine/.venv (for `tauri dev`)
python scripts/build_sidecar.py              # release: PyInstaller one-file binary ->
                                             #   ../desktop/src-tauri/binaries/ucc-engine-<target-triple>
```

### Where the models live

The models are plain files (~150 MB: an ONNX graph + tokenizer each) that ONNX
Runtime loads into the engine process once per session. There is no model
server and no network inference. Options:

1. **Default** - downloaded on first use (or via *Prepare models*) into
   `model_cache_dir` (`%LOCALAPPDATA%\ucc-engine\models` on Windows,
   `~/Library/Application Support/ucc-engine/models` on macOS). Needs internet once per PC.
2. **Network share, copied to each PC (recommended)** - prepare the share once,
   from any machine with internet:

   ```bash
   ucc-engine prefetch-models --set "model_cache_dir=\\\\fileserver\\apps\\ucc-models"
   ```

   then set `model_share_dir` to that path on every PC (the *Network model
   folder* on the app's Ingest panel, or `--set model_share_dir=...`). The
   engine mirrors the `models--*` folders into the local `model_cache_dir`
   (`ucc-engine sync-models --share ...` does it on demand); files are
   re-copied only when the share has a newer / different-size version, and a
   PC that already has the files keeps working when the share is unreachable.
   ~150 MB crosses the LAN once per PC, not once per session.
3. **Network share, read in place** - run once, from any machine with internet:

   ```bash
   ucc-engine prefetch-models --set "model_cache_dir=\\\\fileserver\\apps\\ucc-models"
   ```

   then set `model_dir` to that path on every PC (advanced settings in the app,
   or `--set model_dir=...`). Files are only read; ~150 MB crosses the LAN per
   app session, the share can be read-only, and no PC needs internet.
4. **Share as distribution point** - copy that folder to each PC's
   `model_cache_dir` yourself (login script / installer); same result as 2.

The folder layout is fastembed's cache layout (`models--Qdrant--bge-small-en-v1.5-onnx-Q/`,
`models--Xenova--ms-marco-MiniLM-L-6-v2/`); `prefetch-models` writes exactly that.
