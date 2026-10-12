# UCC Desktop Application

Maps the secured-party names on UCC filings to a canonical bank mapping, enriches
them with Firm IDs, and produces the updated mapping and lead-bank outputs - as a
desktop app (Windows, built on macOS) with everything running locally.

```
input/     UCC filings, the current (enriched) Master Mapping, the bank Universe
output/    runs, updated mappings, the review-decisions database        (not in git)
models/    local copy of the two matching models; deployed from a network share (not in git)
engine/    Python matching engine + local HTTP API (the app's sidecar; also a CLI)
desktop/   Tauri 2 + React desktop app
docs/      reference documentation
.github/   Windows installer build (GitHub Actions)
```

## Pipeline

```
Ingest -> Direct match -> Fuzzy match -> | High probability  (auto-accepted)
                                         | Review            (a person decides, by bank)
                                         | Unmatched         (left as filed)     -> Outputs
```

* **Direct match**: exact on the filed name, then on the normalised name.
* **Fuzzy match**: token-level string similarity, then embedding + multi-signal
  scoring; a cross-encoder reranks the medium band (review-only).
* Every result lands in one of the three buckets; approved review rows are
  remembered across runs and inherited by the same filer name elsewhere.

## Develop (macOS)

```bash
cd engine
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,sql,build]"
.venv/bin/python -m pytest -q                       # includes a hold-out quality test on input/Master Mapping.csv
.venv/bin/python scripts/build_sidecar.py --dev-stub # engine runs from .venv; output/ and models/ are used
cd ../desktop && npm install && npm run app:dev
```

Drop `input/ucc.csv`, `input/Master Mapping.csv` and `input/Universe.csv` onto
the Ingest step (or any three CSVs), then *Run pipeline*. The first run needs
the models: click *Download models* (internet) or set the network model folder.

## Build the Windows installer

Pushes to `main` that touch `engine/` or `desktop/` build
`UCC Mapping_<version>_x64-setup.exe` on GitHub Actions (see
`.github/workflows/windows-installer.yml`); `v*` tags attach it to a Release.
Local builds: `desktop/README.md`.

## Where things are documented

* `engine/README.md` - API, configuration, matching-quality baseline, models on a network share, review decisions
* `desktop/README.md` - the app, drag-and-drop, building installers
* `docs/PIPELINE_DOCUMENTATION.md` - stage-by-stage description of the matching logic (written for the original Dagster version; the logic is unchanged)
