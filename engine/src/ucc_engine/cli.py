"""``ucc-engine`` command line.

    ucc-engine serve [--port 0] [--host 127.0.0.1] [--exit-on-stdin-close]
        Start the local API. With --port 0 a free port is chosen and printed as
        the first stdout line: ``UCC_ENGINE_PORT=<n>`` (the Tauri shell reads it).
        ``--exit-on-stdin-close`` ties the server's life to the parent's stdin pipe.

    ucc-engine run --ucc ucc.csv --master "Master Mapping.csv" [--universe Universe.csv]
                   [--out DIR] [--sample N] [--set key=value ...]
        Run the whole pipeline headless and print the summary.

    ucc-engine prefetch-models [--set key=value ...]
        Download the embedding + reranker models into the local model cache.

    ucc-engine sync-models --share \\\\server\\share\\ucc-models
        Copy the models from a network share into the local model folder.
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from . import __version__


def _kv(pairs: List[str] | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _exit_when_stdin_closes() -> None:
    """Terminate the process as soon as the host closes our stdin pipe.

    The desktop shell holds the write end of that pipe for its whole life, so
    EOF means the app quit, crashed or was force-quit. This works even when the
    frozen (PyInstaller one-file) bootloader was SIGKILLed and so could not
    forward the signal to this Python child, which would otherwise leave an
    orphaned server behind.
    """
    import os
    import threading

    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except Exception:  # noqa: BLE001 - treat any stdin failure as "host gone"
            pass
        os._exit(0)

    threading.Thread(target=watch, name="stdin-watchdog", daemon=True).start()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app
    from .runner import RunStore

    if args.exit_on_stdin_close:
        _exit_when_stdin_closes()
    port = args.port or _free_port(args.host)
    # Announce the port before the server starts so the parent can connect.
    print(f"UCC_ENGINE_PORT={port}", flush=True)
    app = create_app(RunStore(Path(args.runs_dir) if args.runs_dir else None))
    uvicorn.run(app, host=args.host, port=port, log_level=args.log_level, access_log=False)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .datasources import CsvSourceSpec
    from .runner import RunStore

    spec = CsvSourceSpec(ucc_path=args.ucc, master_path=args.master, universe_path=args.universe or "")
    cfg = PipelineConfig.from_overrides(_kv(args.set))
    store = RunStore(Path(args.out) if args.out else None)
    run = store.create(spec, cfg, sample_rows=args.sample, name=args.name or "").start()

    last = 0
    while run.status in ("queued", "running"):
        for line in run.logs(after=last):
            last = line["seq"]
            if line["level"] != "DEBUG":
                print(f"{line['ts']} {line['level']:<7} [{line['stage']}] {line['message']}", flush=True)
        time.sleep(0.5)
    for line in run.logs(after=last):
        print(f"{line['ts']} {line['level']:<7} [{line['stage']}] {line['message']}", flush=True)

    print(f"\nstatus: {run.status}" + (f"  error: {run.error}" if run.error else ""))
    print(f"outputs: {run.output_dir}")
    if run.summary:
        print(json.dumps(run.summary, indent=2, default=str))
    return 0 if run.status == "succeeded" else 1


def cmd_prefetch(args: argparse.Namespace) -> int:
    from .config import PipelineConfig
    from .embeddings import prefetch_models

    print(json.dumps(prefetch_models(PipelineConfig.from_overrides(_kv(args.set))), indent=2))
    return 0


def cmd_sync_models(args: argparse.Namespace) -> int:
    """Copy the models from the network share into this PC's model folder."""
    from .config import PipelineConfig
    from .model_sync import sync_models_from_share

    cfg = PipelineConfig.from_overrides({**_kv(args.set), "model_share_dir": args.share or ""})
    if not cfg.model_share_dir:
        print("error: give the share with --share \\\\server\\share\\ucc-models (or --set model_share_dir=...)", file=sys.stderr)
        return 2
    print(json.dumps(sync_models_from_share(cfg.model_share_dir, cfg.model_cache_dir, lambda _l, m: print(m)), indent=2))
    return 0


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ucc-engine", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the local HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    s.add_argument("--runs-dir", default="", help="override the runs directory")
    s.add_argument("--log-level", default="info")
    s.add_argument("--exit-on-stdin-close", action="store_true",
                   help="quit when stdin reaches EOF (the desktop shell passes this so the server dies with the app)")
    s.set_defaults(fn=cmd_serve)

    r = sub.add_parser("run", help="run the pipeline headless on CSV inputs")
    r.add_argument("--ucc", required=True)
    r.add_argument("--master", required=True)
    r.add_argument("--universe", default="")
    r.add_argument("--out", default="", help="runs directory (default: app data dir)")
    r.add_argument("--sample", type=int, default=0, help="random sample of N UCC rows")
    r.add_argument("--name", default="")
    r.add_argument("--set", action="append", metavar="KEY=VALUE", help="PipelineConfig override")
    r.set_defaults(fn=cmd_run)

    f = sub.add_parser("prefetch-models", help="download local models into the cache")
    f.add_argument("--set", action="append", metavar="KEY=VALUE")
    f.set_defaults(fn=cmd_prefetch)

    m = sub.add_parser("sync-models", help="copy the models from a network share into the local model folder")
    m.add_argument("--share", default="", help="network share / UNC path holding the models (model_share_dir)")
    m.add_argument("--set", action="append", metavar="KEY=VALUE")
    m.set_defaults(fn=cmd_sync_models)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":  # python -m ucc_engine.cli
    raise SystemExit(main())
