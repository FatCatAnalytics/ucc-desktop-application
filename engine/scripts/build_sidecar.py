#!/usr/bin/env python3
"""Freeze the engine into a single executable and drop it where Tauri expects it.

    python engine/scripts/build_sidecar.py            # -> desktop/src-tauri/binaries/ucc-engine-<triple>
    python engine/scripts/build_sidecar.py --no-copy  # leave it in engine/dist/

Run from the engine's own environment (``uv pip install -e ".[build]"``). The
result bundles Python, pandas, faiss, rapidfuzz, fastembed/onnxruntime and the
API server - roughly 250-350 MB, no torch. Model *weights* are not bundled; they
download to the user's model cache on first use (or ship them separately and set
``model_dir``).
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ENGINE = HERE.parents[1]
REPO = ENGINE.parent
BINARIES = REPO / "desktop" / "src-tauri" / "binaries"

LAUNCHER = ENGINE / "build" / "launcher.py"
LAUNCHER_SRC = '''\
"""PyInstaller entry point for the ucc-engine sidecar."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from ucc_engine.cli import main
    sys.exit(main())
'''

# Modules PyInstaller's static analysis misses (dynamic imports).
HIDDEN = [
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto", "uvicorn.loops.asyncio",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl", "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "uvicorn.lifespan.off",
    "tabulate", "sqlalchemy.dialects.sqlite",
]
COLLECT_ALL = ["fastembed", "onnxruntime", "tokenizers", "huggingface_hub", "faiss", "rapidfuzz"]
# Packages whose dist-info must ship: pandas checks ``tabulate.__version__`` (read via importlib.metadata,
# "unknown" without metadata -> InvalidVersion), and the others version-gate themselves the same way.
COPY_METADATA = ["tabulate", "pandas", "numpy", "fastembed", "onnxruntime", "tokenizers", "huggingface_hub",
                 "sqlalchemy", "fastapi", "uvicorn", "pydantic", "pydantic_core", "ucc_engine"]


def host_triple() -> str:
    """Rust target triple of this machine (what Tauri appends to sidecar names)."""
    try:
        out = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("host:"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001 - fall back to a best guess
        pass
    arch = {"arm64": "aarch64", "x86_64": "x86_64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin"
    if sys.platform.startswith("win"):
        return f"{arch}-pc-windows-msvc"
    return f"{arch}-unknown-linux-gnu"


def write_dev_stub() -> int:
    """Install a tiny wrapper at the sidecar path that runs the engine from engine/.venv.

    Tauri's build script insists the sidecar file exists, and during development
    you want code changes picked up without re-freezing. The release build
    (`build_sidecar.py` without flags) overwrites this stub with the real binary.
    """
    if sys.platform.startswith("win"):
        print("Dev stub is POSIX-only; on Windows build the real sidecar or set UCC_ENGINE_CMD.", file=sys.stderr)
        return 2
    venv_exe = ENGINE / ".venv" / "bin" / "ucc-engine"
    if not venv_exe.exists():
        print(f"{venv_exe} not found - create the engine venv first (see engine/README.md)", file=sys.stderr)
        return 2
    BINARIES.mkdir(parents=True, exist_ok=True)
    target = BINARIES / f"ucc-engine-{host_triple()}"
    # Development layout: runs / decisions under <repo>/output, models under <repo>/models.
    # The packaged app sets neither, so it uses the per-user app-data folder (+ the network model share).
    target.write_text(
        "#!/bin/sh\n# dev stub - replaced by build_sidecar.py for release builds\n"
        f'export UCC_ENGINE_HOME="${{UCC_ENGINE_HOME:-{REPO / "output"}}}"\n'
        f'export UCC_MODELS_DIR="${{UCC_MODELS_DIR:-{REPO / "models"}}}"\n'
        f'exec "{venv_exe}" "$@"\n'
    )
    target.chmod(0o755)
    print(f"dev stub -> {target.relative_to(REPO)}  (runs {venv_exe.relative_to(REPO)})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-copy", action="store_true")
    ap.add_argument("--onedir", action="store_true", help="faster startup, but not a single file (debugging)")
    ap.add_argument("--dev-stub", action="store_true", help="write a shell stub that runs engine/.venv (no freeze)")
    args = ap.parse_args()

    if args.dev_stub:
        return write_dev_stub()

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller missing: run  uv pip install -e '.[build]'  in engine/", file=sys.stderr)
        return 2

    LAUNCHER.parent.mkdir(exist_ok=True)
    LAUNCHER.write_text(LAUNCHER_SRC)

    dist = ENGINE / "dist"
    work = ENGINE / "build" / "pyinstaller"
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", "ucc-engine",
        "--onedir" if args.onedir else "--onefile",
        "--console",
        "--distpath", str(dist), "--workpath", str(work), "--specpath", str(work),
        "--paths", str(ENGINE / "src"),
        # No GUI / scientific extras the engine never imports.
        "--exclude-module", "torch", "--exclude-module", "tkinter", "--exclude-module", "matplotlib",
        "--exclude-module", "IPython", "--exclude-module", "pytest",
    ]
    for m in HIDDEN:
        cmd += ["--hidden-import", m]
    for pkg in COLLECT_ALL:
        cmd += ["--collect-all", pkg]
    for pkg in COPY_METADATA:
        cmd += ["--copy-metadata", pkg]
    cmd.append(str(LAUNCHER))

    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ENGINE)

    exe = dist / ("ucc-engine.exe" if sys.platform.startswith("win") else "ucc-engine")
    if args.onedir:
        exe = dist / "ucc-engine" / exe.name
    if not exe.exists():
        print(f"build finished but {exe} not found", file=sys.stderr)
        return 1
    print(f"built {exe} ({exe.stat().st_size / 1e6:.0f} MB)")

    if not args.no_copy and not args.onedir:
        BINARIES.mkdir(parents=True, exist_ok=True)
        target = BINARIES / f"ucc-engine-{host_triple()}{exe.suffix}"
        shutil.copy2(exe, target)
        target.chmod(0o755)
        print(f"copied -> {target.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
