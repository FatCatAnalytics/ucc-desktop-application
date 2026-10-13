//! Tauri shell for the UCC mapping engine.
//!
//! Responsibilities (kept deliberately small - all pipeline logic lives in the
//! Python `ucc-engine` sidecar):
//!   * start the engine, discover the port it announces, stop it on exit
//!   * expose that URL to the webview (`engine_url`)
//!   * native file dialogs and "reveal in Finder/Explorer" via Rust commands
//!
//! Engine resolution order (lets `tauri dev` work without a frozen binary):
//!   1. `UCC_ENGINE_URL`  - connect to an already-running engine
//!   2. `UCC_ENGINE_CMD`  - spawn this executable (e.g. engine/.venv/bin/ucc-engine)
//!   3. bundled sidecar   - `binaries/ucc-engine-<target-triple>`

use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_opener::OpenerExt;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
struct Engine {
    url: Option<String>,
    child: Option<CommandChild>,
    error: Option<String>,
    source: Option<String>,
}

type EngineState = Mutex<Engine>;

#[derive(Serialize)]
struct EngineInfo {
    url: Option<String>,
    error: Option<String>,
    source: Option<String>,
}

#[tauri::command]
fn engine_url(state: State<'_, EngineState>) -> EngineInfo {
    let e = state.lock().unwrap();
    EngineInfo { url: e.url.clone(), error: e.error.clone(), source: e.source.clone() }
}

#[tauri::command]
async fn pick_file(app: AppHandle, title: String) -> Option<String> {
    app.dialog()
        .file()
        .set_title(&title)
        .add_filter("CSV / text", &["csv", "tsv", "txt"])
        .blocking_pick_file()
        .map(|p| p.to_string())
}

#[tauri::command]
async fn pick_save_file(app: AppHandle, title: String, default_name: String) -> Option<String> {
    app.dialog()
        .file()
        .set_title(&title)
        .set_file_name(&default_name)
        .add_filter("CSV", &["csv"])
        .blocking_save_file()
        .map(|p| p.to_string())
}

#[tauri::command]
async fn pick_folder(app: AppHandle, title: String) -> Option<String> {
    app.dialog().file().set_title(&title).blocking_pick_folder().map(|p| p.to_string())
}

#[tauri::command]
fn reveal_path(app: AppHandle, path: String) -> Result<(), String> {
    app.opener().reveal_item_in_dir(&path).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_path(app: AppHandle, path: String) -> Result<(), String> {
    app.opener().open_path(&path, None::<&str>).map_err(|e| e.to_string())
}

fn parse_port(line: &str) -> Option<u16> {
    line.trim().strip_prefix("UCC_ENGINE_PORT=")?.trim().parse().ok()
}

/// Spawn the engine and resolve its URL from the `UCC_ENGINE_PORT=<n>` line.
fn start_engine(app: &AppHandle) {
    let state = app.state::<EngineState>();

    if let Ok(url) = std::env::var("UCC_ENGINE_URL") {
        let mut e = state.lock().unwrap();
        e.url = Some(url.trim_end_matches('/').to_string());
        e.source = Some("UCC_ENGINE_URL".into());
        return;
    }

    // `--exit-on-stdin-close`: the plugin gives the child a stdin pipe that we hold for the
    // app's lifetime, so the engine exits on EOF even if we crash or the frozen bootloader
    // was SIGKILLed before it could forward the signal to its Python child.
    const ARGS: [&str; 4] = ["serve", "--port", "0", "--exit-on-stdin-close"];
    let shell = app.shell();
    let spawned = match std::env::var("UCC_ENGINE_CMD") {
        Ok(cmd) => shell.command(cmd).args(ARGS).spawn(),
        Err(_) => match shell.sidecar("ucc-engine") {
            Ok(c) => c.args(ARGS).spawn(),
            Err(err) => Err(err),
        },
    };

    let (mut rx, child) = match spawned {
        Ok(pair) => pair,
        Err(err) => {
            let mut e = state.lock().unwrap();
            e.error = Some(format!(
                "Could not start the mapping engine: {err}. \
                 Set UCC_ENGINE_URL to a running engine or UCC_ENGINE_CMD to the ucc-engine executable."
            ));
            return;
        }
    };

    {
        let mut e = state.lock().unwrap();
        e.child = Some(child);
        e.source = Some(if std::env::var("UCC_ENGINE_CMD").is_ok() { "UCC_ENGINE_CMD".into() } else { "sidecar".into() });
    }

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        let mut stderr_tail: Vec<String> = Vec::new();
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    if let Some(port) = parse_port(&line) {
                        let st = handle.state::<EngineState>();
                        st.lock().unwrap().url = Some(format!("http://127.0.0.1:{port}"));
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).trim_end().to_string();
                    if !line.is_empty() {
                        eprintln!("[ucc-engine] {line}");
                        stderr_tail.push(line);
                        if stderr_tail.len() > 20 {
                            stderr_tail.remove(0);
                        }
                    }
                }
                CommandEvent::Terminated(payload) => {
                    let st = handle.state::<EngineState>();
                    let mut e = st.lock().unwrap();
                    e.child = None;
                    if e.url.is_none() {
                        e.error = Some(format!(
                            "Engine exited before it was ready (code {:?}).\n{}",
                            payload.code,
                            stderr_tail.join("\n")
                        ));
                    } else {
                        e.error = Some("Engine process stopped.".into());
                        e.url = None;
                    }
                    break;
                }
                _ => {}
            }
        }
    });
}

/// Stop the engine *gracefully*: dropping the child handle closes its stdin, and
/// the engine (started with `--exit-on-stdin-close`) exits on EOF. That lets the
/// PyInstaller one-file bootloader remove its `%TEMP%\_MEIxxxx` extraction -
/// killing it outright would leak ~250 MB per session on Windows. Only if the
/// engine is still listening after a grace period do we hard-kill it.
fn stop_engine(app: &AppHandle) {
    let (child, url) = {
        let state = app.state::<EngineState>();
        let mut e = state.lock().unwrap();
        (e.child.take(), e.url.clone())
    };
    let Some(child) = child else { return };
    let pid = child.pid();
    drop(child);

    let Some(port) = url.as_deref().and_then(|u| u.rsplit(':').next()).and_then(|p| p.parse::<u16>().ok()) else {
        return force_kill(pid);
    };
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    for _ in 0..30 {
        if std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_millis(100)).is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    force_kill(pid);
}

fn force_kill(pid: u32) {
    let pid = pid.to_string();
    #[cfg(windows)]
    let status = std::process::Command::new("taskkill").args(["/PID", &pid, "/T", "/F"]).status();
    #[cfg(not(windows))]
    let status = std::process::Command::new("kill").args(["-9", &pid]).status();
    if let Err(err) = status {
        eprintln!("[ucc-engine] could not force-kill pid {pid}: {err}");
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(EngineState::default())
        .invoke_handler(tauri::generate_handler![engine_url, pick_file, pick_save_file, pick_folder, reveal_path, open_path])
        .setup(|app| {
            start_engine(&app.handle().clone());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let RunEvent::Exit | RunEvent::ExitRequested { .. } = event {
                stop_engine(app);
            }
        });
}
