#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Deserialize;
use std::{
    fs,
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::Mutex,
};
use tauri::{Manager, State};
use tauri_plugin_shell::ShellExt;

#[derive(Default)]
struct BackendState {
    child: Mutex<Option<BackendChild>>,
    port_file: Mutex<Option<PathBuf>>,
}

enum BackendChild {
    Std(Child),
    Sidecar(tauri_plugin_shell::process::CommandChild),
}

impl BackendChild {
    fn kill(self) {
        match self {
            BackendChild::Std(mut child) => {
                let _ = child.kill();
                let _ = child.wait();
            }
            BackendChild::Sidecar(child) => {
                let _ = child.kill();
            }
        }
    }
}

#[derive(Deserialize)]
struct PortFile {
    base_url: String,
}

#[tauri::command]
fn backend_url(state: State<'_, BackendState>) -> Result<String, String> {
    let port_file = state
        .port_file
        .lock()
        .map_err(|_| "Backend state is unavailable".to_string())?
        .clone()
        .ok_or_else(|| "Backend has not started yet".to_string())?;

    let raw = fs::read_to_string(&port_file)
        .map_err(|_| format!("Backend is still starting ({})", port_file.display()))?;
    let parsed: PortFile =
        serde_json::from_str(&raw).map_err(|err| format!("Invalid backend port file: {err}"))?;
    Ok(parsed.base_url)
}

fn start_backend(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let state = app.state::<BackendState>();
    let app_data_dir = app.path().app_data_dir()?;
    let runtime_dir = app_data_dir.join("runtime");
    fs::create_dir_all(&runtime_dir)?;
    let port_file = runtime_dir.join("backend-port.json");
    let _ = fs::remove_file(&port_file);

    *state.port_file.lock().expect("backend port file lock poisoned") = Some(port_file.clone());

    let child = if let Ok(path) = std::env::var("VOICEFORGE_SIDECAR_PATH") {
        let child = Command::new(path)
            .env("VOICEFORGE_PORT_FILE", &port_file)
            .env("VOICEFORGE_BACKEND_PORT", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        BackendChild::Std(child)
    } else if cfg!(debug_assertions) {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("Could not resolve repository root")?
            .to_path_buf();
        let venv_python = repo_root.join("venv").join("bin").join("python");
        let python = if venv_python.exists() {
            venv_python
        } else {
            PathBuf::from("python3")
        };
        let child = Command::new(python)
            .arg("-m")
            .arg("backend.sidecar")
            .current_dir(repo_root)
            .env("VOICEFORGE_PORT_FILE", &port_file)
            .env("VOICEFORGE_BACKEND_PORT", "0")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        BackendChild::Std(child)
    } else {
        let (_rx, child) = app
            .shell()
            .sidecar("voiceforge-sidecar")?
            .env("VOICEFORGE_PORT_FILE", port_file)
            .env("VOICEFORGE_BACKEND_PORT", "0")
            .spawn()?;
        BackendChild::Sidecar(child)
    };

    *state.child.lock().expect("backend child lock poisoned") = Some(child);
    Ok(())
}

fn stop_backend(state: &BackendState) {
    if let Ok(mut guard) = state.child.lock() {
        if let Some(child) = guard.take() {
            child.kill();
        }
    }

    if let Ok(port_file) = state.port_file.lock() {
        if let Some(port_file) = port_file.as_ref() {
            let _ = fs::remove_file(port_file);
        }
    }
}

fn main() {
    tauri::Builder::default()
        .manage(BackendState::default())
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![backend_url])
        .setup(|app| {
            start_backend(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::CloseRequested { .. }) {
                let state = window.state::<BackendState>();
                stop_backend(&state);
            }
        })
        .build(tauri::generate_context!())
        .expect("error while running VoiceForge")
        .run(|app_handle, event| {
            if matches!(event, tauri::RunEvent::ExitRequested { .. }) {
                let state = app_handle.state::<BackendState>();
                stop_backend(&state);
            }
        });
}
