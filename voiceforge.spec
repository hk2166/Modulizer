# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the VoiceForge Python sidecar.

Produces a single-folder onedir bundle (not a single .exe — because
torch/TTS are too large to unpack at runtime on every launch).

Run:
    ./scripts/build_sidecar.sh

The output goes to dist/voiceforge-sidecar/voiceforge-sidecar (Linux/Mac)
or dist/voiceforge-sidecar/voiceforge-sidecar.exe (Windows).

Tauri then copies the binary to src-tauri/binaries/voiceforge-sidecar-<TARGET_TRIPLE>.
"""
import sys
from pathlib import Path

ROOT = Path(SPEC).parent   # repository root

# Hidden imports that PyInstaller misses because they're loaded dynamically
HIDDEN_IMPORTS = [
    # FastAPI / Pydantic
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "pydantic.v1",
    "pydantic_core",
    "email_validator",
    # TTS / torch
    "torch",
    "torch.nn",
    "torchaudio",
    "TTS",
    "TTS.api",
    "TTS.tts.models.xtts",
    "TTS.utils.manage",
    "trainer",
    # Audio processing
    "librosa",
    "soundfile",
    "scipy.signal",
    "scipy.io.wavfile",
    "imageio_ffmpeg",
    # Whisper / inference
    "faster_whisper",
    "ctranslate2",
    # VoiceForge backend
    "backend",
    "backend.main",
    "backend.sidecar",
    "backend.api.projects",
    "backend.api.jobs",
    "backend.api.inference",
    "backend.api.landing",
    "backend.audio.recorder",
    "backend.audio.validator",
    "backend.audio.preprocessor",
    "backend.audio.cleaner",
    "backend.audio.importer",
    "backend.audio.transcriber",
    "backend.audio.text_cleaners",
    "backend.pipelines.training",
    "backend.pipelines.training_config",
    "backend.pipelines.dataset_builder",
    "backend.services.project_service",
    "backend.services.inference_service",
    "backend.system.hardware",
    "backend.core.settings",
    "backend.core.logger",
    "backend.jobs.job",
    "backend.jobs.job_manager",
    "backend.jobs.job_status",
    "backend.jobs.instance",
    "backend.runtime",
    # Frontend (Gradio)
    "frontend",
    "frontend.app",
    "frontend.client",
    "gradio",
    "gradio.routes",
    # Text normalisation
    "indic_transliteration",
    "num2words",
    # DeepFilterNet (optional denoiser)
    "df",
    # bitsandbytes (optional 8-bit Adam)
    "bitsandbytes",
]

# Data files (non-Python assets that must ship with the bundle)
DATAS = [
    # Prompt scripts used by transcriber / recording UI
    (str(ROOT / "data" / "scripts"), "data/scripts"),
]

a = Analysis(
    [str(ROOT / "backend" / "sidecar.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude dev-only and unneeded heavy packages
    excludes=[
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "setuptools",
        "pkg_resources",
        "tensorboard",
        "tensorboardX",
        "tkinter",
        "wx",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "cv2",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir mode (no single-file unpack overhead)
    name="voiceforge-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="voiceforge-sidecar",
)
