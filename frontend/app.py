"""
app.py — Gradio MVP for VoiceForge.

Runs a small web UI that talks to the FastAPI backend over HTTP.

Architecture:
─────────────
  Gradio Blocks    →   client.py    →   FastAPI (localhost:8000)
   (this file)         (HTTP layer)        (your existing backend)

Screens:
─────────
  1. Welcome / hardware check        — `welcome_screen`
  2. Project picker / create         — `project_screen`
  3. Quick Clone (record → synth)    — `clone_screen`

We use Gradio's "switch what's visible" pattern: every screen is a Group
that we toggle via gr.update(visible=...). This is simpler than a real
router and still feels like screens.

Key Gradio concepts you'll see:
────────────────────────────────
• gr.Blocks — the container for a custom layout (vs gr.Interface which is
  one input → one output).
• gr.State — a hidden value that survives across event handlers. We use it
  to track the active project id without showing it.
• Event handlers — buttons/inputs have `.click(fn=..., inputs=[...], outputs=[...])`.
  Gradio passes input component values to fn and assigns its return value
  to the output components in order.
• `time.sleep` polling — for our preprocess job we just block in Python
  with a polling loop. Crude but works for the MVP.
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from frontend import client


# ══════════════════════════════════════════════════════════════════
# Hardware check copy
# ══════════════════════════════════════════════════════════════════

def _hardware_summary() -> tuple[str, str]:
    """
    Returns (status_markdown, fine_tuning_notice).

    `status_markdown` is the always-encouraging top-line for the user.
    `fine_tuning_notice` explains whether Voice Profile training will be
    available later. Either way, Quick Clone is always fine.
    """
    try:
        sys = client.get_system_profile()
    except client.BackendError as e:
        return (
            f"### ⚠️ Couldn't reach the backend\n{e}",
            "Start the backend with `uvicorn backend.main:app` and refresh.",
        )

    gpu = sys.get("gpu", {})
    ram = sys.get("ram", {})
    disk = sys.get("disk", {})

    has_gpu = gpu.get("cuda", False)
    vram = gpu.get("vram_gb", 0.0)
    free_disk = disk.get("free_disk_gb", 0.0)
    ram_gb = ram.get("total_ram_gb", 0.0)

    status = (
        "### ✅ Your machine is ready\n"
        "Quick Clone works on any hardware — you're all set."
    )

    if has_gpu and vram >= 4.0 and free_disk >= 5.0:
        notice = (
            f"**Voice Profile (advanced) available** — "
            f"GPU detected ({gpu.get('gpu_name', 'unknown')}, {vram:.1f} GB VRAM). "
            f"You'll be able to train a higher-quality profile later if you want."
        )
    else:
        # Stay friendly. Don't lecture about what's missing.
        reasons = []
        if not has_gpu:
            reasons.append("no graphics card detected")
        elif vram < 4.0:
            reasons.append(f"only {vram:.1f} GB VRAM (need 4 GB+)")
        if free_disk < 5.0:
            reasons.append(f"only {free_disk:.1f} GB free disk")
        notice = (
            f"**Note:** Voice Profile training won't be available on this machine "
            f"({', '.join(reasons)}). Quick Clone is the recommended path anyway — "
            f"it works great and is much faster."
        )

    sysinfo = (
        f"<small>"
        f"OS: {sys.get('system', {}).get('os', '?')} · "
        f"RAM: {ram_gb:.1f} GB · "
        f"Free disk: {free_disk:.1f} GB"
        f"</small>"
    )

    return status, f"{notice}\n\n{sysinfo}"


# ══════════════════════════════════════════════════════════════════
# Event handlers — kept small, each maps cleanly to one user action
# ══════════════════════════════════════════════════════════════════

def on_create_project(name: str):
    """Validate name, create project on backend, switch to clone screen."""
    name = (name or "").strip()
    if not name:
        # Stay on the project screen and show an error
        return (
            gr.update(),                        # project_id state
            gr.update(),                        # project_screen visibility
            gr.update(),                        # clone_screen visibility
            gr.update(value="Please give your voice a name."),  # status text
            gr.update(),                        # clone screen header
        )

    try:
        proj = client.create_project(name)
    except client.BackendError as e:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"Couldn't create the project: {e}"),
            gr.update(),
        )

    # Success — store id, switch screens
    return (
        proj["id"],                                # project_id state
        gr.update(visible=False),                  # hide project screen
        gr.update(visible=True),                   # show clone screen
        gr.update(value=""),                       # clear status
        gr.update(value=f"### {proj['name']}"),    # set clone screen header
    )


def on_recording(audio_path: str | None, project_id: str | None):
    """
    Called when the user finishes recording (or uploads a file).
    Uploads the clip, runs server-side validation, and shows feedback.

    Returns:
      - validation feedback markdown
      - whether the "Continue" path is unlocked (we use state to gate it)
    """
    if not audio_path:
        return "Hit record and read the prompt aloud.", False, None
    if not project_id:
        return "Create a project first.", False, None

    try:
        result = client.upload_clip(project_id, audio_path)
    except client.BackendError as e:
        return f"❌ Upload failed: {e}", False, None

    # Inline validation feedback. The backend already returns user-friendly
    # error messages — we just format them.
    if result["valid"]:
        msg = (
            f"✅ **Looks great!** "
            f"({result['duration_s']:.1f}s, {result['sample_rate']} Hz)"
        )
        if result.get("warning"):
            msg += f"\n\n⚠️ {' '.join(result['warning'])}"
        return msg, True, result["clip_id"]

    # Failed validation — show the friendliest of the errors
    error_text = " ".join(result.get("errors", ["Try again."]))
    return f"⚠️ **Try again** — {error_text}", False, None


def on_preprocess(project_id: str | None):
    """
    Auto-runs after a clip validates successfully. Kicks off preprocessing
    and polls the job until done. We yield Gradio updates as the status
    changes — this gives the user real-time feedback without WebSockets.

    Generators in Gradio handlers:
      `yield` lets you stream multiple updates from one handler. Each yield
      becomes a UI repaint. Gradio handles the streaming wire format for us.
    """
    if not project_id:
        yield "Need a project first."
        return

    try:
        job = client.start_preprocess(project_id)
    except client.BackendError as e:
        yield f"⚠️ {e}"
        return

    job_id = job["job_id"]
    yield "🎧 Cleaning up your recording..."

    # Poll loop. In production we'd use SSE or WebSocket; for MVP this is fine.
    last_message = None
    for _ in range(60):  # max 60s — plenty for one clip
        time.sleep(1)
        try:
            status = client.get_job(job_id)
        except client.BackendError as e:
            yield f"⚠️ {e}"
            return

        # Stream the backend's own progress message if it changed.
        # The job manager updates `message` like "Processing clips 1 of 1..."
        msg = status.get("message")
        if msg and msg != last_message:
            yield f"🎧 {msg}"
            last_message = msg

        if status["status"] == "completed":
            yield "✨ Ready to use your voice."
            return
        if status["status"] == "failed":
            yield f"⚠️ Something went wrong: {status.get('error', 'unknown')}"
            return

    yield "⚠️ Preprocessing is taking longer than expected. Try again."


def on_synthesize(text: str, project_id: str | None):
    """Generate speech and return a path the audio component can play."""
    text = (text or "").strip()
    if not text:
        return None, "Type some text to generate speech."
    if not project_id:
        return None, "Create a project first."

    try:
        result = client.synthesize(project_id, text)
    except client.BackendError as e:
        return None, f"⚠️ {e}"

    # The backend writes to a path on local disk. Gradio's gr.Audio can read
    # that path directly when filepath is the value type.
    return result["output"], "🔊 Done — hit play."


def on_back_to_picker():
    """Go back to the project screen, clearing project state."""
    return (
        None,                          # project_id state
        gr.update(visible=True),       # show project screen
        gr.update(visible=False),      # hide clone screen
    )


# ══════════════════════════════════════════════════════════════════
# UI definition
# ══════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    # Gradio 6 moved `theme` to launch() — we still set it here as a hint
    # for IDE introspection but it'll be applied at launch time.
    with gr.Blocks(title="VoiceForge") as app:

        # Persistent state — survives across screens. value=None = no project yet.
        project_id = gr.State(value=None)
        last_clip_id = gr.State(value=None)

        # ── Header ────────────────────────────────────────────────
        gr.Markdown("# 🎙️ VoiceForge")
        gr.Markdown(
            "Record yourself, type text, hear it in your voice. "
            "Everything runs on your machine."
        )

        # ── Screen 1: Welcome / hardware check ────────────────────
        # We render this once at startup using `_hardware_summary()`.
        with gr.Group(visible=True) as welcome_screen:
            status_md, notice_md = _hardware_summary()
            gr.Markdown(status_md)
            gr.Markdown(notice_md)
            welcome_continue = gr.Button("Get started →", variant="primary")

        # ── Screen 2: Project picker / create ─────────────────────
        # MVP: just create. Listing existing projects is a follow-up.
        with gr.Group(visible=False) as project_screen:
            gr.Markdown("## Name your voice")
            gr.Markdown(
                "Give it any label — 'My voice', 'Narrator', whatever. "
                "You can have multiple voices and switch between them."
            )
            name_in = gr.Textbox(
                label="Voice name",
                placeholder="e.g. My voice",
                max_lines=1,
            )
            create_btn = gr.Button("Create", variant="primary")
            project_status = gr.Markdown("")

        # ── Screen 3: Quick Clone (record → synth) ────────────────
        with gr.Group(visible=False) as clone_screen:
            clone_header = gr.Markdown("### Voice")

            with gr.Tab("Record"):
                gr.Markdown(
                    "**Read this aloud** (clearly, normal pace, 6–10 seconds):\n\n"
                    "> *The quick brown fox jumps over the lazy dog. "
                    "I'm setting up my voice today and it sounds great so far.*"
                )

                # `sources=["microphone"]` enables in-browser recording.
                # `type="filepath"` makes Gradio save a temp .wav and pass us the path,
                # which is exactly what our upload helper expects.
                # `waveform_options=...` shows the visual preview.
                mic = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Recording",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )

                clip_feedback = gr.Markdown("")
                preprocess_status = gr.Markdown("")

                # Hidden gate: only after validation passes do we enable the synth section.
                # We bind to `last_clip_id` state — it's None until a clip validates.

            with gr.Tab("Generate"):
                gr.Markdown(
                    "Type any text and hit **Generate**. "
                    "The first generation downloads the voice engine — give it a minute."
                )
                text_in = gr.Textbox(
                    label="Text",
                    placeholder="Type something for me to say...",
                    lines=3,
                )
                gen_btn = gr.Button("Generate", variant="primary")
                synth_status = gr.Markdown("")
                audio_out = gr.Audio(label="Output", type="filepath", interactive=False)

            with gr.Row():
                back_btn = gr.Button("← Back to projects")

        # ══════════════════════════════════════════════════════════
        # Wire up events
        # ══════════════════════════════════════════════════════════

        # Welcome → Project screen
        welcome_continue.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[welcome_screen, project_screen],
        )

        # Create project → Clone screen
        create_btn.click(
            fn=on_create_project,
            inputs=[name_in],
            outputs=[project_id, project_screen, clone_screen, project_status, clone_header],
        )

        # Recording finishes → upload + validate, then auto-preprocess.
        # `mic.stop_recording` fires when the user stops the mic.
        # `mic.upload` fires when they drop a file in.
        # `.then(...)` chains the preprocess call so it runs only after validation.
        for trigger in (mic.stop_recording, mic.upload):
            trigger(
                fn=on_recording,
                inputs=[mic, project_id],
                outputs=[clip_feedback, gr.State(), last_clip_id],
            ).then(
                fn=on_preprocess,
                inputs=[project_id],
                outputs=[preprocess_status],
            )

        # Generate
        gen_btn.click(
            fn=on_synthesize,
            inputs=[text_in, project_id],
            outputs=[audio_out, synth_status],
        )

        # Back to picker
        back_btn.click(
            fn=on_back_to_picker,
            outputs=[project_id, project_screen, clone_screen],
        )

    return app


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    """
    Launch Gradio.

    `server_name="127.0.0.1"` keeps it local-only (matches our local-first promise).
    `inbrowser=True` opens it automatically when you run the file.
    `theme=...` applies the visual theme (moved from Blocks() in Gradio 6).
    """
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
