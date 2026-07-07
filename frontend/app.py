"""
app.py — Gradio MVP for VoiceForge.

Screens:
─────────
  1. Welcome / hardware check        — welcome_screen
  2. Project picker / create         — project_screen
  3. Quick Clone (record → synth)    — clone_screen
  4. Voice Profile setup             — profile_screen
     4a. Resource check / disclosure
     4b. Record 30 clips (with prompts, progress)
     4c. Review clips
     4d. Train + live progress
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from frontend import client
from frontend import theme as vf_theme

# ── Language config ──────────────────────────────────────────────
_LANG_LABEL_TO_CODE = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
}
LANGUAGE_CHOICES = list(_LANG_LABEL_TO_CODE.keys())

VOICE_PROFILE_CLIPS = 30     # target clip count for a Voice Profile

# Default prompts (fallback when backend isn't reachable for prompt loading)
_DEFAULT_PROMPTS_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells sea shells by the sea shore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "I scream, you scream, we all scream for ice cream.",
]
_DEFAULT_PROMPTS_HI = [
    "कच्चा पापड़, पक्का पापड़।",
    "चंदा मामा दूर के, पुए पकाएं बूर के।",
    "बाल कृष्ण लीला करें, माखन चुराते हैं।",
    "सात समंदर पार से, गुरु ने भेजा प्यार से।",
    "हवा में उड़ता जाए, मेरा लाल दुपट्टा मलमल का।",
]


# ══════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════

def _hardware_summary() -> tuple[str, str, bool]:
    """Returns (status_md, notice_md, can_train)."""
    try:
        sys = client.get_system_profile()
    except client.BackendError as e:
        return (
            f"### ⚠️ Couldn't reach the backend\n{e}",
            "Start the backend with `uvicorn backend.main:app` and refresh.",
            False,
        )

    gpu = sys.get("gpu", {})
    ram = sys.get("ram", {})
    disk = sys.get("disk", {})

    has_gpu = gpu.get("cuda", False)
    vram = gpu.get("vram_gb", 0.0)
    free_disk = disk.get("free_disk_gb", 0.0)
    ram_gb = ram.get("total_ram_gb", 0.0)
    can_train = has_gpu and vram >= 4.0 and free_disk >= 5.0

    status = (
        "### ✅ Your machine is ready\n"
        "Quick Clone works on any hardware — you're all set."
    )

    if can_train:
        notice = (
            f"**Voice Profile (advanced) available** — "
            f"GPU detected ({gpu.get('gpu_name', 'unknown')}, {vram:.1f} GB VRAM). "
            f"You can train a higher-quality voice later."
        )
    else:
        reasons = []
        if not has_gpu:
            reasons.append("no graphics card detected")
        elif vram < 4.0:
            reasons.append(f"only {vram:.1f} GB VRAM (need 4 GB+)")
        if free_disk < 5.0:
            reasons.append(f"only {free_disk:.1f} GB free disk")
        notice = (
            f"**Note:** Voice Profile training won't be available on this machine "
            f"({', '.join(reasons)}). Quick Clone is the recommended path anyway."
        )

    sysinfo = (
        f"<small>OS: {sys.get('system', {}).get('os', '?')} · "
        f"RAM: {ram_gb:.1f} GB · Free disk: {free_disk:.1f} GB</small>"
    )
    return status, f"{notice}\n\n{sysinfo}", can_train


_ONBOARDING_FLAG_FILE = Path.home() / ".local" / "share" / "voiceforge" / ".onboarding_done"


def _is_first_run() -> bool:
    """Returns True if the user has never dismissed the onboarding screen."""
    return not _ONBOARDING_FLAG_FILE.exists()


def _mark_onboarding_done() -> None:
    _ONBOARDING_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ONBOARDING_FLAG_FILE.touch()


def on_onboarding_done():
    """User dismissed onboarding — mark it done and show welcome screen."""
    _mark_onboarding_done()
    return gr.update(visible=False), gr.update(visible=True)


def on_load_project_list():
    """Fetch all projects and render a markdown list."""
    try:
        projects = client.list_projects()
    except client.BackendError as e:
        return f"⚠️ Couldn't load projects: {e}"
    if not projects:
        return "*No voices yet — create one below.*"
    lines = []
    for p in projects:
        name = p.get("name", "Unnamed")
        clips = p.get("validated_count", 0)
        has_profile = "🧠 profile" if p.get("has_voice_profile") else "🎤 quick clone"
        created = (p.get("created_at", "")[:10])
        lines.append(
            f"**{name}** — {clips} clip{'s' if clips != 1 else ''}, "
            f"{has_profile}, created {created} "
            f"[`{p['id'][:8]}…`]"
        )
    return "\n\n".join(lines)


def on_import_profile(zip_path: str | None):
    """Restore a project from a zip file."""
    if not zip_path:
        return "Drop a .zip file above."
    try:
        project = client.import_profile(zip_path)
        return f"✅ Imported **{project.get('name', 'project')}** (id: `{project['id'][:8]}…`)"
    except client.BackendError as e:
        return f"⚠️ Import failed: {e}"


def on_export_project(project_id: str | None):
    """Trigger export and return file path for gr.File download."""
    if not project_id:
        return "Create or open a project first.", gr.update(visible=False)
    import tempfile
    try:
        dest = client.export_project(project_id, tempfile.gettempdir())
        return (
            f"✅ Ready — click the file below to download.",
            gr.update(visible=True, value=dest),
        )
    except client.BackendError as e:
        return f"⚠️ Export failed: {e}", gr.update(visible=False)


def on_delete_project_confirmed(project_id: str | None):
    """Delete the current project and return to the picker."""
    if not project_id:
        return "No project selected.", None, gr.update(visible=True), gr.update(visible=False)
    try:
        client.delete_project(project_id)
        return (
            "✅ Deleted.",
            None,
            gr.update(visible=True),
            gr.update(visible=False),
        )
    except client.BackendError as e:
        return f"⚠️ Delete failed: {e}", project_id, gr.update(), gr.update()


def _get_prompts(language: str, count: int = VOICE_PROFILE_CLIPS) -> list[str]:
    """Load prompts from the backend transcriber; fall back to defaults."""
    try:
        from backend.audio.transcriber import load_prompts
        prompts = load_prompts(language)
        texts = [p["text"] for p in prompts]
        # Cycle if we need more than available
        if len(texts) < count:
            texts = (texts * ((count // len(texts)) + 1))[:count]
        return texts[:count]
    except Exception:
        defaults = _DEFAULT_PROMPTS_EN if language == "en" else _DEFAULT_PROMPTS_HI
        cycled = (defaults * ((count // len(defaults)) + 1))[:count]
        return cycled


def _poll_job(job_id: str, *, max_seconds: int = 600):
    """
    Generator: yields (message_str, eta_seconds | None, result | None).
    Stops when job is completed / failed / cancelled or timeout.
    """
    last_message = None
    for _ in range(max_seconds):
        time.sleep(1)
        try:
            status = client.get_job(job_id)
        except client.BackendError as e:
            yield f"⚠️ {e}", None, None
            return

        msg = status.get("message") or ""
        eta = status.get("eta_seconds")
        sample = status.get("validation_sample_path")

        if msg != last_message or sample:
            last_message = msg
            yield msg, eta, sample if sample else None

        job_status = status.get("status")
        if job_status in ("completed", "cancelled"):
            yield msg, None, status.get("result")
            return
        if job_status == "failed":
            yield f"⚠️ {status.get('error', 'Something went wrong.')}", None, None
            return


# ══════════════════════════════════════════════════════════════════
# Screen 3 — Quick Clone event handlers
# ══════════════════════════════════════════════════════════════════

def on_create_project(name: str):
    name = (name or "").strip()
    if not name:
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(value="Please give your voice a name."),
            gr.update(),
        )
    try:
        proj = client.create_project(name)
    except client.BackendError as e:
        return (
            gr.update(), gr.update(), gr.update(),
            gr.update(value=f"Couldn't create the project: {e}"),
            gr.update(),
        )
    return (
        proj["id"],
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=""),
        gr.update(value=f"### {proj['name']}"),
    )


def on_recording(audio_path: str | None, project_id: str | None):
    if not audio_path:
        return "Hit record and read the prompt aloud.", False, None
    if not project_id:
        return "Create a project first.", False, None
    try:
        result = client.upload_clip(project_id, audio_path)
    except client.BackendError as e:
        return f"❌ Upload failed: {e}", False, None

    if result["valid"]:
        msg = f"✅ **Looks great!** ({result['duration_s']:.1f}s, {result['sample_rate']} Hz)"
        if result.get("warning"):
            msg += f"\n\n⚠️ {' '.join(result['warning'])}"
        return msg, True, result["clip_id"]

    error_text = " ".join(result.get("errors", ["Try again."]))
    return f"⚠️ **Try again** — {error_text}", False, None


def on_preprocess(project_id: str | None):
    if not project_id:
        yield "Need a project first."
        return
    try:
        job = client.start_preprocess(project_id)
    except client.BackendError as e:
        yield f"⚠️ {e}"
        return
    yield "🎧 Cleaning up your recording..."
    for msg, _eta, _result in _poll_job(job["job_id"], max_seconds=60):
        if _result is not None:
            yield "✨ Ready to use your voice."
            return
        if msg:
            yield f"🎧 {msg}"
    yield "⚠️ Preprocessing is taking longer than expected. Try again."


def on_synthesize(
    text: str,
    project_id: str | None,
    clip_id: str | None,
    language: str | None,
    speed: float = 1.0,
    temperature: float = 0.75,
):
    text = (text or "").strip()
    if not text:
        return None, "Type some text to generate speech."
    if not project_id:
        return None, "Create a project first."
    if not clip_id:
        return None, "Record a voice sample on the Record tab first."
    lang_code = _LANG_LABEL_TO_CODE.get(language or "English", "en")
    try:
        result = client.synthesize(
            project_id, text, language=lang_code,
            speed=speed, temperature=temperature,
        )
    except client.BackendError as e:
        return None, f"⚠️ {e}"
    return result["output"], "🔊 Done — hit play."


def on_import(source_path: str | None, project_id: str | None):
    if not source_path:
        yield "Drop an audio or video file above to import.", None
        return
    if not project_id:
        yield "Create a project first.", None
        return
    yield "📥 Uploading...", None
    try:
        job = client.import_recording(project_id, source_path)
    except client.BackendError as e:
        yield f"⚠️ Upload failed: {e}", None
        return
    yield "🎬 Pulling out the audio...", None
    for msg, _eta, result in _poll_job(job["job_id"], max_seconds=300):
        if result is not None:
            kept = result.get("segments_kept", 0)
            found = result.get("segments_found", 0)
            clip_ids = result.get("clip_ids", [])
            if kept == 0:
                yield "⚠️ We couldn't find usable speech segments in that file.", None
                return
            yield (
                f"✨ **Imported {kept} clip{'s' if kept != 1 else ''}** "
                f"(out of {found} speech regions found). "
                f"You can now generate speech on the Generate tab.",
                clip_ids[0] if clip_ids else None,
            )
            return
        if msg:
            yield f"🎬 {msg}", None
    yield "⚠️ This is taking longer than expected. Try a shorter file.", None


def on_back_to_picker():
    return (
        None,
        gr.update(visible=True),
        gr.update(visible=False),
    )


# ══════════════════════════════════════════════════════════════════
# Screen 4 — Voice Profile event handlers
# ══════════════════════════════════════════════════════════════════

def on_open_voice_profile(project_id: str | None):
    """
    CTA handler: check hardware, fetch training plan, render disclosure.
    Returns (profile_screen visible, disclosure_md, proceed_btn visible,
             refuse_notice, clone_screen visible).
    """
    if not project_id:
        return (
            gr.update(visible=False),
            "Create a project first.",
            gr.update(visible=False),
            "",
            gr.update(visible=True),
        )
    try:
        plan = client.get_training_plan(project_id)
    except client.BackendError as e:
        return (
            gr.update(visible=False),
            f"⚠️ {e}",
            gr.update(visible=False),
            "",
            gr.update(visible=True),
        )

    if not plan.get("can_train"):
        refusal = plan.get("refusal_reason", "This machine can't run training.")
        suggestion = plan.get("suggested_action", "")
        return (
            gr.update(visible=True),
            f"## Voice Profile — Not available on this machine\n\n{refusal}",
            gr.update(visible=False),
            f"\n💡 **{suggestion}**" if suggestion else "",
            gr.update(visible=False),
        )

    summary = plan.get("summary", "")
    data_summary = plan.get("data_summary", "")
    disclosure_md = (
        f"## Set up your Voice Profile\n\n"
        f"Provide a clean voice recording and we'll crop it into up to "
        f"**{VOICE_PROFILE_CLIPS} usable clips** automatically. You can record "
        f"short prompts, upload one long take, or import a video/audio file; "
        f"the app extracts the speech parts that work for training.\n\n"
        f"---\n\n"
        f"**What happens next:**\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"**What gets saved:**\n\n"
        f"{data_summary}"
    )
    return (
        gr.update(visible=True),
        disclosure_md,
        gr.update(visible=True),
        "",
        gr.update(visible=False),
    )


def on_start_recording_session(project_id: str | None, language: str | None):
    """
    User confirmed the disclosure. Load prompts, switch to the recording tab.
    Returns (prompts state, prompt_text, progress_label, rec_panel visible,
             disclosure_panel visible).
    """
    lang_code = _LANG_LABEL_TO_CODE.get(language or "English", "en")
    prompts = _get_prompts(lang_code, VOICE_PROFILE_CLIPS)
    first_prompt = prompts[0] if prompts else "Speak clearly for a few seconds."
    return (
        prompts,                                   # prompts_state
        0,                                         # clip_index_state (0-based)
        f"> *{first_prompt}*",                     # prompt_display
        f"Clip 1 of {VOICE_PROFILE_CLIPS}",        # progress_label
        gr.update(visible=True),                   # rec_panel
        gr.update(visible=False),                  # disclosure_panel
    )


def on_profile_recording(
    audio_path: str | None,
    project_id: str | None,
    prompts: list,
    clip_index: int,
):
    """
    Voice Profile recording handler.

    Two paths depending on audio length:
      3–15 s  → single-clip upload + validate (fast, synchronous)
      other   → import pipeline: silence-boundary extraction plus fallback
                cropping, returning multiple training-sized clips when
                possible. One long recording can fill many clip slots at once.

    Progress jumps by the number of clips extracted, so a 90-second
    recording might jump from clip 3 to clip 12 in one shot.
    """
    if not audio_path:
        prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else "Speak now."
        return f"Hit record and read the prompt.", clip_index, f"> *{prompt}*", f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}", 0
    if not project_id:
        return "Create a project first.", clip_index, "", "", 0

    # Check file duration to decide which path to take.
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        duration_s = info.duration
    except Exception:
        duration_s = 0.0

    if duration_s > 15.0:
        # ── Source recording path: import pipeline ────────────────
        # The backend splits it at silence boundaries into 3–15 s clips.
        # We poll until done, then advance the counter by however many
        # clips were extracted.
        try:
            job = client.import_recording(project_id, audio_path)
        except client.BackendError as e:
            prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
            return f"❌ Upload failed: {e}", clip_index, f"> *{prompt}*", f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}", 0

        # Poll — this is synchronous inside a Gradio event handler,
        # so we block until complete (import of a 90s file takes ~5 s).
        import time
        for _ in range(600):
            time.sleep(1)
            try:
                status = client.get_job(job["job_id"])
            except client.BackendError:
                continue

            if status.get("status") == "completed":
                result = status.get("result") or {}
                kept = result.get("segments_kept", 0)
                found = result.get("segments_found", 0)

                if kept == 0:
                    prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
                    return (
                        "⚠️ No clear speech found in that recording. "
                        "Try speaking a bit louder or closer to the mic.",
                        clip_index,
                        f"> *{prompt}*",
                        f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}",
                        0,
                    )

                new_index = min(clip_index + kept, VOICE_PROFILE_CLIPS)
                if new_index >= VOICE_PROFILE_CLIPS:
                    return (
                        f"✅ Extracted {kept} clip{'s' if kept != 1 else ''} "
                        f"from {duration_s:.0f}s recording.\n\n"
                        f"🎉 All {VOICE_PROFILE_CLIPS} clips recorded! "
                        f"Go to the **Review** tab.",
                        new_index,
                        "",
                        f"Done — {VOICE_PROFILE_CLIPS} of {VOICE_PROFILE_CLIPS}",
                        kept,
                    )

                next_prompt = prompts[new_index] if new_index < len(prompts) else "Speak clearly."
                return (
                    f"✅ Extracted **{kept} clip{'s' if kept != 1 else ''}** "
                    f"from {duration_s:.0f}s recording "
                    f"({found} speech regions found).",
                    new_index,
                    f"> *{next_prompt}*",
                    f"Clip {new_index+1} of {VOICE_PROFILE_CLIPS}",
                    kept,
                )

            if status.get("status") == "failed":
                prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
                return (
                    f"⚠️ Couldn't process that recording: {status.get('error', 'unknown error')}",
                    clip_index,
                    f"> *{prompt}*",
                    f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}",
                    0,
                )

        prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
        return "⚠️ Processing timed out. Try a shorter recording or video file.", clip_index, f"> *{prompt}*", f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}", 0

    # ── Short recording path: single-clip upload ──────────────────
    try:
        result = client.upload_clip(project_id, audio_path)
    except client.BackendError as e:
        prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
        return f"❌ Upload failed: {e}", clip_index, f"> *{prompt}*", f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}", 0

    if not result["valid"]:
        error_text = " ".join(result.get("errors", ["Try again."]))
        prompt = prompts[clip_index] if prompts and clip_index < len(prompts) else ""
        return (
            f"⚠️ **Try again** — {error_text}",
            clip_index,
            f"> *{prompt}*",
            f"Clip {clip_index+1} of {VOICE_PROFILE_CLIPS}",
            0,
        )

    new_index = clip_index + 1
    feedback = f"✅ Clip {clip_index+1} saved ({result['duration_s']:.1f}s)"
    if new_index >= VOICE_PROFILE_CLIPS:
        return (
            f"{feedback}\n\n🎉 All {VOICE_PROFILE_CLIPS} clips recorded! "
            f"Go to the **Review** tab.",
            new_index,
            "",
            f"Done — {VOICE_PROFILE_CLIPS} of {VOICE_PROFILE_CLIPS}",
            1,
        )

    next_prompt = prompts[new_index] if new_index < len(prompts) else "Speak clearly."
    return (
        feedback,
        new_index,
        f"> *{next_prompt}*",
        f"Clip {new_index+1} of {VOICE_PROFILE_CLIPS}",
        1,
    )


def on_load_review(project_id: str | None):
    """Fetch all clips and render a review table."""
    if not project_id:
        return "No project selected."
    try:
        clips = client.list_clips(project_id)
    except client.BackendError as e:
        return f"⚠️ {e}"
    if not clips:
        return "No clips recorded yet. Go back to the Record tab."
    lines = [f"**{len(clips)} clips recorded:**\n"]
    for i, c in enumerate(clips, 1):
        lines.append(
            f"{i}. `{c['clip_id'][:8]}…` — "
            f"{c['duration_s']:.1f}s "
            f"@ {c['sample_rate']} Hz"
        )
    return "\n".join(lines)


def on_delete_last_clip(project_id: str | None, clip_index: int, prompts: list):
    """Delete the most recently recorded clip and step back one prompt."""
    if not project_id or clip_index <= 0:
        return "Nothing to delete.", clip_index, ""
    try:
        clips = client.list_clips(project_id)
        if clips:
            client.delete_clip(project_id, clips[-1]["clip_id"])
    except client.BackendError as e:
        return f"⚠️ Couldn't delete: {e}", clip_index, ""

    new_index = max(0, clip_index - 1)
    prompt = prompts[new_index] if prompts and new_index < len(prompts) else ""
    return (
        f"↩️ Clip {clip_index} deleted — re-record it.",
        new_index,
        f"> *{prompt}*",
    )


def on_start_training(project_id: str | None, language: str | None):
    """
    Build dataset then kick off training. Streams progress messages.
    Yields (status_md, eta_label, preview_audio_path).
    """
    if not project_id:
        yield "No project selected.", "", None
        return

    lang_code = _LANG_LABEL_TO_CODE.get(language or "English", "en")

    # ── Step 1: preprocess all clips ──────────────────────────────
    yield "🎧 Cleaning up your recordings...", "", None
    try:
        preprocess_job = client.start_preprocess(project_id)
    except client.BackendError as e:
        yield f"⚠️ {e}", "", None
        return
    done, last_msg = False, ""
    for msg, _eta, result in _poll_job(preprocess_job["job_id"], max_seconds=300):
        if msg:
            last_msg = msg
        if result is not None:
            done = True
            break
        if msg:
            yield f"🎧 {msg}", "", None
    # Don't advance unless this step actually finished. A timeout or failure
    # here would otherwise cascade into a confusing 422 at the train step.
    if not done:
        yield f"⚠️ Couldn't finish cleaning up your recordings — {last_msg or 'please try again.'}", "", None
        return

    # ── Step 2: build dataset ─────────────────────────────────────
    yield "📋 Preparing your training dataset...", "", None
    try:
        ds_job = client.build_dataset(project_id, language=lang_code)
    except client.BackendError as e:
        yield f"⚠️ {e}", "", None
        return
    # Generous cap: the first build for a language triggers a one-time
    # transcriber download (Whisper "base" ~150 MB for English, "medium"
    # ~1.5 GB for Hindi/Indic), which can take many minutes on a slow link.
    # Giving up early here is what previously fell through to /train → 422.
    done, last_msg = False, ""
    for msg, _eta, result in _poll_job(ds_job["job_id"], max_seconds=1800):
        if msg:
            last_msg = msg
        if result is not None:
            done = True
            break
        if msg:
            yield f"📋 {msg}", "", None
    if not done:
        yield (
            "⚠️ Couldn't prepare the training dataset. "
            f"{last_msg or 'The voice engine may still be downloading (one-time, up to ~2 GB) — wait for it to finish, then try again.'}"
        ), "", None
        return

    # ── Step 3: train ─────────────────────────────────────────────
    yield "🧠 Starting training — this will take a while...", "", None
    try:
        train = client.start_training(project_id, language=lang_code)
    except client.BackendError as e:
        yield f"⚠️ {e}", "", None
        return

    job_id = train["job_id"]
    last_sample = None

    for msg, eta, sample in _poll_job(job_id, max_seconds=6 * 3600):
        # Friendly progress copy that avoids ML jargon
        friendly = _training_copy(msg)
        eta_label = _format_eta(eta) if eta else ""

        if sample and sample != last_sample:
            last_sample = sample

        if sample or (msg and "ready" in msg.lower()):
            yield friendly, eta_label, last_sample
        elif msg:
            yield friendly, eta_label, None

        # Detect completion
        if msg and "ready" in msg.lower():
            yield "🎉 Your Voice Profile is ready! Try it in the Generate tab.", "", last_sample
            return

    yield "⚠️ Training timed out. Check the logs.", "", last_sample


def _training_copy(backend_msg: str) -> str:
    """Map backend progress messages to friendly, jargon-free copy."""
    msg = (backend_msg or "").lower()
    if not msg:
        return "Training in progress..."
    if "getting things ready" in msg or "ready" in msg and "voice" not in msg:
        return "🔧 Getting everything ready..."
    if "listening" in msg:
        return "👂 Listening to your voice..."
    if "learning" in msg:
        return "🧠 Learning your voice..."
    if "almost" in msg:
        return "✨ Almost ready..."
    if "saved progress" in msg or "round" in msg:
        # Extract round info if present e.g. "round 3 of 6"
        return f"💾 {backend_msg}"
    if "stopping" in msg or "stopped" in msg:
        return "⏹️ Stopping and saving..."
    if "profile is ready" in msg:
        return "🎉 Your Voice Profile is ready!"
    return f"⏳ {backend_msg}"


def _format_eta(eta_seconds: int | None) -> str:
    if not eta_seconds or eta_seconds <= 0:
        return ""
    m, s = divmod(int(eta_seconds), 60)
    if m >= 60:
        h, m2 = divmod(m, 60)
        return f"~{h}h {m2}m remaining"
    if m > 0:
        return f"~{m}m {s}s remaining"
    return f"~{s}s remaining"


# ══════════════════════════════════════════════════════════════════
# UI definition
# ══════════════════════════════════════════════════════════════════

def build_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceForge", analytics_enabled=False) as app:

        # Persistent state
        project_id    = gr.State(value=None)
        last_clip_id  = gr.State(value=None)   # Quick Clone active clip
        prompts_state = gr.State(value=[])      # Voice Profile prompt list
        clip_index    = gr.State(value=0)       # Voice Profile recording index

        # ── Header ────────────────────────────────────────────────
        gr.Markdown("# 🎙️ VoiceForge")
        gr.Markdown(
            "Record yourself, type text, hear it in your voice. "
            "Everything runs on your machine."
        )

        # ─────────────────────────────────────────────────────────
        # Screen 0: Onboarding (first-run only)
        # ─────────────────────────────────────────────────────────
        first_run = _is_first_run()
        with gr.Group(visible=first_run) as onboarding_screen:
            gr.Markdown("# 👋 Welcome to VoiceForge")
            gr.Markdown(
                "VoiceForge lets you clone your voice and generate speech in it — "
                "all on your own computer. No cloud, no account, no data sent anywhere.\n\n"
                "**Two ways to use it:**\n\n"
                "🎤 **Quick Clone** (default) — record one 6–10 second clip and "
                "start generating immediately. Works on any machine.\n\n"
                "🧠 **Voice Profile** (advanced) — record ~30 clips to train a "
                "higher-quality model. Needs a GPU with at least 6 GB of memory.\n\n"
                "**How to get started:**\n"
                "1. Give your voice a name\n"
                "2. Record one short clip\n"
                "3. Type any text and hit Generate\n\n"
                "That's it. You can come back and record more clips any time."
            )
            onboarding_done_btn = gr.Button("Let's go →", variant="primary")

        # ─────────────────────────────────────────────────────────
        # Screen 1: Welcome / hardware check
        # ─────────────────────────────────────────────────────────
        with gr.Group(visible=not first_run) as welcome_screen:
            status_md, notice_md, _can_train = _hardware_summary()
            gr.Markdown(status_md)
            gr.Markdown(notice_md)
            welcome_continue = gr.Button("Get started →", variant="primary")

        # ─────────────────────────────────────────────────────────
        # Screen 2: Project picker / create
        # ─────────────────────────────────────────────────────────
        with gr.Group(visible=False) as project_screen:
            gr.Markdown("## Your voices")

            # Existing projects list
            with gr.Accordion("Open an existing voice", open=True):
                project_list_md = gr.Markdown("*Loading...*")
                refresh_btn = gr.Button("↻ Refresh", size="sm")

            gr.Markdown("---")
            gr.Markdown("### Create a new voice")
            gr.Markdown("Give it any label — 'My voice', 'Narrator', whatever.")
            name_in = gr.Textbox(
                label="Voice name", placeholder="e.g. My voice", max_lines=1,
            )
            create_btn = gr.Button("Create", variant="primary", elem_id="vf-create-btn")
            project_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Import a voice profile")
            gr.Markdown(
                "Drop a `.zip` exported from VoiceForge to restore it on this machine."
            )
            import_zip_file = gr.File(
                label="VoiceForge .zip",
                file_count="single",
                file_types=[".zip"],
                type="filepath",
            )
            import_profile_btn = gr.Button("Import", variant="secondary")
            import_profile_status = gr.Markdown("")

        # ─────────────────────────────────────────────────────────
        # Screen 3: Quick Clone
        # ─────────────────────────────────────────────────────────
        with gr.Group(visible=False) as clone_screen:
            clone_header = gr.Markdown("### Voice")

            with gr.Tab("Record"):
                gr.Markdown(
                    "**Read this aloud** (clearly, normal pace, 6–10 seconds):\n\n"
                    "> *The quick brown fox jumps over the lazy dog. "
                    "I'm setting up my voice today and it sounds great so far.*"
                )
                mic = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Recording",
                    elem_id="vf-mic",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                clip_feedback = gr.Markdown("")
                preprocess_status = gr.Markdown("")

            with gr.Tab("Import recording"):
                gr.Markdown(
                    "**Got a longer recording?** Drop a video or audio file and "
                    "we'll split it into clean segments."
                )
                source_file = gr.File(
                    label="Audio or video file",
                    file_count="single",
                    type="filepath",
                )
                import_btn = gr.Button("Import", variant="primary")
                import_status = gr.Markdown("")

            with gr.Tab("Generate"):
                gr.Markdown(
                    "Type any text and hit **Generate**. "
                    "The first generation downloads the voice engine — give it a minute."
                )
                language_in = gr.Radio(
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                    label="Language",
                    info="Hindi uses a more accurate speech recognizer.",
                )
                text_in = gr.Textbox(
                    label="Text",
                    placeholder="Type something for me to say...",
                    lines=3,
                    elem_id="vf-text-in",
                )
                with gr.Accordion("Voice pacing (advanced)", open=False):
                    gr.Markdown(
                        "<small>The voice engine doesn't copy speaking speed "
                        "from your recording — tune it here if the pace feels "
                        "off. Slower + a touch more expression usually sounds "
                        "most natural.</small>"
                    )
                    speed_in = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                        label="Speaking speed",
                        info="Below 1 = slower, above 1 = faster.",
                    )
                    temperature_in = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.75, step=0.05,
                        label="Expression",
                        info="Low = steady and even. High = more natural rhythm "
                             "(slight glitch risk).",
                    )
                gen_btn = gr.Button("Generate", variant="primary", elem_id="vf-generate")
                synth_status = gr.Markdown("")
                audio_out = gr.Audio(label="Output", type="filepath", interactive=False)

            with gr.Tab("Voice Profile ✨"):
                gr.Markdown(
                    "Upload or record voice audio and we'll crop usable clips "
                    "for a higher-quality Voice Profile. Requires a GPU with "
                    "at least 4 GB memory."
                )
                profile_lang_in = gr.Radio(
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                    label="Training language",
                )
                setup_profile_btn = gr.Button("Set up Voice Profile →", variant="secondary")
                profile_cta_status = gr.Markdown("")

            with gr.Tab("Export / Delete"):
                gr.Markdown("### Save this voice profile")
                gr.Markdown(
                    "Downloads a `.zip` with your recordings and trained "
                    "profile (if you've done Voice Profile training). "
                    "You can import it on another machine."
                )
                export_btn = gr.Button("⬇️ Download voice profile", variant="secondary")
                export_status = gr.Markdown("")
                export_file_out = gr.File(label="Downloaded file", visible=False, interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Delete this voice")
                gr.Markdown(
                    "⚠️ This permanently deletes all recordings and the "
                    "trained profile. Cannot be undone."
                )
                delete_project_btn = gr.Button("🗑️ Delete voice", variant="stop")
                delete_status = gr.Markdown("")

            with gr.Row():
                back_btn = gr.Button("← Back to projects")

        # ─────────────────────────────────────────────────────────
        # Screen 4: Voice Profile
        # ─────────────────────────────────────────────────────────
        with gr.Group(visible=False) as profile_screen:
            gr.Markdown("## 🎤 Voice Profile")

            # ── 4a: Disclosure panel ──────────────────────────────
            with gr.Group(visible=True) as disclosure_panel:
                disclosure_md = gr.Markdown("")
                disclosure_refuse = gr.Markdown("")
                profile_lang_confirm = gr.Radio(
                    choices=LANGUAGE_CHOICES,
                    value=LANGUAGE_CHOICES[0],
                    label="Training language",
                    visible=True,
                )
                start_recording_btn = gr.Button(
                    "I understand — start recording →",
                    variant="primary",
                    visible=False,
                )
                back_to_clone_from_disclosure = gr.Button("← Back")

            # ── 4b: Recording panel ───────────────────────────────
            with gr.Group(visible=False) as rec_panel:
                profile_progress_label = gr.Markdown(f"Clip 1 of {VOICE_PROFILE_CLIPS}")
                prompt_display = gr.Markdown("")

                gr.Markdown(
                    "<small>Read the prompt aloud, or upload a longer clean "
                    "recording. We'll automatically crop usable speech clips "
                    "from it for training.</small>"
                )
                profile_mic = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Voice audio",
                    elem_id="vf-profile-mic",
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                profile_clip_feedback = gr.Markdown("")

                with gr.Row():
                    delete_last_btn = gr.Button("↩️ Re-record last clip", variant="stop")
                    go_to_review_btn = gr.Button("Review clips →", variant="secondary")

            # ── 4c: Review panel ──────────────────────────────────
            with gr.Group(visible=False) as review_panel:
                gr.Markdown("### Review your clips")
                review_list = gr.Markdown("")
                with gr.Row():
                    back_to_rec_btn = gr.Button("← Back to recording")
                    start_train_btn = gr.Button("Train voice profile →", variant="primary")

            # ── 4d: Training panel ────────────────────────────────
            with gr.Group(visible=False) as train_panel:
                gr.Markdown("### Training your Voice Profile")
                train_status = gr.Markdown("Starting...")
                train_eta = gr.Markdown("")
                gr.Markdown("**Preview** (updates each round once training begins):")
                train_preview = gr.Audio(
                    label="Voice preview",
                    type="filepath",
                    interactive=False,
                    autoplay=True,
                )
                cancel_train_btn = gr.Button("Stop training", variant="stop")
                train_job_id = gr.State(value=None)

            back_to_clone_from_profile = gr.Button("← Back to Quick Clone", visible=False)

        # ══════════════════════════════════════════════════════════
        # Wiring
        # ══════════════════════════════════════════════════════════

        # Onboarding → Welcome
        onboarding_done_btn.click(
            fn=on_onboarding_done,
            outputs=[onboarding_screen, welcome_screen],
        )

        # Load project list when project screen becomes visible
        welcome_continue.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[welcome_screen, project_screen],
        ).then(
            fn=on_load_project_list,
            outputs=[project_list_md],
        )

        refresh_btn.click(fn=on_load_project_list, outputs=[project_list_md])

        # Import profile
        import_profile_btn.click(
            fn=on_import_profile,
            inputs=[import_zip_file],
            outputs=[import_profile_status],
        ).then(fn=on_load_project_list, outputs=[project_list_md])

        # Screen 2 → 3
        create_btn.click(
            fn=on_create_project,
            inputs=[name_in],
            outputs=[project_id, project_screen, clone_screen,
                     project_status, clone_header],
        )

        # Quick Clone: record → validate → preprocess
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

        # Quick Clone: import
        import_btn.click(
            fn=on_import,
            inputs=[source_file, project_id],
            outputs=[import_status, last_clip_id],
        )

        # Quick Clone: generate
        gen_btn.click(
            fn=on_synthesize,
            inputs=[text_in, project_id, last_clip_id, language_in, speed_in, temperature_in],
            outputs=[audio_out, synth_status],
        )

        # Screen 3 → 4 (Voice Profile CTA)
        setup_profile_btn.click(
            fn=on_open_voice_profile,
            inputs=[project_id],
            outputs=[
                profile_screen,
                disclosure_md,
                start_recording_btn,
                disclosure_refuse,
                clone_screen,
            ],
        )

        # Disclosure → start recording session
        start_recording_btn.click(
            fn=on_start_recording_session,
            inputs=[project_id, profile_lang_confirm],
            outputs=[
                prompts_state,
                clip_index,
                prompt_display,
                profile_progress_label,
                rec_panel,
                disclosure_panel,
            ],
        )

        # Back buttons: disclosure → clone
        back_to_clone_from_disclosure.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[profile_screen, clone_screen],
        )

        # Voice Profile recording
        for trigger in (profile_mic.stop_recording, profile_mic.upload):
            trigger(
                fn=on_profile_recording,
                inputs=[profile_mic, project_id, prompts_state, clip_index],
                outputs=[
                    profile_clip_feedback,
                    clip_index,
                    prompt_display,
                    profile_progress_label,
                    gr.State(),  # returned clip_ids (not used directly in UI)
                ],
            )

        # Re-record last clip
        delete_last_btn.click(
            fn=on_delete_last_clip,
            inputs=[project_id, clip_index, prompts_state],
            outputs=[profile_clip_feedback, clip_index, prompt_display],
        )

        # Recording → Review
        go_to_review_btn.click(
            fn=on_load_review,
            inputs=[project_id],
            outputs=[review_list],
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[rec_panel, review_panel],
        )

        # Review → back to recording
        back_to_rec_btn.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
            outputs=[rec_panel, review_panel],
        )

        # Review → Train
        start_train_btn.click(
            fn=lambda: (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            ),
            outputs=[review_panel, train_panel, back_to_clone_from_profile],
        ).then(
            fn=on_start_training,
            inputs=[project_id, profile_lang_confirm],
            outputs=[train_status, train_eta, train_preview],
        )

        # Back to clone from profile screen (bottom of profile)
        back_to_clone_from_profile.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
            outputs=[profile_screen, clone_screen],
        )

        # Quick Clone back to picker — refresh list on return
        back_btn.click(
            fn=on_back_to_picker,
            outputs=[project_id, project_screen, clone_screen],
        ).then(fn=on_load_project_list, outputs=[project_list_md])

        # Export voice profile
        export_btn.click(
            fn=on_export_project,
            inputs=[project_id],
            outputs=[export_status, export_file_out],
        )

        # Delete voice profile
        delete_project_btn.click(
            fn=on_delete_project_confirmed,
            inputs=[project_id],
            outputs=[delete_status, project_id, project_screen, clone_screen],
        )

    return app


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    import os
    from backend.core.settings import DATA_DIR

    # Port is chosen by the sidecar (a guaranteed-free port) and passed via env.
    # GRADIO_SERVER_PORT is also read natively by Gradio; we read it explicitly
    # so a standalone `python -m frontend.app` still works. 0 = let Gradio pick.
    port = int(os.environ.get("GRADIO_SERVER_PORT", "0")) or None

    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        theme=vf_theme.build_theme(),
        css=vf_theme.CSS,
        head=vf_theme.HEAD,
        allowed_paths=[str(DATA_DIR)],
    )


if __name__ == "__main__":
    main()
