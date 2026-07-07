"""
landing.py — pretty landing page for the FastAPI backend.

WHY THIS EXISTS
───────────────
Out of the box FastAPI gives you:
  • /docs  — Swagger UI (auto-generated, interactive)
  • /redoc — ReDoc (auto-generated, read-only, prettier)
  • /openapi.json — machine-readable spec

These are great for engineers, but they're not curated:
  - they list every endpoint flat,
  - they don't explain *when* to use each route,
  - they don't show meaningful end-to-end examples.

This module renders a hand-curated landing page at `/` that:
  1. Confirms the backend is running ("System is running ✅")
  2. Groups endpoints by user-facing concept (Projects / Clips / Synthesis / Jobs)
  3. For each one: what it does, sample request, expected response shape
  4. Links to /docs and /redoc for live testing

When we add new endpoints, this page is the human-facing changelog.

DESIGN CHOICES
──────────────
• Pure HTML + inline CSS, no JS. The page is essentially a static doc; we
  don't want to ship a build pipeline for it.
• Data-driven: each endpoint is a Python dict; the renderer is generic.
  Adding a new endpoint = adding a dict entry.
• Code samples use real curl / JSON the user can copy-paste.
"""

from __future__ import annotations

from html import escape


# ── Endpoint catalogue ────────────────────────────────────────────
# Each section is a logical grouping. Each endpoint has:
#   method, path, summary, body (None or example), response (str), notes (list)
#
# Updating this list updates the landing page. Keep it in sync as
# the backend grows.

SECTIONS: list[dict] = [
    {
        "title": "Health & system",
        "blurb": "Is the backend alive, and what hardware is it running on?",
        "endpoints": [
            {
                "method": "GET",
                "path": "/health",
                "summary": "Liveness check.",
                "body": None,
                "response": '{"status": "ok"}',
                "notes": ["Use this from the frontend to detect when the sidecar is ready."],
            },
            {
                "method": "GET",
                "path": "/system",
                "summary": "Detect GPU / RAM / disk / OS.",
                "body": None,
                "response": (
                    '{\n'
                    '  "system": {"os": "Linux", "os_version": "...", "python_version": "3.11.13"},\n'
                    '  "gpu":    {"cuda": false, "gpu_name": null, "vram_gb": 0.0, "low_vram_mode": false},\n'
                    '  "ram":    {"total_ram_gb": 15.4, "available_ram_gb": 9.2},\n'
                    '  "disk":   {"free_disk_gb": 47.9}\n'
                    '}'
                ),
                "notes": [
                    "First-run hardware check uses this.",
                    "Decides whether Voice Profile training will be available.",
                ],
            },
        ],
    },
    {
        "title": "Projects",
        "blurb": "Each cloned voice lives inside a project. Multi-voice support is built-in.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/projects",
                "summary": "Create a new project.",
                "body": '{"name": "My voice"}',
                "response": (
                    '{\n'
                    '  "id": "uuid",\n'
                    '  "name": "My voice",\n'
                    '  "created_at": "2026-...",\n'
                    '  "status": "created",\n'
                    '  "clip_count": 0\n'
                    '}'
                ),
                "notes": ["The returned `id` is used in every subsequent endpoint."],
            },
            {
                "method": "GET",
                "path": "/projects/{project_id}",
                "summary": "Fetch project metadata + clip counts.",
                "body": None,
                "response": '{"id": "...", "name": "...", "clip_count": 3, "validated_count": 3, ...}',
                "notes": ["404 if the project doesn't exist."],
            },
        ],
    },
    {
        "title": "Clips (record / upload / manage)",
        "blurb": "Add voice samples to a project. Each clip is validated automatically on upload.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/projects/{project_id}/clips",
                "summary": "Upload a single .wav recording.",
                "body": "(multipart/form-data; field `file` is the .wav)",
                "response": (
                    '{\n'
                    '  "clip_id": "...",\n'
                    '  "duration_s": 9.8,\n'
                    '  "sample_rate": 44100,\n'
                    '  "valid": true,\n'
                    '  "errors": [],\n'
                    '  "warning": []\n'
                    '}'
                ),
                "notes": [
                    "Server-side validation: duration 3–15s, SNR ≥ 20 dB, peak < -1 dBFS, sample-rate ≥ 24 kHz.",
                    "Failures come back as friendly user messages, not technical errors.",
                ],
            },
            {
                "method": "GET",
                "path": "/projects/{project_id}/clips",
                "summary": "List clips for the project.",
                "body": None,
                "response": "[{\"clip_id\": \"...\", \"duration_s\": 9.8, \"path\": \"...\"}]",
                "notes": [],
            },
            {
                "method": "DELETE",
                "path": "/projects/{project_id}/clips/{clip_id}",
                "summary": "Remove a clip (re-record support).",
                "body": None,
                "response": "(204 No Content)",
                "notes": [],
            },
            {
                "method": "POST",
                "path": "/projects/{project_id}/import",
                "summary": "Upload a long audio or video file.",
                "body": "(multipart/form-data; field `file` is .mp4/.mp3/.wav/etc.)",
                "response": '{"job_id": "...", "status": "started", "filename": "podcast.mp4"}',
                "notes": [
                    "Background job. Splits silence-aware, concatenates speech bursts, slices into ~10s clips.",
                    "Poll /jobs/{id} for completion. Result includes `segments_kept`, `clip_ids`.",
                    "Useful for podcasts, interviews, voice memos — converts a long source into a usable clip set.",
                ],
            },
        ],
    },
    {
        "title": "Preprocessing",
        "blurb": "Normalize clips so they're ready for synthesis or training.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/projects/{project_id}/preprocess",
                "summary": "Resample, trim, normalize all clips.",
                "body": None,
                "response": '{"job_id": "...", "status": "started", "clip_count": 4}',
                "notes": [
                    "Background job — poll /jobs/{id}.",
                    "Results saved into `data/projects/{id}/processed/`.",
                ],
            },
        ],
    },
    {
        "title": "Synthesis (Quick Clone)",
        "blurb": "Generate speech in the project's cloned voice.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/projects/{project_id}/synthesize",
                "summary": "Text + reference clip → generated .wav.",
                "body": '{"text": "Hello world", "language": "en", "clean_reference": false}',
                "response": (
                    '{\n'
                    '  "output": "/.../exports/uuid.wav",\n'
                    '  "reference_clip": "/.../processed/uuid.wav",\n'
                    '  "language": "en",\n'
                    '  "cleaned_reference": false\n'
                    '}'
                ),
                "notes": [
                    "Picks the first valid processed clip as the reference automatically.",
                    "Output saved into `data/projects/{id}/exports/{uuid}.wav`.",
                    "`clean_reference` is off by default — XTTS clones better from natural processed audio.",
                ],
            },
            {
                "method": "GET",
                "path": "/projects/{project_id}/preview/{clip_id}",
                "summary": "Stream a generated .wav back to the UI.",
                "body": None,
                "response": "(audio/wav binary stream)",
                "notes": ["`clip_id` is the UUID from `/synthesize` (the part before `.wav`)."],
            },
            {
                "method": "POST",
                "path": "/tts",
                "summary": "Direct synthesis (no project needed).",
                "body": '{"text": "Hello", "speaker_wav": "/path/to/ref.wav", "language": "en"}',
                "response": '{"status": "success", "output": "/.../output.wav"}',
                "notes": ["Lower-level. Useful for testing or non-project flows."],
            },
        ],
    },
    {
        "title": "Voice Profile training (M2)",
        "blurb": "Fine-tune XTTS on a project's clips for higher-fidelity cloning.",
        "endpoints": [
            {
                "method": "GET",
                "path": "/projects/{project_id}/training-plan",
                "summary": "What training would look like on this machine.",
                "body": None,
                "response": (
                    '{\n'
                    '  "can_train": false,\n'
                    '  "summary": "",\n'
                    '  "refusal_reason": "Voice Profile training needs a graphics card...",\n'
                    '  "suggested_action": "Use Quick Clone instead...",\n'
                    '  "detected_hardware": {"cuda": false, "vram_gb": 0.0, ...},\n'
                    '  "plan": null\n'
                    '}'
                ),
                "notes": [
                    "If `can_train` is false, show `refusal_reason` and `suggested_action` to the user.",
                    "If `can_train` is true, show `summary` in the disclosure modal before starting.",
                    "Auto-picks STANDARD or LOW_VRAM preset based on detected hardware.",
                ],
            },
            {
                "method": "POST",
                "path": "/projects/{project_id}/dataset",
                "summary": "Build an XTTS-ready dataset from processed clips.",
                "body": '{"language": "en", "eval_fraction": 0.05}',
                "response": '{"job_id": "...", "status": "started"}',
                "notes": [
                    "Background job. Auto-transcribes via Whisper if no transcripts provided.",
                    "Output: LJSpeech-format `dataset/` folder with metadata.csv + wavs/.",
                    "Result has paths and clip counts.",
                ],
            },
        ],
    },
    {
        "title": "Jobs",
        "blurb": "Anything labeled \"background job\" returns a job_id; poll it here.",
        "endpoints": [
            {
                "method": "GET",
                "path": "/jobs/{job_id}",
                "summary": "Poll a background job.",
                "body": None,
                "response": (
                    '{\n'
                    '  "job_id": "...",\n'
                    '  "type": "preprocess|import|dataset_build",\n'
                    '  "status": "pending|running|completed|failed",\n'
                    '  "progress": 0,\n'
                    '  "message": "Processing clips 1 of 4...",\n'
                    '  "result": {...},\n'
                    '  "error": null,\n'
                    '  "created_at": "..."\n'
                    '}'
                ),
                "notes": [
                    "Poll every ~1 second until `status` is `completed` or `failed`.",
                    "On `completed`, `result` carries the job-specific payload.",
                ],
            },
        ],
    },
]


# ── HTML rendering ────────────────────────────────────────────────

# Single-string template. Inline CSS so it ships in one response.
# We escape user-visible content with `html.escape` to keep this safe even
# if we ever feed dynamic data through.

_PAGE_CSS = """
:root {
    --bg: #fafaf7;
    --fg: #1a1a1a;
    --muted: #6b6b6b;
    --accent: #4a5cf6;
    --card: #ffffff;
    --border: #e5e5e0;
    --code-bg: #f3f3ee;
    --get: #2b9b6c;
    --post: #4a5cf6;
    --delete: #d04545;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--fg);
}
.wrap {
    max-width: 980px;
    margin: 0 auto;
    padding: 56px 24px 96px;
}
h1 {
    margin: 0 0 8px;
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.subtitle {
    color: var(--muted);
    margin: 0 0 32px;
    font-size: 16px;
}
.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #e6f7ed;
    color: #2b6f4d;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 24px;
}
.status::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #2b9b6c;
}
.cta-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 48px;
}
.cta {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: 500;
    font-size: 14px;
    transition: opacity 0.15s;
}
.cta:hover { opacity: 0.85; }
.cta-primary {
    background: var(--accent);
    color: white;
}
.cta-secondary {
    background: var(--card);
    color: var(--fg);
    border: 1px solid var(--border);
}
.section {
    margin-bottom: 40px;
}
.section h2 {
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 4px;
    letter-spacing: -0.01em;
}
.section .blurb {
    color: var(--muted);
    margin: 0 0 16px;
    font-size: 14px;
}
.endpoint {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.endpoint-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 4px;
    flex-wrap: wrap;
}
.method {
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    color: white;
    letter-spacing: 0.04em;
}
.method-GET    { background: var(--get); }
.method-POST   { background: var(--post); }
.method-DELETE { background: var(--delete); }
.path {
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 14px;
    color: var(--fg);
}
.summary {
    margin: 4px 0 12px;
    color: var(--fg);
    font-size: 14px;
}
.kicker {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    margin: 12px 0 4px;
    font-weight: 600;
}
pre {
    margin: 0;
    background: var(--code-bg);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: ui-monospace, SFMono-Regular, monospace;
    font-size: 12.5px;
    line-height: 1.5;
    overflow-x: auto;
    color: #2a2a2a;
    white-space: pre;
}
.notes {
    margin: 8px 0 0;
    padding-left: 18px;
    color: var(--muted);
    font-size: 13px;
}
.notes li { margin-bottom: 2px; }
.foot {
    color: var(--muted);
    font-size: 13px;
    margin-top: 56px;
    padding-top: 24px;
    border-top: 1px solid var(--border);
}
"""


def _render_endpoint(ep: dict) -> str:
    """One card per endpoint."""
    method = escape(ep["method"])
    path = escape(ep["path"])
    summary = escape(ep["summary"])
    body = ep.get("body")
    response = ep.get("response", "")
    notes = ep.get("notes", [])

    parts = [
        '<div class="endpoint">',
        '  <div class="endpoint-head">',
        f'    <span class="method method-{method}">{method}</span>',
        f'    <span class="path">{path}</span>',
        '  </div>',
        f'  <div class="summary">{summary}</div>',
    ]

    if body is not None:
        parts.append('  <div class="kicker">Request body</div>')
        parts.append(f'  <pre>{escape(body)}</pre>')

    parts.append('  <div class="kicker">Response</div>')
    parts.append(f'  <pre>{escape(response)}</pre>')

    if notes:
        parts.append('  <ul class="notes">')
        for n in notes:
            parts.append(f'    <li>{escape(n)}</li>')
        parts.append('  </ul>')

    parts.append('</div>')
    return "\n".join(parts)


def _render_section(s: dict) -> str:
    """A heading + blurb + the endpoint cards under it."""
    parts = [
        '<section class="section">',
        f'  <h2>{escape(s["title"])}</h2>',
        f'  <p class="blurb">{escape(s["blurb"])}</p>',
    ]
    for ep in s["endpoints"]:
        parts.append(_render_endpoint(ep))
    parts.append('</section>')
    return "\n".join(parts)


def render_landing_page() -> str:
    """Build the full HTML page string."""
    sections_html = "\n".join(_render_section(s) for s in SECTIONS)

    # Count for the subtitle
    total_endpoints = sum(len(s["endpoints"]) for s in SECTIONS)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceForge Backend</title>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="wrap">
  <span class="status">System is running</span>
  <h1>🎙️ VoiceForge Backend</h1>
  <p class="subtitle">
    Local-first API for AI voice cloning.
    {total_endpoints} endpoints, grouped below by purpose.
  </p>

  <div class="cta-row">
    <a class="cta cta-primary" href="/docs">Interactive docs (Swagger)</a>
    <a class="cta cta-secondary" href="/redoc">Read-only docs (ReDoc)</a>
    <a class="cta cta-secondary" href="/openapi.json">OpenAPI spec</a>
    <a class="cta cta-secondary" href="/health">Health check</a>
  </div>

  {sections_html}

  <div class="foot">
    Tip: every endpoint is also fully testable from
    <a href="/docs">/docs</a> — click "Try it out" on any of them.
    Frontend (Gradio) runs separately on a free port chosen at startup
    (see <code>/frontend-url</code>) when started with
    <code>python -m frontend.app</code>.
  </div>
</div>
</body>
</html>
"""
