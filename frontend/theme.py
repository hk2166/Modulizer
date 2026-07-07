"""
theme.py — Neon / synth-studio visual theme for the VoiceForge Gradio app.

This module is styling only. It exposes:

    build_theme() -> gr.Theme      # color/spacing tokens for gr.Blocks
    CSS          -> str            # custom CSS injected into gr.Blocks

Design language
───────────────
Dark near-black base with neon accents, tuned to feel like a hardware
synth / DAW dark mode rather than a generic neon-gradient SaaS page:

    cyan     #00e5ff   primary / active (Generate, Record idle)
    magenta  #ff2fd0   recording-in-progress
    violet   #8b5cf6   Voice Profile / advanced path
    green    #39ff88   success / validation passed
    amber    #ff9f1c   friendly warnings / hardware-gate refusals

Glows are state-tied (recording, generating, training), not decorative on
every element, so the waveform visualizations stay the focal point.

Fonts are self-hosted and base64-embedded into the CSS so they load with no
external CDN calls — Gradio serves its own page on port 7860, so it can't
reach the Tauri-served font files; embedding keeps everything offline.
"""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

# ── Palette ─────────────────────────────────────────────────────────
BG = "#0a0a0f"
SURFACE = "#12121a"
SURFACE_2 = "#1a1a26"
BORDER = "#262636"

CYAN = "#00e5ff"
MAGENTA = "#ff2fd0"
VIOLET = "#8b5cf6"
GREEN = "#39ff88"
AMBER = "#ff9f1c"

TEXT = "#e8e8f0"
TEXT_MUTED = "#8888a0"

# Fonts live alongside the Tauri shell assets.
_FONT_DIR = Path(__file__).resolve().parent.parent / "src" / "assets" / "fonts"


def _font_face(family: str, filename: str, weight: str = "400") -> str:
    """Return an @font-face rule with the woff2 base64-embedded (offline-safe).

    Falls back to an empty string if the font file is missing so the app still
    renders (system fonts) rather than crashing.
    """
    path = _FONT_DIR / filename
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return (
        "@font-face{"
        f"font-family:'{family}';"
        f"src:url(data:font/woff2;base64,{b64}) format('woff2');"
        f"font-weight:{weight};font-display:swap;font-style:normal;"
        "}"
    )


_FONT_FACES = "".join(
    [
        _font_face("Space Grotesk", "space-grotesk.woff2", "300 700"),
        _font_face("Chakra Petch", "chakra-petch.woff2", "500"),
        _font_face("Inter", "inter.woff2", "100 900"),
    ]
)


def build_theme() -> gr.Theme:
    """A dark neon theme built on gr.themes.Base."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.slate,
        # Plain family names (fonts are self-hosted via base64 @font-face in CSS
        # below) — avoids gr.themes.GoogleFont emitting a Google Fonts CDN call,
        # which would break the offline-desktop guarantee.
        font=("Inter", "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=("Space Grotesk", "ui-monospace", "monospace"),
    ).set(
        # Surfaces
        body_background_fill=BG,
        body_background_fill_dark=BG,
        background_fill_primary=SURFACE,
        background_fill_secondary=SURFACE_2,
        block_background_fill=SURFACE,
        block_border_color=BORDER,
        block_border_width="1px",
        block_label_background_fill=SURFACE_2,
        block_label_text_color=TEXT_MUTED,
        panel_background_fill=SURFACE,
        border_color_primary=BORDER,
        # Text
        body_text_color=TEXT,
        body_text_color_subdued=TEXT_MUTED,
        block_title_text_color=TEXT,
        # Inputs
        input_background_fill=SURFACE_2,
        input_border_color=BORDER,
        input_border_color_focus=CYAN,
        input_placeholder_color=TEXT_MUTED,
        # Buttons — primary = cyan
        button_primary_background_fill=CYAN,
        button_primary_background_fill_hover=CYAN,
        button_primary_text_color=BG,
        button_primary_border_color=CYAN,
        # secondary = violet outline
        button_secondary_background_fill=SURFACE_2,
        button_secondary_text_color=VIOLET,
        button_secondary_border_color=VIOLET,
        # Radius / spacing kept tight and technical
        block_radius="14px",
        button_large_radius="12px",
        button_small_radius="10px",
        input_radius="10px",
    )
    return theme


# ── Custom CSS ──────────────────────────────────────────────────────
# Selectors target Gradio's generated structure plus a few elem_id hooks
# added in app.py (vf-mic, vf-profile-mic, vf-generate, vf-record-panel...).
# No logic or wiring depends on any of this — styling only.

# ── Head injection ─────────────────────────────────────────────────
# Gradio's HTML template hardcodes <link rel="preconnect"> hints to
# fonts.googleapis.com / fonts.gstatic.com. We self-host every font (base64
# below), so those hints are dead weight — and this is an offline desktop
# app, so we strip them at load to guarantee zero external font-CDN calls.
HEAD = """
<script>
(function () {
  function strip() {
    document
      .querySelectorAll('link[href*="googleapis"], link[href*="gstatic"]')
      .forEach(function (el) { el.parentNode && el.parentNode.removeChild(el); });
  }
  strip();
  document.addEventListener("DOMContentLoaded", strip);
})();
</script>
"""


CSS = (
    _FONT_FACES
    + """
:root {
  --vf-bg: #0a0a0f;
  --vf-surface: #12121a;
  --vf-surface-2: #1a1a26;
  --vf-border: #262636;
  --vf-cyan: #00e5ff;
  --vf-magenta: #ff2fd0;
  --vf-violet: #8b5cf6;
  --vf-green: #39ff88;
  --vf-amber: #ff9f1c;
  --vf-text: #e8e8f0;
  --vf-text-muted: #8888a0;
  --vf-glow-cyan: 0 0 12px rgba(0,229,255,.5), 0 0 30px rgba(0,229,255,.2);
  --vf-glow-magenta: 0 0 14px rgba(255,47,208,.6), 0 0 36px rgba(255,47,208,.28);
  --vf-glow-violet: 0 0 12px rgba(139,92,246,.5), 0 0 28px rgba(139,92,246,.2);
  --vf-glow-green: 0 0 12px rgba(57,255,136,.5), 0 0 28px rgba(57,255,136,.2);
  --vf-font-display: "Space Grotesk","Chakra Petch",ui-sans-serif,system-ui,sans-serif;
  --vf-font-body: "Inter",ui-sans-serif,system-ui,sans-serif;
}

/* ── Base canvas + subtle grid texture ────────────────────────── */
.gradio-container {
  background: var(--vf-bg) !important;
  color: var(--vf-text) !important;
  font-family: var(--vf-font-body) !important;
  background-image:
    linear-gradient(rgba(139,92,246,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,.035) 1px, transparent 1px) !important;
  background-size: 46px 46px !important;
}

/* Headings in the geometric display face */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3 {
  font-family: var(--vf-font-display) !important;
  letter-spacing: .01em;
  color: var(--vf-text) !important;
}
.gradio-container h1 {
  background: linear-gradient(90deg, var(--vf-cyan), var(--vf-violet));
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 26px rgba(0,229,255,.18);
}

/* Cards / blocks */
.block, .form, .gr-box, .gr-panel {
  background: var(--vf-surface) !important;
  border-color: var(--vf-border) !important;
}

/* ── Primary buttons — steady cyan glow at idle ───────────────── */
button.primary, .primary button, button[variant="primary"] {
  background: var(--vf-cyan) !important;
  color: var(--vf-bg) !important;
  border: 1px solid var(--vf-cyan) !important;
  font-family: var(--vf-font-display) !important;
  font-weight: 600 !important;
  letter-spacing: .02em;
  box-shadow: var(--vf-glow-cyan);
  transition: box-shadow .25s ease, transform .1s ease;
}
button.primary:hover, .primary button:hover {
  box-shadow: 0 0 18px rgba(0,229,255,.75), 0 0 44px rgba(0,229,255,.3) !important;
}
button.primary:active, .primary button:active { transform: translateY(1px); }

/* Secondary — violet outline (advanced path) */
button.secondary, .secondary button {
  background: var(--vf-surface-2) !important;
  color: var(--vf-violet) !important;
  border: 1px solid var(--vf-violet) !important;
  font-family: var(--vf-font-display) !important;
  box-shadow: 0 0 0 rgba(139,92,246,0);
  transition: box-shadow .25s ease;
}
button.secondary:hover, .secondary button:hover {
  box-shadow: var(--vf-glow-violet) !important;
}

/* Stop / destructive — magenta warning glow */
button.stop, .stop button {
  background: var(--vf-surface-2) !important;
  color: var(--vf-magenta) !important;
  border: 1px solid var(--vf-magenta) !important;
  font-family: var(--vf-font-display) !important;
}
button.stop:hover, .stop button:hover {
  box-shadow: var(--vf-glow-magenta) !important;
}

/* ── Text inputs styled like a console line ───────────────────── */
input[type="text"], textarea, .gr-text-input, input[type="number"] {
  background: var(--vf-surface-2) !important;
  color: var(--vf-text) !important;
  border: 1px solid var(--vf-border) !important;
  font-family: var(--vf-font-body) !important;
}
input[type="text"]:focus, textarea:focus {
  border-color: var(--vf-cyan) !important;
  box-shadow: 0 0 0 1px var(--vf-cyan), var(--vf-glow-cyan) !important;
  outline: none !important;
  caret-color: var(--vf-cyan);
}
/* The Generate textbox reads like a terminal line */
#vf-text-in textarea {
  font-family: var(--vf-font-display) !important;
  caret-color: var(--vf-cyan);
}

/* ── Tabs ─────────────────────────────────────────────────────── */
.tab-nav button, button.tab {
  color: var(--vf-text-muted) !important;
  font-family: var(--vf-font-display) !important;
  letter-spacing: .03em;
  border: none !important;
  background: transparent !important;
}
.tab-nav button.selected, button.tab.selected {
  color: var(--vf-cyan) !important;
  border-bottom: 2px solid var(--vf-cyan) !important;
  text-shadow: 0 0 10px rgba(0,229,255,.5);
}

/* ── Sliders (pacing / expression) ────────────────────────────── */
input[type="range"] { accent-color: var(--vf-cyan); }

/* ── Progress / loading — cyan pulse while a job runs ─────────── */
.progress-bar, .meta-text, .eta-bar {
  color: var(--vf-cyan) !important;
}
.gradio-container .progress-bar {
  background: linear-gradient(90deg, var(--vf-cyan), var(--vf-violet)) !important;
  box-shadow: var(--vf-glow-cyan);
}

/* ── Audio components — neon waveform surface ─────────────────── */
.gradio-container .audio-container,
.gradio-container [data-testid="waveform"] {
  background: var(--vf-surface-2) !important;
  border: 1px solid var(--vf-border) !important;
  border-radius: 12px;
}

/* The Quick Clone + Voice Profile mic get a cyan-lit frame.
   While recording, Gradio adds a recording state we tint magenta. */
#vf-mic, #vf-profile-mic {
  border: 1px solid rgba(0,229,255,.35) !important;
  border-radius: 14px !important;
  box-shadow: inset 0 0 20px rgba(0,229,255,.06);
  transition: box-shadow .3s ease, border-color .3s ease;
}
#vf-mic:has(button[aria-label*="Stop" i]),
#vf-profile-mic:has(button[aria-label*="Stop" i]) {
  border-color: var(--vf-magenta) !important;
  box-shadow: var(--vf-glow-magenta), inset 0 0 22px rgba(255,47,208,.1) !important;
}

/* The record button itself: cyan idle, magenta glow while capturing */
#vf-mic button[aria-label*="Record" i],
#vf-profile-mic button[aria-label*="Record" i] {
  color: var(--vf-cyan) !important;
}
#vf-mic button[aria-label*="Stop" i],
#vf-profile-mic button[aria-label*="Stop" i] {
  color: var(--vf-magenta) !important;
  filter: drop-shadow(0 0 6px rgba(255,47,208,.7));
  animation: vf-rec-pulse 1.4s ease-in-out infinite;
}

/* ── Validation feedback banners (friendly copy, glowing edge) ── */
/* app.py renders these as Markdown starting with ✅ / ⚠️ — we can't
   class each line, so the glow lives on the mic frame + button state.
   Markdown emphasis still reads clearly on the dark surface. */
.gradio-container .prose strong { color: var(--vf-text); }
.gradio-container .prose code {
  background: var(--vf-surface-2) !important;
  color: var(--vf-cyan) !important;
  border: 1px solid var(--vf-border);
  border-radius: 6px;
  padding: 1px 6px;
  font-family: var(--vf-font-display) !important;
}

/* Links / accents */
.gradio-container a { color: var(--vf-cyan) !important; }

/* Scrollbar to match the dark studio look */
.gradio-container ::-webkit-scrollbar { width: 10px; height: 10px; }
.gradio-container ::-webkit-scrollbar-track { background: var(--vf-bg); }
.gradio-container ::-webkit-scrollbar-thumb {
  background: var(--vf-surface-2);
  border-radius: 8px;
  border: 2px solid var(--vf-bg);
}
.gradio-container ::-webkit-scrollbar-thumb:hover { background: var(--vf-violet); }

/* ── Keyframes (state-tied, no decorative strobing) ───────────── */
@keyframes vf-rec-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: .45; }
}
@keyframes vf-cta-pulse {
  0%, 100% { box-shadow: var(--vf-glow-cyan); }
  50% { box-shadow: 0 0 20px rgba(0,229,255,.8), 0 0 50px rgba(0,229,255,.35); }
}
/* The New Project / Create CTA breathes softly (not aggressive strobe) */
#vf-create-btn button, #vf-create-btn.primary {
  animation: vf-cta-pulse 2.6s ease-in-out infinite;
}
"""
)
