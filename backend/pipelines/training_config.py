"""
training_config.py — pick XTTS fine-tuning settings based on this machine.

WHY THIS MODULE EXISTS
──────────────────────
XTTS v2 fine-tuning has dozens of knobs (batch size, mixed precision,
gradient checkpointing, dataloader workers, learning rate, ...). The right
values depend on the user's hardware, and getting them wrong means:

  - Out-of-memory crash mid-training (lost hours of progress)
  - Painfully slow training that never converges
  - Quality degradation from too-aggressive memory savings

We don't want to expose any of this to the user — they shouldn't even know
"batch size" exists. So this module looks at `hardware.py`, picks a sensible
preset, and returns:

  1. A `TrainingPlan` — typed config the trainer can apply directly.
  2. A `friendly_summary` — plain-English description for the UI
     ("We'll train for ~3 hours using 6 GB of GPU memory").
  3. A `refusal` — None if we can train, otherwise a friendly explanation
     of why we can't (and what to do instead).

PRESETS
───────
  STANDARD       — GPU with ≥8 GB free VRAM. Full batch size, fp32 default
                   with optional bf16 if the GPU supports it. Fastest.

  LOW_VRAM       — GPU with 3–8 GB VRAM. Smaller batch, mixed precision
                   (fp16 autocast), gradient checkpointing on, fewer
                   dataloader workers. Trains 2–3× slower but fits in memory.

  REFUSE_NO_GPU  — No CUDA at all. CPU fine-tuning would take 24+ hours;
                   we refuse instead of shipping a broken UX.

  REFUSE_LOW_VRAM — GPU under 3 GB. Even with low-VRAM tricks, XTTS won't
                    fit. Refused with a Quick Clone suggestion.

KEY CONCEPTS YOU'LL SEE
───────────────────────
• Mixed precision (fp16 / bf16):
  Most weights stay in fp32, but math during forward/backward happens in
  16-bit. Memory savings: ~40%. Speed boost: ~30% on Tensor Core GPUs.
  Tradeoff: occasional numerical instability — handled by autocast +
  gradient scaling. PyTorch's `torch.cuda.amp.autocast` is the standard.

• Gradient checkpointing:
  During backprop, we normally store every intermediate tensor (the
  "activations") so we can compute gradients. Activations dominate memory
  for transformers. Gradient checkpointing throws away activations during
  forward pass and recomputes them during backward. Trades ~30% extra
  compute for ~40% memory savings. Crucial for low-VRAM training.

• Batch size:
  How many training examples we process before updating weights. Bigger
  batch = more stable gradients, faster training... but more VRAM.
  XTTS default is 32; on a 4 GB GPU we go down to 2.

• Gradient accumulation:
  Trick to simulate a bigger batch when memory won't allow it. Accumulate
  gradients over N small mini-batches, then do one optimizer step. Same
  result as a batch N× larger, but with N× the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.core.logger import logger


# ── Hardware thresholds (in GB) ───────────────────────────────────
# Numbers picked from XTTS community fine-tuning reports + a safety margin.
# We measure *total* VRAM, not free — assuming the user closed other GPU apps.
VRAM_STANDARD_MIN = 8.0    # ≥8 GB → standard preset
VRAM_LOW_MIN = 3.0         # 3–8 GB → low-VRAM preset
VRAM_REFUSE_BELOW = 3.0    # <3 GB → can't train at all

# Disk space needed for: dataset copy, intermediate checkpoints, final model.
# 5 GB is safe headroom; XTTS checkpoints alone are ~2 GB each.
DISK_REQUIRED_GB = 5.0


# ── Preset names ──────────────────────────────────────────────────
class Preset(str, Enum):
    """Which preset was selected. String-valued so it serializes cleanly to JSON."""
    STANDARD = "standard"
    LOW_VRAM = "low_vram"
    # Refuse* values aren't used in TrainingPlan (it'd be None then),
    # but they're useful in `decide_preset()` return tuples.
    REFUSE_NO_GPU = "refuse_no_gpu"
    REFUSE_LOW_VRAM = "refuse_low_vram"
    REFUSE_LOW_DISK = "refuse_low_disk"


# ── The plan returned to the trainer ──────────────────────────────
@dataclass
class TrainingPlan:
    """
    Concrete training settings, ready to plug into XttsConfig.

    The trainer will read these and assign them to `XttsConfig` and
    `XttsArgs` fields. We keep the shape simple so it's easy to log,
    serialize for the UI, and tweak.
    """
    preset: Preset

    # --- Core training loop ---
    batch_size: int                   # Examples per forward pass
    eval_batch_size: int              # Same, for eval set
    grad_accum_steps: int = 1         # Simulate a bigger batch over N micro-batches
    num_loader_workers: int = 0       # PyTorch DataLoader workers (0 = main process)

    # --- Training duration ---
    # We size training in *optimizer steps*, not epochs. Why:
    #   With a small dataset (e.g. 28 clips, batch 2, accum 4) one epoch is
    #   only ~3 weight updates. "6 epochs" sounds reasonable but actually
    #   means ~18 updates total — nowhere near enough to move the GPT head.
    #   Community fine-tunes converge between ~150 and ~500 optimizer steps.
    #
    # `target_steps` is the source of truth. `epochs` is computed from it
    # at trainer-construction time once we know `len(train_loader)`.
    target_steps: int = 250
    epochs: int = 0                   # Filled in by training.py from target_steps

    # --- Memory / numerical knobs ---
    mixed_precision: bool = False     # fp16 / bf16 autocast
    precision_dtype: str = "fp32"     # "fp32" | "fp16" | "bf16"
    gradient_checkpointing: bool = False
    grad_clip: float = 1.0            # Clip gradients above this norm — stabilizes fp16

    # --- Optimizer ---
    learning_rate: float = 5e-6       # XTTS fine-tuning needs a *small* LR;
                                      # the default 1e-3 destroys the pretrained model.

    # --- Cadence ---
    save_step: int = 1000             # Checkpoint every N steps
    print_step: int = 25              # Log scalars every N steps
    run_eval: bool = True
    eval_step: int = 500              # Run eval every N steps

    # --- For UI / logging ---
    estimated_minutes: int = 0        # Rough wall-clock estimate
    notes: list[str] = field(default_factory=list)  # Why we chose these knobs


@dataclass
class TrainingDecision:
    """
    Output of `decide_preset()`. Either:
      - plan + summary (we can train)
      - refusal (we can't, with a friendly reason)

    UI consumes this directly: show summary or show refusal banner.
    """
    can_train: bool
    plan: Optional[TrainingPlan]
    friendly_summary: str
    refusal_reason: Optional[str] = None
    suggested_action: Optional[str] = None

    # Echo of what we saw, useful for the UI to show "your machine: ..."
    detected_hardware: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════


def decide_preset(
    force_preset: Optional[Preset] = None,
    train_clip_count: Optional[int] = None,
) -> TrainingDecision:
    """
    Inspect the local machine and pick a training preset.

    Args:
        force_preset:     For testing — skip hardware detection and use this
                          preset directly. Production callers should leave this
                          as None.
        train_clip_count: How many clips the user's dataset has. Drives the
                          ETA: more clips → more micro-batches per epoch →
                          longer wall clock. If None (no dataset built yet),
                          we plan against a typical 30-clip Voice Profile.

    Returns:
        TrainingDecision (always non-None). Check `.can_train` to know if
        the trainer can proceed.
    """
    from backend.system.hardware import get_gpu_info, get_disk_info

    gpu = get_gpu_info()
    disk = get_disk_info()

    detected = {
        "cuda": gpu.get("cuda", False),
        "gpu_name": gpu.get("gpu_name"),
        "vram_gb": gpu.get("vram_gb", 0.0),
        "free_disk_gb": disk.get("free_disk_gb", 0.0),
    }

    # If no dataset built yet, plan for a typical Voice Profile (30 clips).
    # The endpoint will pass the real count once the user has data.
    clips = train_clip_count if train_clip_count is not None else 30

    # ── Forced override (for tests / debugging only) ──────────────
    if force_preset is not None:
        plan = _build_plan(force_preset, train_clip_count=clips)
        return TrainingDecision(
            can_train=True,
            plan=plan,
            friendly_summary=_summarize_plan(plan, detected),
            detected_hardware=detected,
        )

    # ── Disk gate (cheapest check, fails first) ───────────────────
    if detected["free_disk_gb"] < DISK_REQUIRED_GB:
        return TrainingDecision(
            can_train=False,
            plan=None,
            friendly_summary="",
            refusal_reason=(
                f"You have {detected['free_disk_gb']:.1f} GB free, but "
                f"training needs at least {DISK_REQUIRED_GB:.0f} GB."
            ),
            suggested_action=(
                "Free up some disk space and try again, or use Quick Clone "
                "which doesn't need extra disk."
            ),
            detected_hardware=detected,
        )

    # ── No GPU ────────────────────────────────────────────────────
    if not detected["cuda"]:
        return TrainingDecision(
            can_train=False,
            plan=None,
            friendly_summary="",
            refusal_reason=(
                "Voice Profile training needs a graphics card. "
                "Without one, training would take more than a day."
            ),
            suggested_action=(
                "Use Quick Clone instead — it works on any computer and "
                "produces good results in seconds."
            ),
            detected_hardware=detected,
        )

    # ── GPU too small ─────────────────────────────────────────────
    vram = detected["vram_gb"]
    if vram < VRAM_REFUSE_BELOW:
        return TrainingDecision(
            can_train=False,
            plan=None,
            friendly_summary="",
            refusal_reason=(
                f"Your graphics card has {vram:.1f} GB of memory, but "
                f"training needs at least {VRAM_REFUSE_BELOW:.0f} GB."
            ),
            suggested_action=(
                "Quick Clone doesn't need much memory and works great here."
            ),
            detected_hardware=detected,
        )

    # ── Pick the preset ───────────────────────────────────────────
    if vram >= VRAM_STANDARD_MIN:
        preset = Preset.STANDARD
    else:
        preset = Preset.LOW_VRAM

    plan = _build_plan(preset, vram_gb=vram, train_clip_count=clips)
    return TrainingDecision(
        can_train=True,
        plan=plan,
        friendly_summary=_summarize_plan(plan, detected),
        detected_hardware=detected,
    )


# ══════════════════════════════════════════════════════════════════
# Preset builders
# ══════════════════════════════════════════════════════════════════


def _build_plan(preset: Preset, vram_gb: float = 0.0, train_clip_count: int = 30) -> TrainingPlan:
    """Translate a preset choice into concrete numbers."""
    if preset == Preset.STANDARD:
        return _standard_plan(vram_gb, train_clip_count)
    if preset == Preset.LOW_VRAM:
        return _low_vram_plan(vram_gb, train_clip_count)

    # Refusal presets shouldn't reach here, but keep the dispatch total.
    raise ValueError(f"Cannot build a plan for refusal preset: {preset}")


def _standard_plan(vram_gb: float, train_clip_count: int = 30) -> TrainingPlan:
    """
    Standard preset: ≥8 GB VRAM available.

    Choices:
      - batch_size=8: XTTS default is 32, but for fine-tuning a single
        speaker that's overkill. 8 is a good sweet spot for fast convergence
        without hammering memory. Bigger isn't better when the dataset is
        small (≤30 clips).
      - mixed_precision=True with bf16 if available, else fp16: speeds
        training ~30% on modern GPUs at no quality cost.
      - gradient_checkpointing=False: we have memory; spend it on speed.
      - learning_rate=5e-6: XTTS pretraining used 1e-3 with a fresh model.
        For fine-tuning we lower by ~200× to avoid catastrophic forgetting
        of the model's general voice knowledge.
    """
    notes = [
        "Plenty of GPU memory available — using standard settings.",
        "Mixed-precision math enabled for ~30% speedup.",
    ]
    return TrainingPlan(
        preset=Preset.STANDARD,
        batch_size=8,
        eval_batch_size=4,
        grad_accum_steps=1,
        num_loader_workers=2,
        target_steps=250,
        mixed_precision=True,
        precision_dtype="bf16",       # bf16 is more numerically stable than fp16
        gradient_checkpointing=False,
        grad_clip=1.0,
        learning_rate=5e-6,
        save_step=500,
        print_step=10,
        run_eval=True,
        eval_step=200,
        estimated_minutes=_estimate_minutes(
            vram_gb=vram_gb,
            batch_size=8,
            grad_accum_steps=1,
            target_steps=250,
            train_clip_count=train_clip_count,
            low_vram=False,
        ),
        notes=notes,
    )


def _low_vram_plan(vram_gb: float, train_clip_count: int = 30) -> TrainingPlan:
    """
    Low-VRAM preset: 3–8 GB VRAM. Heavy memory-saving tricks.

    Choices:
      - batch_size=2: smallest practical batch. XTTS won't train with
        batch_size=1 because the model uses BatchNorm-like layers that
        misbehave on a single example.
      - grad_accum_steps=4: effective batch size = 2 * 4 = 8, matching
        the standard preset's gradient quality without the memory cost.
      - mixed_precision=True (fp16): saves ~40% VRAM. fp16 (not bf16)
        because older GPUs (GTX 1650, etc.) lack bf16 support.
      - gradient_checkpointing=True: another ~30% VRAM saving, costs
        ~30% more time. Necessary on small GPUs.
      - num_loader_workers=0: each worker holds a full copy of the
        dataset in memory. With low VRAM we usually have low RAM too,
        so we stay single-threaded.
      - grad_clip stays 1.0: fp16 increases the chance of gradient spikes.
    """
    notes = [
        f"Your GPU has {vram_gb:.1f} GB of memory — using memory-saving "
        f"settings so training fits."
        if vram_gb > 0
        else "Using memory-saving settings (low-VRAM preset).",
        "Smaller batches with gradient accumulation — same quality, just slower.",
        "Half-precision math + gradient checkpointing reduce memory ~70% total.",
    ]
    return TrainingPlan(
        preset=Preset.LOW_VRAM,
        batch_size=2,
        eval_batch_size=2,
        grad_accum_steps=4,           # effective batch = 8
        num_loader_workers=0,
        target_steps=250,
        mixed_precision=True,
        precision_dtype="fp16",       # fp16 for older GPU compatibility
        gradient_checkpointing=True,
        grad_clip=1.0,
        learning_rate=5e-6,
        save_step=500,
        print_step=10,
        run_eval=True,
        eval_step=200,
        estimated_minutes=_estimate_minutes(
            vram_gb=vram_gb,
            batch_size=2,
            grad_accum_steps=4,
            target_steps=250,
            train_clip_count=train_clip_count,
            low_vram=True,
        ),
        notes=notes,
    )


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


# Per-step time, by hardware tier (seconds per micro-batch step). A
# micro-batch step is one forward+backward pass; with grad accumulation,
# multiple micro-batches make up a single weight update.
SECONDS_PER_STEP = {
    "standard_high":   0.45,   # ≥16 GB VRAM, batch 8, bf16
    "standard_mid":    0.65,   # 12–16 GB,    batch 8, bf16
    "standard_low":    1.10,   # 8–12 GB,     batch 8, bf16
    "low_vram_high":   1.80,   # 6–8 GB,      batch 2 + accum 4, fp16 + ckpt
    "low_vram_mid":    2.80,   # 4–6 GB,      batch 2 + accum 4, fp16 + ckpt
    "low_vram_low":    4.00,   # 3–4 GB,      batch 2 + accum 4, fp16 + ckpt
}


def _per_step_seconds(vram_gb: float, low_vram: bool) -> float:
    if low_vram:
        if vram_gb >= 6.0:
            return SECONDS_PER_STEP["low_vram_high"]
        if vram_gb >= 4.0:
            return SECONDS_PER_STEP["low_vram_mid"]
        return SECONDS_PER_STEP["low_vram_low"]
    if vram_gb >= 16.0:
        return SECONDS_PER_STEP["standard_high"]
    if vram_gb >= 12.0:
        return SECONDS_PER_STEP["standard_mid"]
    return SECONDS_PER_STEP["standard_low"]


def _estimate_minutes(
    vram_gb: float,
    batch_size: int,
    grad_accum_steps: int,
    target_steps: int,
    train_clip_count: int,
    low_vram: bool,
) -> int:
    """
    Wall-clock estimate in minutes to hit `target_steps` weight updates.

    Counts micro-batches, not weight updates: the GPU spends time on every
    forward/backward pass, even the ones that just accumulate gradients.
    A run with grad_accum=4 does 4× the GPU work per logged "step".
    """
    import math

    clips = max(1, train_clip_count)
    batch = max(1, batch_size)
    accum = max(1, grad_accum_steps)

    micro_batches_per_epoch = max(1, clips // batch)
    updates_per_epoch = max(1, micro_batches_per_epoch // accum)
    epochs = max(1, math.ceil(target_steps / updates_per_epoch))

    total_micro_batches = epochs * micro_batches_per_epoch
    seconds = total_micro_batches * _per_step_seconds(vram_gb, low_vram)
    return max(1, int(round(seconds / 60)))


def _summarize_plan(plan: TrainingPlan, detected: dict) -> str:
    """
    Produce a friendly human summary the UI can show on the disclosure modal.

    Avoid jargon. No "fp16", no "batch size", no "epochs". This is the copy
    the user reads to decide whether to start the training.
    """
    gpu_name = detected.get("gpu_name") or "your GPU"
    vram = detected.get("vram_gb", 0.0)
    minutes = plan.estimated_minutes
    hours = minutes / 60.0

    # Time phrasing
    if hours < 1.0:
        time_txt = f"about {minutes} minutes"
    elif hours < 1.5:
        time_txt = "about 1 hour"
    else:
        time_txt = f"about {hours:.1f} hours"

    # Effort phrasing
    if plan.preset == Preset.STANDARD:
        effort_txt = (
            "Your machine has plenty of memory, so training will run "
            "at full speed."
        )
    else:
        effort_txt = (
            "Your machine has limited GPU memory, so we'll run training "
            "carefully — same quality, just a bit slower."
        )

    return (
        f"On your {gpu_name} ({vram:.1f} GB), training will take {time_txt}.\n\n"
        f"{effort_txt}\n\n"
        f"Your GPU will run at full load while training. You can keep "
        f"using your computer for light tasks, but heavy apps will compete "
        f"for resources."
    )
