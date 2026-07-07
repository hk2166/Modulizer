"""
training.py — XTTS fine-tuning loop wrapper with progress callbacks.

WHAT THIS MODULE DOES
─────────────────────
Coqui ships a ready-made training loop in their `trainer` package. We
don't reinvent it. What we add is:

  1. A small **adapter** (`ProgressBridge`) that listens to Coqui's
     callback events and translates them into our `JobManager`'s
     progress updates — using friendly UX copy, no ML jargon.

  2. A single entrypoint, `run_training(project_id, job_id)`, that
     glues together everything M2 has built so far:

         training_config.decide_preset()  →  picks safe knobs for this GPU
         dataset_builder.build_dataset()  →  produced LJSpeech metadata
         GPTTrainerConfig / GPTArgs       →  Coqui's XTTS fine-tune model
         Trainer(callbacks=bridge.dict()) →  loop with our progress hooks

THE ADAPTER PATTERN, IN ONE PARAGRAPH
─────────────────────────────────────
Coqui's `Trainer` accepts `callbacks: dict[str, Callable]` — keys like
`on_train_step_end` map to functions that get `trainer` as their only
arg. So we build a `ProgressBridge` object whose methods match those
keys exactly, ask it for `.callbacks_dict()`, and hand the dict to
`Trainer(callbacks=...)`. The bridge owns the state we need (job_id,
total_epochs, last percent we pushed to avoid spamming) and translates
trainer events into UI-friendly progress.

UX RULES (from TASKS.md)
────────────────────────
The user must never see ML jargon. We translate:

  epoch        → "round"          (e.g. "round 3 of 6")
  step / batch → hidden          (we use it for percent only)
  loss         → hidden          (no numbers like 2.43)
  checkpoint   → "saved progress"

Progress messages follow the shape:
    early   → "Getting things ready..."
    middle  → "Learning your voice (round 2 of 6)..."
    late    → "Almost ready (round 5 of 6)..."
    done    → "Wrapping up..."

CANCELLATION
────────────
The cooperative-cancellation task (next on the M2 list) will set a
`threading.Event`. We accept it here as `cancel_event` and check it
in `on_train_step_end` — when set, we raise `KeyboardInterrupt`,
which Coqui already handles cleanly (saves a checkpoint and exits).
Today we just thread the seam through; the API endpoint that flips
the flag comes in the next task.

WHY THE "RUN_TRAINING" FUNCTION ISN'T A METHOD
──────────────────────────────────────────────
It's pure procedure: take inputs, do work, return a result. No state
worth carrying across calls. A free function keeps the call sites in
the API layer trivial:

    background_tasks.add_task(run_training, project_id, job.id)

If we ever need to keep state (live trainer ref for cancellation, e.g.)
we'll wrap *that* in a small registry, not turn this into a class.
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend.core.logger import logger
from backend.core.settings import DATA_DIR
from backend.jobs.instance import job_manager
from backend.pipelines.training_config import (
    Preset,
    TrainingDecision,
    TrainingPlan,
    decide_preset,
)


# ══════════════════════════════════════════════════════════════════
# Public result type
# ══════════════════════════════════════════════════════════════════


@dataclass
class TrainingResult:
    """
    What we hand back to JobManager on completion.

    The API layer turns this into the JSON `result` field on the job.
    Keep it flat and JSON-serializable.
    """
    success: bool
    project_id: str
    output_dir: str | None = None          # checkpoints/ folder for this run
    best_checkpoint: str | None = None     # path to best.pth, if found
    last_checkpoint: str | None = None     # most recent checkpoint
    epochs_run: int = 0
    total_steps: int = 0
    error: str | None = None
    refusal: dict | None = None            # if hardware gate refused us


# ══════════════════════════════════════════════════════════════════
# ProgressBridge — the heart of this module
# ══════════════════════════════════════════════════════════════════


@dataclass
class _BridgeState:
    """
    Mutable state the bridge keeps across callback fires.

    We keep this in a tiny dataclass instead of bare attributes so it's
    obvious in logs / debugging what survives between calls.
    """
    last_percent: int = -1            # last value we pushed to JobManager
    steps_per_epoch: int = 0          # discovered when train_loader is built
    total_steps_planned: int = 0      # epochs * steps_per_epoch (for percent)
    epochs_done: int = 0
    last_message: str = ""
    # ETA tracking — seeded from the static plan estimate, refined as rounds
    # complete using observed wall-clock time. None until first round end.
    run_started_at: float = 0.0       # time.monotonic() at on_init_end
    last_eta_seconds: int | None = None
    # Early-stopping tracking. patience=0 means disabled.
    best_eval_loss: float = float("inf")
    rounds_without_improvement: int = 0


class ProgressBridge:
    """
    Coqui Trainer callback adapter → JobManager.update_progress.

    Usage:

        bridge = ProgressBridge(
            job_id="...", total_epochs=6, cancel_event=event,
        )
        trainer = Trainer(..., callbacks=bridge.callbacks_dict())
        trainer.fit()

    WHAT'S WORTH LEARNING HERE
    ──────────────────────────
    • The callback dict trick: Coqui matches dict keys to known event
      names (`on_train_step_end`, etc.). Anything not in their allow-list
      throws a ValueError. We only register the events we actually use.

    • Throttling: training fires `on_train_step_end` *thousands* of
      times. Calling `update_progress` on every step would spam logs and
      churn the UI. We only push when the integer percent has moved up.

    • Determining total work: at construction time we know epochs but
      not steps. The train_loader is built lazily. So `_steps_per_epoch`
      is filled in from the first `on_epoch_start`, when `trainer.train_loader`
      exists. After that, total = epochs * steps_per_epoch.

    • The bridge is *pure observation*. It doesn't mutate the trainer.
      That separation is what makes it safe to reuse and easy to test
      (mock trainer, fire callback, assert update_progress was called).
    """

    def __init__(
        self,
        job_id: str,
        total_epochs: int,
        cancel_event: Optional[threading.Event] = None,
        update_fn: Optional[Callable[..., None]] = None,
        initial_eta_seconds: Optional[int] = None,
        early_stopping_patience: int = 3,
        project_id: Optional[str] = None,
        language: str = "en",
    ) -> None:
        """
        Args:
            job_id:        JobManager job id this run reports against.
            total_epochs:  Plan's epoch count, used for the "round X of Y"
                           UI copy and percent denominator.
            cancel_event:  Threading event the cancellation endpoint flips.
            update_fn:     Override for testing. Defaults to job_manager.update_progress.
            initial_eta_seconds: Static planning estimate, refined each round.
            early_stopping_patience: Stop after this many rounds with no eval-loss
                           improvement. 0 = disabled. Default 3 rounds.
            project_id:    Used for validation sample synthesis after each round.
            language:      Language code for validation sample synthesis.
        """
        self.job_id = job_id
        self.total_epochs = max(1, total_epochs)
        self.cancel_event = cancel_event
        self._update_fn = update_fn or job_manager.update_progress
        self.state = _BridgeState()
        self.state.last_eta_seconds = initial_eta_seconds
        self.early_stopping_patience = early_stopping_patience
        self.project_id = project_id
        self.language = language

    # ── Coqui plumbing ────────────────────────────────────────────

    def callbacks_dict(self) -> dict:
        """
        Build the dict Coqui's Trainer accepts as `callbacks=...`.

        Only register the events we actually handle — the parser raises
        ValueError on unknown keys. on_init_end/on_epoch_start/
        on_train_step_end/on_epoch_end give us a smooth percent without
        being noisy.
        """
        return {
            "on_init_end": self.on_init_end,
            "on_epoch_start": self.on_epoch_start,
            "on_train_step_end": self.on_train_step_end,
            "on_epoch_end": self.on_epoch_end,
            "on_keyboard_interrupt": self.on_keyboard_interrupt,
        }

    # ── Lifecycle handlers (fired by Coqui's TrainerCallback) ─────

    def on_init_end(self, trainer) -> None:
        """
        Trainer finished its constructor. Model is built, optimizer ready,
        loaders may or may not be built yet. Push a friendly start message.
        """
        import time
        self.state.run_started_at = time.monotonic()
        self._push(2, "Getting things ready...")

    def on_epoch_start(self, trainer) -> None:
        """
        About to start a fresh epoch. By this point `trainer.train_loader`
        exists, so we can finally compute total steps.

        We update `steps_per_epoch` once — it can vary slightly between
        epochs in some setups, but never enough to matter for a progress
        bar. First-epoch value is fine for the whole run.
        """
        if self.state.steps_per_epoch == 0:
            try:
                # train_loader supports len() since it's a torch DataLoader
                steps = len(trainer.train_loader) if trainer.train_loader is not None else 0
            except TypeError:
                # Some loaders (IterableDataset) don't support len.
                # Fall back to per-epoch percent only.
                steps = 0
            self.state.steps_per_epoch = max(steps, 1)
            self.state.total_steps_planned = self.state.steps_per_epoch * self.total_epochs

        round_num = getattr(trainer, "epochs_done", self.state.epochs_done) + 1
        self._push(
            self._compute_percent(trainer),
            f"Learning your voice (round {round_num} of {self.total_epochs})...",
        )

    def on_train_step_end(self, trainer) -> None:
        """
        Most frequent callback — fires after every training step. Two jobs:
          1. Push a smoothly increasing percent (throttled).
          2. Honour cancellation by raising KeyboardInterrupt.

        Cancellation note: Coqui's Trainer wraps the training loop in a
        try/except KeyboardInterrupt that calls `on_keyboard_interrupt`
        (which saves a checkpoint). Raising here is the documented way
        to stop training cleanly.
        """
        if self.cancel_event is not None and self.cancel_event.is_set():
            # Friendly status before we tear down
            self._push(self.state.last_percent, "Stopping...")
            raise KeyboardInterrupt("Training cancelled by user request.")

        percent = self._compute_percent(trainer)
        if percent <= self.state.last_percent:
            return  # throttle — only push when percent actually moves

        # Pick the message bucket from how far along we are
        round_num = getattr(trainer, "epochs_done", self.state.epochs_done) + 1
        self._push(percent, self._message_for_percent(percent, round_num))

    def on_epoch_end(self, trainer) -> None:
        """
        A round finished. Recompute ETA, check early-stopping, synthesize
        a validation sample, and push "saved progress" message.
        """
        import time

        self.state.epochs_done = getattr(trainer, "epochs_done", self.state.epochs_done + 1)
        percent = self._compute_percent(trainer)

        # ── ETA from real wall-clock ──────────────────────────────
        if self.state.run_started_at > 0 and self.state.epochs_done > 0:
            elapsed = time.monotonic() - self.state.run_started_at
            avg_per_round = elapsed / self.state.epochs_done
            remaining_rounds = max(0, self.total_epochs - self.state.epochs_done)
            self.state.last_eta_seconds = int(remaining_rounds * avg_per_round)

        # ── Early stopping ────────────────────────────────────────
        # Read the current eval loss from the trainer. Coqui tracks
        # `keep_avg_eval` (a running average of recent eval steps).
        # We skip early-stopping if patience==0 or no eval is running.
        if self.early_stopping_patience > 0:
            try:
                eval_loss = float(
                    getattr(trainer, "keep_avg_eval", {}).get("eval_loss", float("inf"))
                    or float("inf")
                )
                if eval_loss < self.state.best_eval_loss - 1e-4:
                    self.state.best_eval_loss = eval_loss
                    self.state.rounds_without_improvement = 0
                else:
                    self.state.rounds_without_improvement += 1
                    logger.info(
                        f"training: no improvement for "
                        f"{self.state.rounds_without_improvement}/"
                        f"{self.early_stopping_patience} rounds "
                        f"(eval_loss={eval_loss:.4f})"
                    )
                    if self.state.rounds_without_improvement >= self.early_stopping_patience:
                        logger.info("training: early stopping triggered")
                        self._push(percent, "Best quality reached — wrapping up early.")
                        raise KeyboardInterrupt("Early stopping: no eval-loss improvement.")
            except KeyboardInterrupt:
                raise  # re-raise early-stop signal
            except Exception as e:
                logger.debug(f"training: early-stopping eval read failed: {e}")

        # ── Validation sample synthesis ───────────────────────────
        # Synthesise a short phrase with the current partially-trained model
        # so the user can hear their voice improving round by round.
        # We run this synchronously here (adds ~5–15 s per round on CPU/GPU)
        # because it has to happen before ProgressBridge pushes the round-end
        # update so the job result carries the new sample path.
        if self.project_id:
            sample_path = _synthesize_validation_sample(
                trainer=trainer,
                project_id=self.project_id,
                language=self.language,
                epoch=self.state.epochs_done,
            )
            if sample_path:
                # Attach the sample path to the job so the UI can play it.
                # We call update_progress once with just the sample path;
                # the main _push call below carries the human message.
                try:
                    self._update_fn(
                        self.job_id, self.state.last_percent,
                        self.state.last_message,
                        eta_seconds=self.state.last_eta_seconds,
                        validation_sample_path=sample_path,
                    )
                except Exception as e:
                    logger.warning(f"ProgressBridge: sample path update failed: {e}")

        self._push(
            percent,
            f"Saved progress (round {self.state.epochs_done} of {self.total_epochs}).",
        )

    def on_keyboard_interrupt(self, trainer) -> None:
        """
        Coqui calls this when the loop catches a KeyboardInterrupt — either
        from us (cancellation) or from a real Ctrl+C. Don't fail the job
        here; let `run_training` decide based on whether `cancel_event` was
        the cause.
        """
        self._push(self.state.last_percent, "Stopping...")

    # ── Internals ─────────────────────────────────────────────────

    def _compute_percent(self, trainer) -> int:
        """
        Map (epochs_done, total_steps_done) onto a 0..99 integer.

        We cap at 99 here — `run_training` pushes the final 100 itself
        on success, so the UI doesn't briefly hit 100% mid-run if the
        last step lands exactly on the boundary.
        """
        steps_done = getattr(trainer, "total_steps_done", 0)
        if self.state.total_steps_planned > 0:
            frac = steps_done / self.state.total_steps_planned
        else:
            # Loaders without len → fall back to whole-epoch granularity
            frac = self.state.epochs_done / self.total_epochs
        return max(0, min(99, int(frac * 100)))

    @staticmethod
    def _message_for_percent(percent: int, round_num: int) -> str:
        """Plain-English status copy. No jargon."""
        if percent < 5:
            return "Getting things ready..."
        if percent < 30:
            return f"Listening to your voice (round {round_num})..."
        if percent < 70:
            return f"Learning your voice (round {round_num})..."
        if percent < 95:
            return f"Almost ready (round {round_num})..."
        return "Wrapping up..."

    def _push(self, percent: int, message: str) -> None:
        """
        Forward to JobManager but only when something changed. Avoids
        log spam when many steps share the same integer percent.

        Also enforces *monotonic* percent — once we say 12%, we never go
        back to 8%. The bridge can be called from multiple lifecycle
        events that compute their percent off slightly different state
        (e.g. on_init_end pushes a baseline before on_epoch_start sees
        any steps), and the UX requirement is that the bar only goes up.
        """
        if percent < self.state.last_percent:
            percent = self.state.last_percent  # clamp, never regress
        if percent == self.state.last_percent and message == self.state.last_message:
            return
        self.state.last_percent = percent
        self.state.last_message = message
        try:
            self._update_fn(
                self.job_id, percent, message,
                eta_seconds=self.state.last_eta_seconds,
            )
        except Exception as e:
            # Never let a logging hiccup take training down.
            logger.warning(f"ProgressBridge: update_progress failed: {e}")


# ══════════════════════════════════════════════════════════════════
# run_training — the public entrypoint
# ══════════════════════════════════════════════════════════════════


def run_training(
    project_id: str,
    job_id: str,
    *,
    language: str = "en",
    cancel_event: Optional[threading.Event] = None,
    force_preset: Optional[Preset] = None,
) -> TrainingResult:
    """
    End-to-end XTTS fine-tune for a project.

    Steps:
      1. Hardware gate via `decide_preset()`. Refuse early if we can't train.
      2. Verify the project's dataset has been built.
      3. Build XTTS GPT trainer config / args from the chosen plan.
      4. Construct Coqui's `Trainer` with our `ProgressBridge` plugged in.
      5. Run `trainer.fit()`.
      6. Pick best/last checkpoint, return a `TrainingResult`.

    The function is intentionally sync — it's run inside a FastAPI
    BackgroundTask, which already handles thread isolation.

    Raises nothing — failures are returned as `TrainingResult(success=False)`
    so the caller can shape JobManager state cleanly.
    """
    logger.info(f"training.run_training: project={project_id} job={job_id}")

    # ── 1. Hardware gate ──────────────────────────────────────────
    decision: TrainingDecision = decide_preset(force_preset=force_preset)
    if not decision.can_train or decision.plan is None:
        msg = decision.refusal_reason or "This machine can't run training."
        logger.warning(f"training: refusal — {msg}")
        return TrainingResult(
            success=False,
            project_id=project_id,
            error=msg,
            refusal={
                "reason": decision.refusal_reason,
                "suggested_action": decision.suggested_action,
                "detected_hardware": decision.detected_hardware,
            },
        )
    plan: TrainingPlan = decision.plan

    # ── 2. Dataset check ──────────────────────────────────────────
    project_dir = Path(DATA_DIR) / "projects" / project_id
    dataset_dir = project_dir / "dataset"
    metadata_csv = dataset_dir / "metadata.csv"
    if not metadata_csv.exists():
        return TrainingResult(
            success=False,
            project_id=project_id,
            error="No dataset found. Build the dataset first.",
        )
    eval_csv = dataset_dir / "metadata_eval.csv"

    output_dir = project_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect the dataset's actual sample rate from the first clip. The
    # XTTS dataset loader will resample on-the-fly to whatever we set
    # in `XttsAudioConfig.sample_rate`, but matching it to the source
    # avoids wasted work on every step. (Our preprocessor currently
    # writes 24 kHz; long-term plan is to switch that to 22050 to align
    # with XTTS GPT's native rate — see TASKS.md.)
    dataset_sr = _peek_sample_rate(dataset_dir / "wavs")

    # ── 3. Build XTTS configs ─────────────────────────────────────
    # We import these heavy modules inside the function so the FastAPI
    # process doesn't pay the import cost at startup. Coqui's TTS pulls
    # in torch, transformers, einops, etc. — multi-second imports.
    try:
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import (
            GPTArgs,
            GPTTrainer,
            GPTTrainerConfig,
            XttsAudioConfig,
        )
    except ImportError as e:
        return TrainingResult(
            success=False,
            project_id=project_id,
            error=(
                "Training dependencies aren't installed. "
                f"Original error: {e}"
            ),
        )

    base_files = _ensure_xtts_base_files()
    if base_files is None:
        return TrainingResult(
            success=False,
            project_id=project_id,
            error=(
                "Voice engine files aren't downloaded yet. Run a Quick "
                "Clone once first to fetch them, then try again."
            ),
        )

    # LJSpeech dataset wiring — the builder produced this layout
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name=f"voiceforge-{project_id}",
        path=str(dataset_dir),
        meta_file_train="metadata.csv",
        meta_file_val=eval_csv.name if eval_csv.exists() else "",
        language=language,
    )

    # ── XTTS GPT trainer config ──
    # Translating our friendly TrainingPlan into Coqui's XttsConfig fields.
    # Field names come from GPTTrainerConfig (XttsConfig + extras).
    #
    # Sample rate notes:
    #   - audio.sample_rate / dvae_sample_rate must be the SAME as the rate
    #     the dataset is loaded at (XttsDataset resamples to this rate).
    #     We use the actual rate of our processed wavs to skip a resample.
    #   - output_sample_rate stays 24 kHz — that's the HiFi-GAN decoder's
    #     synthesis rate, independent of the GPT/dvae internal rate.
    audio_config = XttsAudioConfig(
        sample_rate=dataset_sr,
        dvae_sample_rate=dataset_sr,
        output_sample_rate=24000,
    )
    # Reduce allocator fragmentation on tight cards. Must be set before the
    # first CUDA allocation to take effect; setting it here is safe because
    # the heavy torch/CUDA work happens below in trainer.fit().
    if plan.expandable_segments:
        prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in prev:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                (prev + "," if prev else "") + "expandable_segments:True"
            )
            logger.info("training: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

    model_args = GPTArgs(
        max_conditioning_length=132300,
        min_conditioning_length=66150,
        debug_loading_failures=False,
        max_wav_length=plan.max_wav_length,
        max_text_length=plan.max_text_length,
        mel_norm_file=base_files["mel_norm"],
        dvae_checkpoint=base_files["dvae"],
        xtts_checkpoint=base_files["xtts"],
        tokenizer_file=base_files["tokenizer"],
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    # ── 4. Load samples first so we can size the schedule ────────
    # We need to know how many training samples there are before we can
    # convert `target_steps` (the source of truth) into `epochs`.
    # eval_split=True lets Coqui load the meta_file_val eval CSV when present;
    # if there's no eval CSV it carves a small split off train.
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=eval_csv.exists(),
        eval_split_max_size=256,
        eval_split_size=0.1,
    )
    if not train_samples:
        return TrainingResult(
            success=False,
            project_id=project_id,
            error="Dataset is empty after loading. Rebuild the dataset.",
        )

    # Minimum-clip guard. XTTS fine-tuning needs a meaningful number of
    # clips — too few and the GPT head overfits to a couple of utterances,
    # producing a voice that only says those exact phrases well and garbles
    # everything else. We refuse below a floor rather than train something
    # broken. (The Voice Profile UX targets ~30 clips.)
    MIN_TRAIN_CLIPS = 5
    if len(train_samples) < MIN_TRAIN_CLIPS:
        return TrainingResult(
            success=False,
            project_id=project_id,
            error=(
                f"Only {len(train_samples)} usable clip"
                f"{'s' if len(train_samples) != 1 else ''} found — "
                f"Voice Profile training needs at least {MIN_TRAIN_CLIPS}. "
                f"Record more clips (aim for ~30) and try again."
            ),
        )

    # Whether we have a real eval set. If the builder didn't produce one
    # (tiny dataset), disable eval so the trainer doesn't crash trying to
    # evaluate against zero samples.
    have_eval = eval_csv.exists() and bool(eval_samples)

    # Coqui's loader yields one optimizer micro-batch per `batch_size`
    # samples. With grad accumulation, true weight updates = micro_batches
    # / accum. Convert target_steps (true updates) into epoch count.
    micro_batches_per_epoch = max(1, len(train_samples) // plan.batch_size)
    updates_per_epoch = max(1, micro_batches_per_epoch // max(1, plan.grad_accum_steps))
    epochs = max(1, math.ceil(plan.target_steps / updates_per_epoch))
    plan.epochs = epochs   # for the bridge's "round X of Y" copy
    logger.info(
        f"training: {len(train_samples)} train samples, batch={plan.batch_size}, "
        f"accum={plan.grad_accum_steps} → {updates_per_epoch} updates/epoch, "
        f"running {epochs} epochs to hit {plan.target_steps} updates"
    )

    config = GPTTrainerConfig(
        output_path=str(output_dir),
        model_args=model_args,
        run_name=f"voiceforge_{project_id[:8]}",
        project_name="voiceforge",
        run_description="VoiceForge fine-tune",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=plan.batch_size,
        batch_group_size=plan.batch_group_size,
        eval_batch_size=plan.eval_batch_size,
        num_loader_workers=plan.num_loader_workers,
        eval_split_max_size=256,
        print_step=plan.print_step,
        plot_step=100,
        log_model_step=plan.save_step,
        save_step=plan.save_step,
        save_n_checkpoints=2,             # keep best + last; trim the rest
        save_checkpoints=True,
        print_eval=False,
        # Optimizer — AdamW is fast but keeps 2 fp32 momentum buffers per
        # weight. On tight VRAM the plan may pick Adafactor, whose factored
        # second moments use a fraction of the optimizer-state memory.
        optimizer=plan.optimizer,
        optimizer_wd_only_on_weights=(plan.optimizer == "AdamW"),
        optimizer_params=(
            {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}
            if plan.optimizer == "AdamW"
            else {"weight_decay": 1e-2}
        ),
        lr=plan.learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        # Scheduling — epochs derived from target_steps, see above
        epochs=epochs,
        run_eval=plan.run_eval and have_eval,
        run_eval_steps=plan.eval_step,
        # Mixed precision (fp16/bf16)
        mixed_precision=plan.mixed_precision,
        precision=plan.precision_dtype,    # "fp16" | "bf16" | "fp32"
        grad_clip=plan.grad_clip,
        # Datasets
        datasets=[dataset_config],
    )

    # ── 5. Build model ────────────────────────────────────────────
    model = GPTTrainer.init_from_config(config)

    # ── 5. Bridge + Trainer ───────────────────────────────────────
    # Seed the bridge with the static estimate so the UI has an ETA from
    # second 1. After the first round ends, on_epoch_end overrides it with
    # the measured wall-clock value.
    initial_eta = plan.estimated_minutes * 60 if plan.estimated_minutes else None
    bridge = ProgressBridge(
        job_id=job_id,
        total_epochs=plan.epochs,
        cancel_event=cancel_event,
        initial_eta_seconds=initial_eta,
        early_stopping_patience=3,
        project_id=project_id,
        language=language,
    )

    # ── Crash-safe resume: use an existing checkpoint if one exists ────────
    # If training was previously interrupted, continue from the last saved
    # checkpoint rather than starting over. Coqui's Trainer reads restore_path
    # and picks up optimizer state + step count from the checkpoint file.
    _best_existing, _last_existing = _find_checkpoints(output_dir)
    restore_path = str(_last_existing) if _last_existing else ""
    if restore_path:
        logger.info(f"training: resuming from checkpoint {_last_existing.name}")

    # parse_command_line_args=False is critical — Coqui otherwise eats
    # uvicorn's argv and dies on the first unknown flag.
    trainer = Trainer(
        TrainerArgs(
            restore_path=restore_path,
            skip_train_epoch=False,
            grad_accum_steps=plan.grad_accum_steps,
        ),
        config,
        output_path=str(output_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        callbacks=bridge.callbacks_dict(),
        parse_command_line_args=False,
    )

    # ── 6. Run the loop ───────────────────────────────────────────
    cancelled = False
    try:
        trainer.fit()
    except KeyboardInterrupt:
        # Either we raised it (cancel_event), or a real ^C from the host.
        cancelled = cancel_event is not None and cancel_event.is_set()
        logger.info(f"training: stopped (cancelled={cancelled})")
    except Exception as e:
        logger.exception(f"training: trainer.fit() failed: {e}")
        return TrainingResult(
            success=False,
            project_id=project_id,
            output_dir=str(output_dir),
            error=f"Training stopped because of an error: {e}",
            epochs_run=getattr(trainer, "epochs_done", 0),
            total_steps=getattr(trainer, "total_steps_done", 0),
        )

    # ── 7. Locate checkpoints + return ────────────────────────────
    best, last = _find_checkpoints(Path(trainer.output_path))

    if cancelled:
        # Treat cancellation as success — we have a usable checkpoint.
        bridge._push(bridge.state.last_percent, "Stopped — saved progress kept.")
    else:
        bridge._push(100, "Voice profile is ready.")

    return TrainingResult(
        success=True,
        project_id=project_id,
        output_dir=str(trainer.output_path),
        best_checkpoint=str(best) if best else None,
        last_checkpoint=str(last) if last else None,
        epochs_run=getattr(trainer, "epochs_done", 0),
        total_steps=getattr(trainer, "total_steps_done", 0),
    )


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _ensure_xtts_base_files() -> dict | None:
    """
    Locate — and if needed, download — the XTTS base files required for fine-tuning.

    The Coqui TTS inference package bundles model.pth (GPT + HiFi-GAN) and
    vocab.json, but fine-tuning also needs:
      - dvae.pth     — the discrete VAE audio codec (separate from model.pth)
      - mel_stats.pth — mel normalisation statistics

    These aren't included in the standard TTS download because they're only
    needed for training. We fetch them from Coqui's public CDN on first use
    (< 50 MB total, one-time).

    Returns a dict with keys: xtts, dvae, mel_norm, tokenizer — all str paths.
    Returns None if anything is unreachable or missing after download.
    """
    # Official Coqui CDN URLs for the fine-tuning-only files.
    # Same URLs used in TTS/demos/xtts_ft_demo/utils/gpt_train.py.
    DVAE_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_URL = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    try:
        from TTS.utils.manage import ModelManager
    except ImportError:
        return None

    try:
        manager = ModelManager()
        model_path, _, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2"
        )
    except Exception as e:
        logger.warning(f"training: couldn't resolve XTTS base files: {e}")
        return None

    base = Path(model_path)

    # model.pth and vocab.json are guaranteed by the standard inference download.
    required = {
        "xtts":      base / "model.pth",
        "tokenizer": base / "vocab.json",
    }
    for name, path in required.items():
        if not path.exists():
            logger.warning(f"training: missing required XTTS file '{name}' at {path}")
            return None

    # dvae.pth and mel_stats.pth are training-only — download if missing.
    training_extras = {
        "dvae":     (base / "dvae.pth",      DVAE_URL),
        "mel_norm": (base / "mel_stats.pth", MEL_NORM_URL),
    }
    for name, (path, url) in training_extras.items():
        if not path.exists():
            logger.info(f"training: downloading {name} from Coqui CDN → {path.name}")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(path))
                logger.info(f"training: {name} downloaded ({path.stat().st_size // 1024} KB)")
            except Exception as e:
                logger.error(f"training: couldn't download {name}: {e}")
                return None

    return {
        "xtts":      str(required["xtts"]),
        "dvae":      str(training_extras["dvae"][0]),
        "mel_norm":  str(training_extras["mel_norm"][0]),
        "tokenizer": str(required["tokenizer"]),
    }


def _peek_sample_rate(wavs_dir: Path, default: int = 22050) -> int:
    """
    Read the sample rate of the first .wav file in `wavs_dir`.

    Why this matters: XttsDataset resamples every clip to whatever rate
    we set in `XttsAudioConfig.sample_rate`. If that matches the source
    rate, the resample is a no-op (free); if it doesn't, every batch
    pays a torchaudio.functional.resample cost on the data-loader thread.
    Reading the actual rate from disk lets us avoid a configuration drift
    bug where `preprocessor.py` writes 24 kHz but training silently
    assumes 22050.
    """
    try:
        import soundfile as sf
        sample = next(wavs_dir.glob("*.wav"))
        return int(sf.info(str(sample)).samplerate)
    except (StopIteration, FileNotFoundError, Exception) as e:
        logger.warning(f"training: couldn't peek dataset sample rate ({e}); falling back to {default} Hz")
        return default


def _synthesize_validation_sample(
    trainer,
    project_id: str,
    language: str,
    epoch: int,
) -> str | None:
    """
    Synthesise a short validation phrase using the model's current weights.

    Called at the end of each training round so the UI can play the voice
    improving over time. Runs inference directly on the live trainer model
    (in eval mode) rather than loading a checkpoint from disk — faster,
    and previews don't need a perfectly stable checkpoint.

    Returns the absolute path to the generated .wav, or None on any failure.
    Synthesis errors must never stop training.
    """
    VALIDATION_TEXT = {
        "en": "Hello, this is a quick voice check after this round of training.",
        "hi": "नमस्ते, यह प्रशिक्षण के बाद एक त्वरित आवाज़ जाँच है।",
    }
    text = VALIDATION_TEXT.get(language, VALIDATION_TEXT["en"])

    try:
        from backend.core.settings import DATA_DIR
        from backend.services.project_service import get_reference_clip

        ref_clip = get_reference_clip(project_id)
        if ref_clip is None:
            logger.warning("_synthesize_validation_sample: no reference clip found")
            return None

        exports_dir = DATA_DIR / "projects" / project_id / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        out_path = exports_dir / f"validation_epoch_{epoch:03d}.wav"

        # Access the raw XTTS model from the GPTTrainer wrapper.
        xtts = getattr(trainer.model, "xtts", None)
        if xtts is None:
            logger.warning("_synthesize_validation_sample: no xtts on trainer.model")
            return None

        import torch
        xtts.eval()
        with torch.no_grad():
            xtts.tts_to_file(
                text=text,
                speaker_wav=ref_clip,
                language=language,
                file_path=str(out_path),
            )
        xtts.train()

        logger.info(f"validation sample: epoch {epoch} → {out_path.name}")
        return str(out_path.resolve())

    except Exception as e:
        logger.warning(f"_synthesize_validation_sample: failed (epoch {epoch}): {e}")
        return None


def _find_checkpoints(run_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Coqui writes checkpoints into a per-run subfolder of `run_dir`.
    `best_model.pth` is the lowest-loss snapshot; `checkpoint_*.pth` are
    periodic saves. Return (best, latest).
    """
    if not run_dir.exists():
        return None, None

    best = None
    latest = None
    latest_mtime = -1.0

    # Coqui's Trainer creates `run_dir/<run_name>-MMDDYY-HHMMSS/`
    candidate_dirs = [run_dir] + [d for d in run_dir.iterdir() if d.is_dir()]
    for d in candidate_dirs:
        bm = d / "best_model.pth"
        if bm.exists() and best is None:
            best = bm
        for cp in d.glob("checkpoint_*.pth"):
            mt = cp.stat().st_mtime
            if mt > latest_mtime:
                latest_mtime = mt
                latest = cp

    return best, latest
