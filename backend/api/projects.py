import threading

from pathlib import Path

from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from uuid import uuid4

from backend.jobs.instance import job_manager
from backend.audio.preprocessor import preprocess_clip
from backend.audio.validator import validate_clip
from backend.audio.recorder import save_clip, RecorderError, delete_clip, list_clips
from backend.audio.cleaner import get_or_clean_reference
from backend.audio.importer import import_recording

from backend.core.settings import DATA_DIR

from backend.services.project_service import create_project, get_project, get_reference_clip
from backend.services.inference_service import generate_speech

router = APIRouter(prefix="/projects", tags=["projects"])

# Registry: job_id → threading.Event, for cooperative cancellation of training jobs.
# Kept at module level so the cancel endpoint in api/jobs.py can reach it via
# `from backend.api.projects import _cancel_training_job`.
_cancel_events: dict[str, threading.Event] = {}


def _cancel_training_job(job_id: str) -> None:
    """Flip the cancel event for a training job, if one is registered."""
    event = _cancel_events.get(job_id)
    if event is not None:
        event.set()

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    language: str = "en"
    # When True, use the fine-tuned Voice Profile checkpoint instead of a
    # reference clip. Requires a completed training job for this project.
    profile: bool = False
    # Synthesis-targeted cleaning is OFF by default. Counter-intuitively,
    # XTTS v2's speaker encoder clones better from a natural processed clip
    # than from one that's been high-pass-filtered, denoised, and RMS-normalized.
    # Enable only for genuinely poor inputs (heavy hum, big level swings).
    clean_reference: bool = False
    # ── Pace / delivery tuning ────────────────────────────────────────────
    # XTTS doesn't copy speaking rate from the reference — rhythm is generated
    # each call. These let the user dial it in. "Too rushed / too robotic" is
    # usually fixed with speed≈0.92 + temperature≈0.8.
    speed: float = Field(1.0, ge=0.5, le=2.0)
    temperature: float = Field(0.75, ge=0.1, le=1.0)
    length_penalty: float = Field(1.0, gt=0.0, le=10.0)
    repetition_penalty: float = Field(5.0, ge=1.0, le=15.0)

@router.post("", status_code=201)
def post_create_project(req: CreateProjectRequest):
    return create_project(name=req.name)


@router.get('/{project_id}')
def get_project_by_id(project_id: str):
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=(f"No Project detected with this {project_id}"))
    return project

@router.post('/{project_id}/clips', status_code=201)
async def upload_clips(project_id: str, file: UploadFile = File(...)):
    #checking if the project even exists or not
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"No Project found with {project_id}")

    # reading the bytes from the upload
    audio_bytes = await file.read()

    #save the bytes to disk
    try:
        clip = save_clip(project_id, audio_bytes, filename=file.filename)
    except RecorderError as e:
        raise HTTPException(status_code=422, detail=str(e))

    
    #validating the clip
    validation = validate_clip(clip['path'])

    #return the results
    return {
        'clip_id': clip['clip_id'],
        "duration_s": clip['duration_s'],
        'sample_rate': clip['sample_rate'],
        "valid": validation.valid,
        "errors": validation.errors,
        "warning":validation.warnings,
    }

@router.delete("/{project_id}/clips/{clip_id}", status_code=204)
def remove_clip(project_id: str, clip_id: str):
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")


    deleted = delete_clip(project_id, clip_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No clip found with id: {clip_id}")


@router.get("/{project_id}/clips")
def get_all_clips(project_id: str):
    project = get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")
    
    return list_clips(project_id)

@router.post("/{project_id}/preprocess",status_code=202)
def start_preprocess(project_id: str, background_tasks: BackgroundTasks): 

    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    clips = list_clips(project_id)
    if not clips:
        raise HTTPException(status_code=422, detail="No clips to preprocess. Uplaod clips first")

    #creating a job
    job = job_manager.create_job("preprocess")


    #Schedule the actual work to run after this response is sent
    background_tasks.add_task(_run_preprocess, project_id, clips, job.id)


    return {"job_id": job.id, "status": "started", "clip_count": len(clips)}
    

def _run_preprocess(project_id: str, clips: list, job_id: str):
    """Background worker - this will run after HTTP response is sent"""
    try:
        job_manager.start_job(job_id)
        total = len(clips)

        for i, clip in enumerate(clips):
            job_manager.update_progress(
                job_id,
                progress=int((i / total) * 100),
                message=f"Processing clips {i+1} of {total}..."
            )
            preprocess_clip(
                project_id=project_id,
                clip_id=clip["clip_id"],
                input_path=clip["path"],
            )

        job_manager.complete_job(job_id, result={"processed": total})

    except Exception as e:
        job_manager.fail_job(job_id, error=str(e))



@router.post('/{project_id}/synthesize')
def synthesize(project_id: str, req: SynthesizeRequest):
    """
    Generate speech for the given text using the project's voice.

    Two modes controlled by `profile`:
      - profile=False (default): Quick Clone — use a reference clip for
        zero-shot voice cloning. Works immediately after recording.
      - profile=True: Voice Profile — use the fine-tuned checkpoint for
        higher-quality synthesis. Requires a completed training run.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    output_path = str(DATA_DIR / "projects" / project_id / "exports" / f"{uuid4()}.wav")

    # Build synthesis tuning params from the request.
    from backend.services.inference_service import SynthesisParams
    params = SynthesisParams(
        speed=req.speed,
        temperature=req.temperature,
        length_penalty=req.length_penalty,
        repetition_penalty=req.repetition_penalty,
    )

    if req.profile:
        # ── Voice Profile path: synthesize from fine-tuned checkpoint ──
        from backend.services.inference_service import generate_speech_from_checkpoint
        from backend.pipelines.training import _find_checkpoints

        checkpoints_dir = DATA_DIR / "projects" / project_id / "checkpoints"
        best, last = _find_checkpoints(checkpoints_dir)
        checkpoint = best or last

        if checkpoint is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No trained voice profile found. "
                    "Train a Voice Profile first, then try again."
                ),
            )

        # The config.json lives in the same run directory as the checkpoint.
        config_path = checkpoint.parent / "config.json"
        if not config_path.exists():
            raise HTTPException(
                status_code=422,
                detail="Voice profile is incomplete — config.json is missing.",
            )

        # We still need a reference clip for speaker conditioning even with
        # the fine-tuned model. The profile captures the voice's style;
        # the reference clip anchors the speaker identity.
        reference = get_reference_clip(project_id)
        if reference is None:
            raise HTTPException(
                status_code=422,
                detail="No processed clips found. Upload and preprocess a recording first.",
            )

        try:
            result_path = generate_speech_from_checkpoint(
                text=req.text,
                output_path=output_path,
                checkpoint_path=str(checkpoint),
                config_path=str(config_path),
                speaker_wav=reference,
                language=req.language,
                params=params,
            )
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {
            "output": result_path,
            "mode": "profile",
            "checkpoint": str(checkpoint),
            "language": req.language,
        }

    # ── Quick Clone path: reference-clip voice cloning ──────────────
    reference = get_reference_clip(project_id)
    if reference is None:
        raise HTTPException(
            status_code=422,
            detail="No processed clips found. Upload and preprocess a recording first.",
        )

    if req.clean_reference:
        try:
            reference_for_synth = get_or_clean_reference(project_id, reference)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        reference_for_synth = reference

    try:
        result_path = generate_speech(
            text=req.text,
            output_path=output_path,
            speaker_wav=reference_for_synth,
            language=req.language,
            params=params,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "output": result_path,
        "mode": "quick_clone",
        "reference_clip": reference_for_synth,
        "language": req.language,
        "cleaned_reference": req.clean_reference,
    }


@router.get("/{project_id}/preview/{clip_id}")
def preview_clip(project_id: str, clip_id: str):
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    exports_dir = Path(DATA_DIR / "projects" / project_id / "exports")
    clip_path = exports_dir / f"{clip_id}.wav"

    if not clip_path.exists():
        raise HTTPException(status_code=404, detail=f"No exported clip found with id: {clip_id}")

    return FileResponse(
        path=str(clip_path),
        media_type="audio/wav",
        filename=f"{clip_id}.wav",
    )



@router.post("/{project_id}/import", status_code=202)
async def import_clip_source(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a long audio or video file (podcast, interview, monologue, etc.).

    The file is split into individual speech segments which become regular
    clips on the project. Useful for:
      - Quick Clone: gives the user many candidate references to pick from
      - Voice Profile: provides the ~30 short clips needed for fine-tuning

    Returns a job_id; poll /jobs/{id} for completion. The job's `result`
    field carries the import summary on success.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    # Save the upload to a temp file. We don't put it in raw/ — that's for
    # individual clips, not source material. Lives in data/temp/ until import
    # completes, then importer.py cleans it up after extraction.
    temp_dir = Path(DATA_DIR) / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Preserve extension so ffmpeg can detect format
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    temp_path = temp_dir / f"upload_{uuid4()}{suffix}"

    # Stream to disk in chunks — long uploads (mp4, etc.) shouldn't be
    # buffered into memory all at once.
    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Couldn't save the upload: {e}")

    # Schedule background work — importing a 10-min file takes 30–60s.
    job = job_manager.create_job("import")
    background_tasks.add_task(_run_import, project_id, str(temp_path), job.id, file.filename or "upload")

    return {
        "job_id": job.id,
        "status": "started",
        "filename": file.filename,
    }


def _run_import(project_id: str, source_path: str, job_id: str, original_filename: str):
    """Background worker for /import."""
    try:
        job_manager.start_job(job_id)
        job_manager.update_progress(
            job_id, progress=10,
            message="Pulling out the audio...",
        )

        result = import_recording(project_id, source_path)

        if not result.success:
            job_manager.fail_job(
                job_id,
                error=result.error or "Import failed for an unknown reason.",
            )
            return

        job_manager.update_progress(
            job_id, progress=100,
            message=f"Imported {result.segments_kept} clips from {original_filename}.",
        )
        job_manager.complete_job(
            job_id,
            result={
                "source_filename": result.source_filename,
                "source_duration_s": result.source_duration_s,
                "segments_found": result.segments_found,
                "segments_kept": result.segments_kept,
                "clip_ids": result.clip_ids,
            },
        )
    except Exception as e:
        job_manager.fail_job(job_id, error=str(e))
    finally:
        # Clean up the upload regardless of outcome
        try:
            Path(source_path).unlink(missing_ok=True)
        except Exception:
            pass



# ── Dataset builder (M2 prep) ─────────────────────────────────────

class BuildDatasetRequest(BaseModel):
    """
    Optional inputs for the dataset builder.
    All fields are optional — sensible defaults handle the typical case.
    """
    language: str = "en"
    eval_fraction: float = 0.05      # 5% held out for validation
    transcripts: dict[str, str] | None = None  # clip_id → manual transcript


@router.post("/{project_id}/dataset", status_code=202)
def build_project_dataset(
    project_id: str,
    background_tasks: BackgroundTasks,
    req: BuildDatasetRequest = BuildDatasetRequest(),
):
    """
    Build an XTTS-ready dataset (LJSpeech format) from this project's
    processed clips. Auto-transcribes anything without a provided text.

    Returns a job_id you can poll via /jobs/{id}; on success the job's
    `result` field contains paths and counts.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    job = job_manager.create_job("dataset_build")
    background_tasks.add_task(
        _run_dataset_build,
        project_id, req.language, req.transcripts, req.eval_fraction, job.id,
    )

    return {"job_id": job.id, "status": "started"}


def _run_dataset_build(
    project_id: str,
    language: str,
    transcripts: dict[str, str] | None,
    eval_fraction: float,
    job_id: str,
):
    """Background worker for /dataset."""
    # Imported here (not at module top) to keep startup fast — the
    # transcriber initializes Whisper which is heavyweight.
    from backend.pipelines.dataset_builder import build_dataset

    try:
        job_manager.start_job(job_id)
        job_manager.update_progress(
            job_id, progress=5,
            message="Listening through your clips for transcripts...",
        )

        result = build_dataset(
            project_id=project_id,
            language=language,
            transcripts=transcripts,
            eval_fraction=eval_fraction,
        )

        if not result.success:
            job_manager.fail_job(job_id, error=result.error or "Dataset build failed.")
            return

        job_manager.update_progress(
            job_id, progress=100,
            message=(
                f"Dataset ready: {result.train_count} training clips, "
                f"{result.eval_count} for evaluation."
            ),
        )
        job_manager.complete_job(
            job_id,
            result={
                "dataset_dir": result.dataset_dir,
                "metadata_csv": result.metadata_csv,
                "manifest_json": result.manifest_json,
                "train_count": result.train_count,
                "eval_count": result.eval_count,
                "skipped_count": result.skipped_count,
                "skipped_reasons": result.skipped_reasons,
                "total_duration_s": result.total_duration_s,
                "language": result.language,
            },
        )
    except Exception as e:
        job_manager.fail_job(job_id, error=str(e))



# ── Training (M2) ─────────────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    language: str = "en"


@router.post("/{project_id}/train", status_code=202)
def start_training(
    project_id: str,
    background_tasks: BackgroundTasks,
    req: StartTrainingRequest = StartTrainingRequest(),
):
    """
    Start a Voice Profile fine-tuning job for this project.

    Prerequisites (checked before scheduling):
      - Project exists.
      - A dataset has been built (dataset/metadata.csv present).
      - Hardware passes the training gate (decided by training_config.decide_preset).

    Returns a job_id. Poll GET /jobs/{id} for progress, ETA, validation
    sample paths, and the final result. POST /jobs/{id}/cancel to stop.
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    # Quick prerequisite check before spinning up a job.
    from pathlib import Path as _Path
    metadata_csv = _Path(DATA_DIR) / "projects" / project_id / "dataset" / "metadata.csv"
    if not metadata_csv.exists():
        raise HTTPException(
            status_code=422,
            detail=(
                "No training dataset found for this project. "
                "Build the dataset first via POST /projects/{id}/dataset."
            ),
        )

    # Hardware gate — refuse CPU-only early so the user doesn't wait for
    # a job that will immediately fail inside run_training.
    from backend.pipelines.training_config import decide_preset
    decision = decide_preset(project_id=project_id)
    if not decision.can_train:
        raise HTTPException(
            status_code=422,
            detail=decision.refusal_reason or "This machine can't run Voice Profile training.",
        )

    job = job_manager.create_job("training")
    cancel_event = threading.Event()
    _cancel_events[job.id] = cancel_event

    background_tasks.add_task(
        _run_training_bg,
        project_id, req.language, job.id, cancel_event,
    )

    return {
        "job_id": job.id,
        "status": "started",
        "summary": decision.friendly_summary,
    }


def _run_training_bg(
    project_id: str,
    language: str,
    job_id: str,
    cancel_event: threading.Event,
) -> None:
    """Background worker for POST /train."""
    from backend.pipelines.training import run_training
    import dataclasses

    try:
        job_manager.start_job(job_id)
        job_manager.update_progress(job_id, 0, "Starting voice profile training...")

        result = run_training(
            project_id=project_id,
            job_id=job_id,
            language=language,
            cancel_event=cancel_event,
        )

        if result.success:
            job_manager.complete_job(
                job_id,
                result=dataclasses.asdict(result),
            )
        else:
            job_manager.fail_job(
                job_id,
                error=result.error or "Training failed for an unknown reason.",
            )
    except Exception as e:
        job_manager.fail_job(job_id, error=str(e))
    finally:
        # Remove the cancel event from the registry regardless of outcome.
        _cancel_events.pop(job_id, None)


@router.get("/{project_id}/training-plan")
def get_training_plan(project_id: str):
    """
    What would training look like on this machine?

    Returns either:
      - A plan + friendly summary the UI can show in a disclosure modal
      - A refusal with a friendly reason and suggested action

    The endpoint takes a project_id even though the plan doesn't depend on
    the project today — keeps the URL shape consistent with the future
    POST /train endpoint, and gives us room to factor in dataset size later
    (more clips → longer ETA).
    """
    project = get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"No project found with id: {project_id}")

    # Imported here so app startup doesn't pay the import cost when training
    # isn't being used.
    from backend.pipelines.training_config import decide_preset
    from dataclasses import asdict
    import json

    # Dataset-aware ETA: more clips → more micro-batches per epoch → longer
    # wall clock. We look for the manifest first (most accurate); if there's
    # no built dataset yet, fall back to counting processed clips so the
    # preview the user sees during recording is still close to reality.
    project_dir = Path(DATA_DIR) / "projects" / project_id
    manifest_path = project_dir / "dataset" / "manifest.json"

    train_clip_count: int | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            train_clip_count = int(manifest.get("train_count", 0)) or None
        except (ValueError, OSError):
            train_clip_count = None

    if train_clip_count is None:
        processed = project_dir / "processed"
        if processed.exists():
            train_clip_count = len(list(processed.glob("*.wav"))) or None

    decision = decide_preset(
        train_clip_count=train_clip_count,
        project_id=project_id,
    )
    return {
        "can_train": decision.can_train,
        "summary": decision.friendly_summary,
        "refusal_reason": decision.refusal_reason,
        "suggested_action": decision.suggested_action,
        "detected_hardware": decision.detected_hardware,
        "data_locations": decision.data_locations,
        "data_summary": decision.data_summary,
        # The plan dict is verbose. UI shouldn't show it; it's there for
        # debug + so the trainer can call this same endpoint server-side
        # when it kicks off.
        "plan": asdict(decision.plan) if decision.plan else None,
    }
