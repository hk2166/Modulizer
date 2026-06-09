from fastapi import APIRouter, HTTPException

from backend.jobs.instance import job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found with id: {job_id}")

    return {
        "job_id": job.id,
        "type": job.type,
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "eta_seconds": job.eta_seconds,
        "validation_sample_path": job.validation_sample_path,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at.isoformat(),
    }


@router.post("/{job_id}/cancel", status_code=202)
def cancel_job(job_id: str):
    """
    Request cooperative cancellation of a running job.

    For training jobs, this sets a threading.Event that ProgressBridge
    checks every training step. The training loop saves a checkpoint and
    exits cleanly; the job resolves as CANCELLED. For other job types the
    status is marked CANCELLED immediately (no mid-task stop mechanism).

    Returns 202 Accepted — the job may still be running for a few seconds
    while it winds down. Poll GET /jobs/{id} until status is CANCELLED or
    COMPLETED.
    """
    from backend.api.projects import _cancel_training_job

    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job found with id: {job_id}")

    # Flip the event for training jobs so the loop exits at the next step.
    _cancel_training_job(job_id)

    # Mark status immediately so the UI sees the intent even before the
    # loop checkpoint-and-exits.
    cancelled = job_manager.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is already {job.status} and cannot be cancelled.",
        )

    return {"job_id": job_id, "status": "cancelling"}
