from backend.jobs.job import Job
from backend.jobs.job_status import JobStatus

from backend.core.logger import logger


class JobManager:
    def __init__(self):
        self.jobs = {}

    def create_job(self, job_type: str):
        job = Job(type=job_type)

        self.jobs[job.id] = job

        logger.info(f"Created job {job.id} ({job.type})")

        return job

    def start_job(self, job_id: str):
        job = self.jobs[job_id]

        job.status = JobStatus.RUNNING

        logger.info(f"Started job {job.id}")

    def complete_job(self, job_id: str, result=None):
        job = self.jobs[job_id]

        job.status = JobStatus.COMPLETED
        job.progress = 100
        job.result = result

        logger.info(f"Completed job {job.id}")

    def fail_job(self, job_id: str, error: str):
        job = self.jobs[job_id]

        job.status = JobStatus.FAILED
        job.error = error

        logger.error(f"Job failed {job.id}: {error}")

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str = "",
        eta_seconds: int | None = None,
        validation_sample_path: str | None = None,
    ):
        job = self.jobs[job_id]

        job.progress = progress
        job.message = message
        if eta_seconds is not None:
            job.eta_seconds = eta_seconds
        if validation_sample_path is not None:
            job.validation_sample_path = validation_sample_path

        eta_txt = f", eta={eta_seconds}s" if eta_seconds is not None else ""
        logger.info(
            f"Job {job.id} progress: {progress}% - {message}{eta_txt}"
        )

    def cancel_job(self, job_id: str) -> bool:
        """
        Mark a job as cancelled. Returns True if the job existed and
        was in a cancellable state, False otherwise.
        The actual training loop stoppage is done by flipping the
        cancel_event registered in the API layer.
        """
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False
        job.status = JobStatus.CANCELLED
        logger.info(f"Job {job.id} marked as cancelled")
        return True

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)