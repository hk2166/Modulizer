from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from backend.jobs.job_status import JobStatus


@dataclass
class Job:
    type: str

    id: str = field(default_factory=lambda: str(uuid4()))
    status: JobStatus = JobStatus.PENDING

    created_at: datetime = field(default_factory=datetime.utcnow)

    progress: int = 0
    message: str = ""

    result: dict | None = None
    error: str | None = None