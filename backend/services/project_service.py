from pathlib import Path
from uuid import uuid4
from datetime import datetime
import json

from backend.core.logger import logger
from backend.core.settings import DATA_DIR

def create_project(name: str) -> dict:
    project_id = str(uuid4())
    project_dir = DATA_DIR / 'projects' / project_id

    #create all subdirectories
    for subdir in ['raw', 'processed', 'checkpoints', 'exports', 'metadata']: 
        (project_dir / subdir).mkdir(parents=True, exist_ok=True)

    #building metadata
    metadata = {
        "id": project_id,
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "status": 'created',
        'clip_count': 0,
    }

    #storing data in persistant storage
    metadata_path = project_dir / "metadata" / "project.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info(f"Project created: id={project_id}, name={name!r}")
    return metadata