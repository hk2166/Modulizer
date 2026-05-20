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

def get_project(project_id:str) -> dict:
    project_dir = DATA_DIR / 'projects' / project_id
    metadata_path = project_dir / 'metadata' / 'project.json'

    #if folder doen't exist return 404 and None
    if not metadata_path.exists():
        return None
    
    #load the saved metadata
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))

    #counting clips from the persistant disk
    raw_dir = project_dir / 'raw'
    clips = list(raw_dir.glob("*.wav")) if raw_dir.exists() else []
    metadata['clip_count'] = len(clips)

    #counting the validated clips(processedOnes)
    processed_dir = project_dir / "processed"
    processed = list(processed_dir.glob("*.wav")) if processed_dir.exists() else []
    metadata['validated_count'] = len(processed)

    return metadata