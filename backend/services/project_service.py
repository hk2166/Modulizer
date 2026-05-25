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


def get_reference_clip(project_id: str) -> str | None:
    """
    Return the path to the project's preferred synthesis reference clip.

    Priority:
      1. The golden clip recorded in dataset/manifest.json — picked by
         select_golden_reference based on duration fit + pitch stability,
         filtered to clips that survived dataset validation.
      2. Lexicographic fallback to processed/ — used by the Quick Clone
         path, before any dataset has been built.
      3. None when there's nothing usable.

    Always returns a path that exists on disk; a stale id in the manifest
    falls through to the lexicographic fallback rather than returning a
    path the caller will then fail to open.
    """
    project_dir = DATA_DIR / "projects" / project_id
    processed_dir = project_dir / "processed"

    if not processed_dir.exists():
        return None

    # 1. Manifest's golden clip wins when present and on disk.
    manifest_path = project_dir / "dataset" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            golden_id = manifest.get("golden_clip_id")
            if golden_id:
                candidate = processed_dir / f"{golden_id}.wav"
                if candidate.exists():
                    return str(candidate.resolve())
                logger.warning(
                    f"get_reference_clip: manifest's golden_clip_id "
                    f"{golden_id!r} no longer on disk — falling back."
                )
        except (ValueError, OSError) as e:
            logger.warning(f"get_reference_clip: bad manifest ({e}) — falling back.")

    # 2. Lexicographic fallback (Quick Clone path).
    clips = sorted(processed_dir.glob("*.wav"))
    if not clips:
        return None
    return str(clips[0].resolve())
