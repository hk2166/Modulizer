from fastapi import APIRouter, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field
from backend.jobs.instance import job_manager
from backend.audio.preprocessor import preprocess_clip
from backend.audio.validator import validate_clip
from backend.audio.recorder import save_clip, RecorderError, delete_clip, list_clips

from backend.services.project_service import create_project, get_project

router = APIRouter(prefix="/projects", tags=["projects"])

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

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