from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.services.project_service import create_project

router = APIRouter(prefix="/projects", tags=["projects"])

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

@router.post("", status_code=201)
def post_create_project(req: CreateProjectRequest):
    return create_project(name=req.name)
