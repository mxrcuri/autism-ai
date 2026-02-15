from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 1. Shared Data Models
# Note: In a larger app, move these to a 'backend/app/models.py' file
class Frame(BaseModel):
    valid: bool
    pose: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    gaze: Optional[Dict[str, Any]] = None

class Payload(BaseModel):
    fps: float
    sequence: List[Frame]

# 2. App Initialization
app = FastAPI(title="Autism AI Backend")

# 3. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Router Imports
# Using relative imports. Ensure you are running from the /backend/ directory
from .routes.infer import router as infer_router
from .routes.assessments import router as assessment_router

# 5. Route Registration
app.include_router(infer_router, tags=["Inference"])
app.include_router(assessment_router, prefix="/assessments", tags=["Assessments"])

@app.get("/")
async def root():
    return {
        "status": "backend running",
        "version": "1.0.0",
        "endpoints": ["/infer", "/assessments/submit-quiz", "/assessments/results/{id}"]
    }