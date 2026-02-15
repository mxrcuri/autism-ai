from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --- Data Models (Shared) ---
class Frame(BaseModel):
    valid: bool
    pose: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    gaze: Optional[Dict[str, Any]] = None

class Payload(BaseModel):
    fps: float
    sequence: List[Frame]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Important: Import the router AFTER defining Payload if you use relative imports
from .routes.infer import router as infer_router
app.include_router(infer_router)

@app.get("/")
def root():
    return {"status": "backend running"}