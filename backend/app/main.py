from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI()

# --------------------------------------------------
# CORS (Allow Next.js frontend)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Data Models (Match Frontend JSON Exactly)
# --------------------------------------------------

class Frame(BaseModel):
    valid: bool
    pose: Optional[Dict[str, Any]] = None
    head: Optional[Dict[str, Any]] = None
    gaze: Optional[Dict[str, Any]] = None

class Payload(BaseModel):
    fps: float
    sequence: List[Frame]

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "backend running"}

@app.post("/infer")
async def infer(payload: Payload):
    print("Received frames:", len(payload.sequence))
    print("FPS:", payload.fps)

    # For now just echo back stats
    return {
        "status": "received",
        "frames": len(payload.sequence),
        "fps": payload.fps
    }

