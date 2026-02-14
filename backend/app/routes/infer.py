from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

router = APIRouter()

class Frame(BaseModel):
    valid: bool
    pose: Optional[Dict[str, Any]]
    head: Optional[Dict[str, Any]]
    gaze: Optional[Dict[str, Any]]

class Payload(BaseModel):
    fps: float
    sequence: List[Frame]

@router.post("/infer")
async def infer(payload: Payload):
    print("Received frames:", len(payload.sequence))
    print("FPS:", payload.fps)

    # For now just return length
    return {
        "status": "received",
        "frames": len(payload.sequence),
        "fps": payload.fps
    }

