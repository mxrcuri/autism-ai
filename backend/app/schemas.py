from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Frame(BaseModel):
    valid: bool
    skeleton: Optional[Dict[str, Any]] = None
    head_gaze: Optional[Dict[str, Any]] = None
    eye_gaze: Optional[Dict[str, Any]] = None


class VideoPayload(BaseModel):
    fps: float
    sequence: List[Frame]


# class AudioPayload(BaseModel):
#     features: Dict[str, Any]  # already processed CHILDES features
# 
# 
# class QuestionnairePayload(BaseModel):
#     responses: Dict[str, Any]


class InferRequest(BaseModel):
    modality: str = Field(..., description="video | audio | questionnaire")
    data: Dict[str, Any]

class SessionCreateResponse(BaseModel):
    session_id: int
    status: str


class SessionStatusResponse(BaseModel):
    session_id: int
    status: str
    created_at: datetime

class InferenceResponse(BaseModel):
    session_id: int
    anomaly_score: float
    surprise_score: Optional[float] = None
    flags: Optional[Dict[str, Any]] = None

class TherapyPlanResponse(BaseModel):
    session_id: int
    plan: Dict[str, Any]
    generated_at: datetime

class FeatureResponse(BaseModel):
    session_id: int
    modality: str
    features: Dict[str, Any]