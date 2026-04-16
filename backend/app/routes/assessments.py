from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Dict, Any

from ..database import get_db
from ..auth import get_current_user
from ..models import User, UserSession, FeatureData, InferenceResult
from pydantic import BaseModel

router = APIRouter()


# =========================
# SCHEMAS
# =========================

class QuizAnswer(BaseModel):
    question_id: str
    selected_option: str
    points: int


class QuizSubmission(BaseModel):
    session_id: int
    quiz_type: str
    answers: List[QuizAnswer]


# =========================
# SUBMIT QUIZ
# =========================

@router.post("/submit-quiz")
async def submit_quiz(
    payload: QuizSubmission,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Validate session
    result = await db.execute(
        select(UserSession).where(
            UserSession.id == payload.session_id,
            UserSession.user_id == user.id
        )
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Compute score
    total_score = sum(a.points for a in payload.answers)

    # 3. Store as FeatureData (modality = questionnaire)
    feature_row = FeatureData(
        session_id=session.id,
        modality="questionnaire",
        features_json={
            "quiz_type": payload.quiz_type,
            "total_score": total_score,
            "answers": [a.dict() for a in payload.answers]
        }
    )

    db.add(feature_row)
    await db.commit()

    return {
        "status": "success",
        "session_id": session.id,
        "total_score": total_score
    }


# =========================
# COMBINED RESULT
# =========================

@router.get("/results/{session_id}")
async def get_comprehensive_result(
    session_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    # 1. Validate session
    result = await db.execute(
        select(UserSession).where(
            UserSession.id == session_id,
            UserSession.user_id == user.id
        )
    )
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # 2. Get inference
    result = await db.execute(
        select(InferenceResult).where(InferenceResult.session_id == session_id)
    )
    inference = result.scalars().first()

    # 3. Get questionnaire
    result = await db.execute(
        select(FeatureData).where(
            FeatureData.session_id == session_id,
            FeatureData.modality == "questionnaire"
        )
    )
    questionnaire = result.scalars().first()

    # 4. Combine (simple fusion for now)
    anomaly_score = inference.anomaly_score if inference else 0.0
    quiz_score = questionnaire.features_json.get("total_score", 0) if questionnaire else 0

    # Simple heuristic fusion
    combined_score = 0.7 * anomaly_score + 0.3 * (quiz_score / 100)

    risk_level = "Low"
    if combined_score > 0.7:
        risk_level = "High"
    elif combined_score > 0.4:
        risk_level = "Medium"

    return {
        "session_id": session_id,
        "status": "complete",
        "results": {
            "anomaly_score": anomaly_score,
            "questionnaire_score": quiz_score,
            "combined_score": combined_score,
            "risk_level": risk_level
        }
    }