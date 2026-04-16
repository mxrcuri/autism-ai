import asyncio
import datetime as dt
from typing import Optional, Dict, Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..auth import get_current_user
from ..config import FEATURE_KEYS
from ..database import get_db
from ..model_loader import get_model
from ..models import FeatureData, InferenceResult, User, UserSession
from ..schemas import VideoPayload
from ml.inference.score_sequence import score_sequence
from pipelines.step4_features.extract import extract_features

router = APIRouter(prefix="/infer", tags=["Inference"])


def _summarize_feature_windows(feature_dicts) -> Dict[str, Any]:
    """
    Converts a list of window-level feature dicts into compact summary stats.
    We store summary stats, not raw frames.
    """
    summary = {}

    for key in FEATURE_KEYS:
        values = np.array([float(d.get(key, 0.0)) for d in feature_dicts], dtype=np.float32)

        summary[key] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
            "min": float(values.min()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0,
        }

    return summary


@router.post("/video")
async def infer_video(
    payload: VideoPayload,
    session_id: Optional[int] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        sequence_data = [f.dict() for f in payload.sequence]
        fps = payload.fps

        if not sequence_data:
            raise HTTPException(status_code=400, detail="Sequence is empty")

        # If session_id is provided, reuse that session.
        # Otherwise create a new session for this run.
        session_obj = None
        new_session_created = False

        if session_id is not None:
            result = await db.execute(
                select(UserSession).where(
                    UserSession.id == session_id,
                    UserSession.user_id == current_user.id,
                )
            )
            session_obj = result.scalars().first()
            if session_obj is None:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session_obj = UserSession(
                user_id=current_user.id,
                session_type="video",
                status="processing",
            )
            db.add(session_obj)
            await db.flush()  # assigns session_obj.id
            new_session_created = True

        # Minimum data check — need at least one full 2-second window
        min_frames = int(fps * 2)
        if len(sequence_data) < min_frames:
            session_obj.status = "pending"
            session_obj.updated_at = dt.datetime.utcnow()

            if new_session_created:
                await db.commit()

            return {
                "session_id": session_obj.id,
                "status": "waiting_for_more_data",
                "score": None,
                "meta": {
                    "frames_received": len(sequence_data),
                    "fps": fps,
                    "min_frames_needed": min_frames,
                },
            }

        # 1. Feature extraction
        feature_dicts = extract_features(sequence_data, fps=fps)

        if not feature_dicts:
            session_obj.status = "pending"
            session_obj.updated_at = dt.datetime.utcnow()
            await db.commit()

            return {
                "session_id": session_obj.id,
                "status": "waiting_for_more_data",
                "score": None,
                "meta": {
                    "frames_processed": len(sequence_data),
                    "windows_generated": 0,
                },
            }

        # 2. Convert list of dicts -> numpy array [num_windows, num_features]
        features_np = np.array(
            [[d.get(k, 0.0) for k in FEATURE_KEYS] for d in feature_dicts],
            dtype=np.float32,
        )

        # 3. Add batch dimension -> [1, num_windows, num_features]
        features_np = features_np[np.newaxis, :, :]

        # 4. Load model
        model, device = get_model()

        # 5. Score asynchronously so the event loop is not blocked
        result = await asyncio.to_thread(score_sequence, model, features_np, device)

        anomaly_score = float(result.get("confidence", 0.0))
        specific_flag = "potential_anomaly" if anomaly_score > 0.7 else "normal"

        # 6. Store compact feature summaries
        feature_summary = _summarize_feature_windows(feature_dicts)

        feature_row = FeatureData(
            session_id=session_obj.id,
            modality="video",
            features_json=feature_summary,
            metadata_json={
                "fps": fps,
                "frames_processed": len(sequence_data),
                "windows_generated": len(feature_dicts),
                "feature_keys": FEATURE_KEYS,
            },
        )
        db.add(feature_row)

        # 7. Store inference output
        inference_row = InferenceResult(
            session_id=session_obj.id,
            anomaly_score=anomaly_score,
            surprise_score=None,
            surprise_curve_json=None,
            flags_json={
                "specific_flag": specific_flag,
                "domain": "visual_attention",
                "threshold": 0.7,
                "mean_deviation": result.get("mean_deviation"),
                "window_scores": result.get("window_scores"),
            },
            model_name="TCN_VAE",
            model_version="v1",
        )
        db.add(inference_row)

        # 8. Update session status
        session_obj.status = "completed"
        session_obj.completed_at = dt.datetime.utcnow()
        session_obj.updated_at = dt.datetime.utcnow()

        await db.commit()

        return {
            "session_id": session_obj.id,
            "status": "success",
            "score": anomaly_score,
            "flag": specific_flag,
            "meta": {
                "fps_received": fps,
                "frames_processed": len(sequence_data),
                "windows_generated": len(feature_dicts),
                "mean_deviation": result.get("mean_deviation"),
                "window_scores": result.get("window_scores"),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        print(f"[ERROR] Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")