from fastapi import APIRouter, HTTPException
# Import Payload from your main file (adjust path if needed)
from ..main import Payload 

# Step 4 & 5 Imports (The ML Pipeline)
from pipelines.step4_features.windowing import sliding_windows
from pipelines.step4_features.extract import extract_features
from pipelines.step5_model.score import score_sequence

router = APIRouter()

@router.post("/infer")
async def infer_endpoint(payload: Payload):
    try:
        # 1. Extract data from Payload
        # Pydantic models convert to dicts/objects automatically
        sequence_data = [f.dict() for f in payload.sequence]
        fps = payload.fps

        if not sequence_data:
            raise HTTPException(status_code=400, detail="Sequence is empty")

        # 2. Windowing (The new entry point)
        # WINDOW_STRIDE is usually 1.0 or 0.5 seconds
        windows = sliding_windows(sequence_data, fps=int(fps), stride_seconds=1.0)

        if not windows:
            return {"status": "waiting_for_more_data", "score": None}

        # 3. Feature Extraction
        # This calls motion_features, symmetry_features, etc.
        features = extract_features(windows)

        # 4. Model Scoring (TCN / Classifier)
        score = score_sequence(features)

        return {
            "status": "success",
            "score": float(score),
            "meta": {
                "frames_processed": len(sequence_data),
                "windows_generated": len(windows)
            }
        }

    except Exception as e:
        print(f"[ERROR] Pipeline Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))