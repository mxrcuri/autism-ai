from typing import Dict

from .video_loader import load_and_resample_video
from .validity import (
    build_validity_mask_with_stats,
    evaluate_video_quality
)

def preprocess_video(video_path: str) -> Dict:
    frames, timestamps = load_and_resample_video(video_path)

    validity = build_validity_mask_with_stats(frames)
    valid_mask = validity["valid_mask"]
    stats = validity["stats"]

    decision = evaluate_video_quality(valid_mask, stats)

    return {
        "frames": frames,
        "timestamps": timestamps,
        "valid_mask": valid_mask,
        "usable": decision["usable"],
        "reason": decision["reason"],
        "stats": stats,
        "metadata": {
            "fps": 25,
            "num_frames": len(frames)
        }
    }

