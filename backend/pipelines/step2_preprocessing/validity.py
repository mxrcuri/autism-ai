from typing import List, Dict
import numpy as np

from .face_filter import detect_faces
from .quality_checks import is_frame_too_dark, is_frame_blurry


def build_validity_mask_with_stats(frames: List[np.ndarray]) -> Dict:
    face_counts = detect_faces(frames)

    valid_mask = []

    stats = {
        "total_frames": len(frames),
        "valid_frames": 0,
        "no_face_frames": 0,
        "multi_face_frames": 0,
        "dark_frames": 0,
        "blurry_frames": 0,
    }

    for frame, n_faces in zip(frames, face_counts):
        valid = True  # RESET every frame

        # Face rules
        if n_faces == 0:
            stats["no_face_frames"] += 1
            valid = False
        elif n_faces > 1:
            stats["multi_face_frames"] += 1
            valid = False

        # Lighting rule
        if is_frame_too_dark(frame):
            stats["dark_frames"] += 1
            valid = False

        # Blur rule (secondary)
        if is_frame_blurry(frame) and n_faces != 1:
            stats["blurry_frames"] += 1
            valid = False

        if valid:
            stats["valid_frames"] += 1

        valid_mask.append(valid)
        

    return {
        "valid_mask": valid_mask,
        "stats": stats
    }


def evaluate_video_quality(
    valid_mask: List[bool],
    stats: Dict,
    min_valid_frames: int = 100,
    min_valid_ratio: float = 0.7,
    max_invalid_gap_sec: float = 3.0,
    fps: int = 25
) -> Dict:

    # Guard: no frames at all
    if stats["total_frames"] == 0:
        return {"usable": False, "reason": "NO_FRAMES_DECODED"}

    # Rule 1: too few valid frames (YOUR REQUEST)
    if stats["valid_frames"] < min_valid_frames:
        return {
            "usable": False,
            "reason": "TOO_FEW_VALID_FRAMES",
        }

    # Rule 2: valid ratio
    valid_ratio = stats["valid_frames"] / stats["total_frames"]
    if valid_ratio < min_valid_ratio:
        return {
            "usable": False,
            "reason": "LOW_VALID_RATIO",
        }

    # Rule 3: long continuous invalid gap
    max_gap = 0
    current_gap = 0
    for v in valid_mask:
        if not v:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 0

    if max_gap / fps > max_invalid_gap_sec:
        return {
            "usable": False,
            "reason": "LONG_INVALID_GAP",
        }

    return {"usable": True, "reason": "OK"}

