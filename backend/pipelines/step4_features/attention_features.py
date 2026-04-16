import numpy as np


def _safe_var(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanvar(arr))


def _safe_std(arr):
    arr = np.array(arr, dtype=np.float32)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanstd(arr))


def gaze_stability(window):
    """
    Variance of true pupil direction over the window (rx, ry).
    """
    rx = [
        s["eye_gaze"]["rx"]
        for s in window
        if s.get("eye_gaze") is not None
        and s["eye_gaze"].get("rx") is not None
    ]

    ry = [
        s["eye_gaze"]["ry"]
        for s in window
        if s.get("eye_gaze") is not None
        and s["eye_gaze"].get("ry") is not None
    ]

    return {
        "eye_gaze_var_rx": _safe_var(rx),
        "eye_gaze_var_ry": _safe_var(ry),
    }


def head_motion(window):
    """
    Std of yaw/pitch over the window, using head_gaze.
    """
    yaw = [
        s["head_gaze"]["yaw"]
        for s in window
        if s.get("head_gaze") is not None
        and s["head_gaze"].get("yaw") is not None
    ]

    pitch = [
        s["head_gaze"]["pitch"]
        for s in window
        if s.get("head_gaze") is not None
        and s["head_gaze"].get("pitch") is not None
    ]

    return {
        "head_yaw_std":   _safe_std(yaw),
        "head_pitch_std": _safe_std(pitch),
    }
