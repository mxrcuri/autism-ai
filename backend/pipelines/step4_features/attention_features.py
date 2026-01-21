import numpy as np


def _safe_var(arr):
    """
    Compute variance ignoring None / NaN.
    Returns 0.0 if all values are invalid.
    """
    arr = np.array(arr, dtype=np.float32)

    if arr.size == 0:
        return 0.0

    if np.all(np.isnan(arr)):
        return 0.0

    return float(np.nanvar(arr))


def _safe_std(arr):
    arr = np.array(arr, dtype=np.float32)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return 0.0

    return float(np.nanstd(arr))


def gaze_stability(window):
    gx = [
        s["gaze"]["gx"]
        for s in window
        if s["gaze"]["gx"] is not None
    ]

    gy = [
        s["gaze"]["gy"]
        for s in window
        if s["gaze"]["gy"] is not None
    ]

    return {
        "gaze_var_x": _safe_var(gx),
        "gaze_var_y": _safe_var(gy),
    }


def head_motion(window):
    yaw = [
        s["head"]["yaw"]
        for s in window
        if s["head"]["yaw"] is not None
    ]

    pitch = [
        s["head"]["pitch"]
        for s in window
        if s["head"]["pitch"] is not None
    ]

    return {
        "yaw_std": _safe_std(yaw),
        "pitch_std": _safe_std(pitch),
    }

