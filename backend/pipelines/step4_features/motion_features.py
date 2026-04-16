import numpy as np

UPPER_BODY_JOINTS = [
    "wrist_left", "wrist_right",
    "elbow_left", "elbow_right",
    "sholder_left", "sholder_right",
    "hand_left", "hand_right",
    "sholder_center",
]

def upper_body_motion_energy(window):
    """
    Computes aggregate motion energy from upper-body joints.
    Skips frames where skeleton is None (pose detector lagged behind capture loop).
    Returns zeros if fewer than 2 valid skeleton frames exist.
    """
    # Filter to frames that actually have skeleton data
    pose_frames = [s for s in window if s.get("skeleton") is not None]

    if len(pose_frames) < 2:
        return {"motion_mean": 0.0, "motion_std": 0.0, "motion_max": 0.0}

    speeds = []

    for joint in UPPER_BODY_JOINTS:
        try:
            coords = np.array([s["skeleton"][joint] for s in pose_frames])
            velocity = np.diff(coords, axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            speeds.append(speed)
        except (KeyError, TypeError):
            # Joint missing from some frames — skip this joint
            continue

    if not speeds:
        return {"motion_mean": 0.0, "motion_std": 0.0, "motion_max": 0.0}

    speeds = np.array(speeds)        # (num_joints, T-1)
    mean_speed = np.mean(speeds, axis=0)

    return {
        "motion_mean": float(np.mean(mean_speed)),
        "motion_std":  float(np.std(mean_speed)),
        "motion_max":  float(np.max(mean_speed)),
    }
