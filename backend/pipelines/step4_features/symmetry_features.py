import numpy as np

def arm_symmetry(window):
    """
    Measures left-right symmetry of wrist and hand motion.
    Skips frames where skeleton is None.
    Returns zeros if fewer than 1 valid pose frame exists.
    """
    pose_frames = [s for s in window if s.get("skeleton") is not None]

    if len(pose_frames) < 1:
        return {
            "wrist_lr_mean_dist": 0.0, "wrist_lr_std_dist": 0.0,
            "hand_lr_mean_dist": 0.0,  "hand_lr_std_dist": 0.0
        }

    try:
        w_left  = np.array([s["skeleton"]["wrist_left"]  for s in pose_frames])
        w_right = np.array([s["skeleton"]["wrist_right"] for s in pose_frames])
        
        h_left  = np.array([s["skeleton"]["hand_left"]   for s in pose_frames])
        h_right = np.array([s["skeleton"]["hand_right"]  for s in pose_frames])
    except (KeyError, TypeError):
        return {
            "wrist_lr_mean_dist": 0.0, "wrist_lr_std_dist": 0.0,
            "hand_lr_mean_dist": 0.0,  "hand_lr_std_dist": 0.0
        }

    w_dist = np.linalg.norm(w_left - w_right, axis=1)
    h_dist = np.linalg.norm(h_left - h_right, axis=1)

    return {
        "wrist_lr_mean_dist": float(np.mean(w_dist)),
        "wrist_lr_std_dist":  float(np.std(w_dist)),
        "hand_lr_mean_dist":  float(np.mean(h_dist)),
        "hand_lr_std_dist":   float(np.std(h_dist)),
    }
