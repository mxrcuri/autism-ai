import numpy as np

def arm_symmetry(window):
    """
    Measures left-right symmetry of wrist motion.
    """
    left = np.array([s["pose"]["left_wrist"] for s in window])
    right = np.array([s["pose"]["right_wrist"] for s in window])

    dist = np.linalg.norm(left - right, axis=1)

    return {
        "lr_mean_dist": float(np.mean(dist)),
        "lr_std_dist": float(np.std(dist)),
    }

