import numpy as np

UPPER_BODY_JOINTS = [
    "left_wrist", "right_wrist",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
]

def upper_body_motion_energy(window):
    """
    Computes aggregate motion energy from upper-body joints.
    """
    speeds = []

    for joint in UPPER_BODY_JOINTS:
        coords = np.array([s["pose"][joint] for s in window])
        velocity = np.diff(coords, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        speeds.append(speed)

    speeds = np.array(speeds)  # shape: (num_joints, T-1)
    mean_speed = np.mean(speeds, axis=0)

    return {
        "motion_mean": float(np.mean(mean_speed)),
        "motion_std": float(np.std(mean_speed)),
        "motion_max": float(np.max(mean_speed)),
    }

