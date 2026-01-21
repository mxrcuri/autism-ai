from .windowing import sliding_windows
from .motion_features import upper_body_motion_energy
from .symmetry_features import arm_symmetry
from .attention_features import gaze_stability, head_motion


TASK_MAP = {
    "imitation": [1, 0, 0],
    "joint_attention": [0, 1, 0],
    "turn_taking": [0, 0, 1],
}


def extract_features(
    sequence,
    fps=25,
    window_sec=2,
    stride_sec=1,
    task_name=None
):
    window_size = window_sec * fps
    stride = stride_sec * fps

    windows = sliding_windows(sequence, window_size, stride)
    feature_vectors = []

    for w in windows:
        feats = {}

        # Core behavioral features
        feats.update(upper_body_motion_energy(w))
        feats.update(arm_symmetry(w))
        feats.update(gaze_stability(w))
        feats.update(head_motion(w))

        # Task context (optional but recommended)
        if task_name is not None:
            task_vec = TASK_MAP.get(task_name.lower())
            if task_vec is None:
                raise ValueError(f"Unknown task: {task_name}")

            feats["task_imitation"] = task_vec[0]
            feats["task_joint_attention"] = task_vec[1]
            feats["task_turn_taking"] = task_vec[2]

        feature_vectors.append(feats)

    return feature_vectors

