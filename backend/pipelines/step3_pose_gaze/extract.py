from .pose_extractor import extract_pose_sequence
from .face_mesh import extract_head_pose_sequence
from .gaze_estimator import estimate_gaze_sequence

def run_step3(step2_output):
    frames = step2_output["frames"]
    valid_mask = step2_output["valid_mask"]
    timestamps = step2_output["timestamps"]

    pose_seq = extract_pose_sequence(frames, valid_mask)
    head_seq = extract_head_pose_sequence(frames, valid_mask)
    gaze_seq = estimate_gaze_sequence(head_seq)

    sequence = []

    for t, pose, head, gaze, valid in zip(
        timestamps, pose_seq, head_seq, gaze_seq, valid_mask
    ):
        sequence.append({
            "t": t,
            "pose": pose,
            "head": head,
            "gaze": gaze,
            "valid": valid
        })

    return sequence

