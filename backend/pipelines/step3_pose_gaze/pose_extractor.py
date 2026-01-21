import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

POSE_JOINTS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip"
]

def extract_pose_sequence(frames, valid_mask):
    pose_seq = []

    with mp_pose.Pose(static_image_mode=False) as pose_model:
        for frame, valid in zip(frames, valid_mask):
            if not valid:
                pose_seq.append(None)
                continue

            result = pose_model.process(frame)
            if not result.pose_landmarks:
                pose_seq.append(None)
                continue

            lm = result.pose_landmarks.landmark

            joints = {}
            for name in POSE_JOINTS:
                idx = mp_pose.PoseLandmark[name.upper()].value
                joints[name] = np.array([lm[idx].x, lm[idx].y, lm[idx].z])

            # Torso center & scale
            torso = (joints["left_hip"] + joints["right_hip"]) / 2
            shoulder_width = np.linalg.norm(
                joints["left_shoulder"] - joints["right_shoulder"]
            ) + 1e-6

            for k in joints:
                joints[k] = (joints[k] - torso) / shoulder_width

            pose_seq.append(joints)

    return pose_seq

