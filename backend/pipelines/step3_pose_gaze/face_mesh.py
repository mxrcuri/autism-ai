import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh

def extract_head_pose_sequence(frames, valid_mask):
    head_seq = []

    with mp_face.FaceMesh(static_image_mode=False) as face_model:
        for frame, valid in zip(frames, valid_mask):
            if not valid:
                head_seq.append(None)
                continue

            result = face_model.process(frame)
            if not result.multi_face_landmarks:
                head_seq.append(None)
                continue

            lm = result.multi_face_landmarks[0].landmark

            # Simple proxy angles (sufficient for behavior)
            left_eye = np.array([lm[33].x, lm[33].y])
            right_eye = np.array([lm[263].x, lm[263].y])
            nose = np.array([lm[1].x, lm[1].y])

            yaw = right_eye[0] - left_eye[0]
            pitch = nose[1] - (left_eye[1] + right_eye[1]) / 2
            roll = right_eye[1] - left_eye[1]

            head_seq.append({
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll)
            })

    return head_seq

