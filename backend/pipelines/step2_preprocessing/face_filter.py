import mediapipe as mp
import numpy as np
from typing import List

mp_face = mp.solutions.face_detection

def detect_faces(frames: List[np.ndarray]) -> List[int]:
    face_counts = []

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.3) as detector:
        for frame in frames:
            results = detector.process(frame)
            if results.detections:
                face_counts.append(len(results.detections))
            else:
                face_counts.append(0)

    return face_counts

