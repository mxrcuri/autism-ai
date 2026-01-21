import cv2
import numpy as np
from typing import List, Tuple

def load_and_resample_video(
    video_path: str,
    target_fps: int = 25,
    max_width: int = 720
) -> Tuple[List[np.ndarray], List[float]]:

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(orig_fps / target_fps))

    frames = []
    timestamps = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            timestamps.append(frame_idx / orig_fps)

        frame_idx += 1

    cap.release()
    return frames, timestamps

