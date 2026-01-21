import cv2
import numpy as np

def is_frame_too_dark(frame: np.ndarray, threshold: float = 40) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.mean(gray) < threshold


def is_frame_blurry(frame: np.ndarray, threshold: float = 30) -> bool:
    """
    Conservative blur check.
    Only flags extreme blur (motion / defocus),
    not smooth low-texture frames.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Debug tip: print(lap_var) once to see values
    return lap_var < threshold

