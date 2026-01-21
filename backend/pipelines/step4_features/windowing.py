import pandas as pd
import numpy as np

def fill_short_gaps(arr, limit=3):
    """
    Linearly interpolate gaps up to `limit` frames.
    Larger gaps remain NaN.
    """
    df = pd.DataFrame(arr)
    return df.interpolate(method="linear", limit=limit).to_numpy()


def sliding_windows(sequence, window_size, stride, max_gap=3):
    windows = []
    T = len(sequence)

    for start in range(0, T - window_size + 1, stride):
        window = sequence[start:start + window_size]

        # Extract validity mask
        valid = np.array([s["valid"] for s in window])

        # If too many invalid frames, drop early
        if np.sum(~valid) > max_gap:
            continue

        windows.append(window)

    return windows

