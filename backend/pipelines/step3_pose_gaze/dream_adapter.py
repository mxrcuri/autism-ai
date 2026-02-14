import json
import numpy as np

def load_dream_sequence(json_path):
    """
    Convert DREAM 1.2 JSON into Step-3-compatible sequence.
    Robust to:
    - mismatched stream lengths
    - JSON nulls
    - DREAM 'sholder' typo
    """

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[SKIP] Corrupted JSON: {json_path}")
        return []
    except Exception as e:
        print(f"[SKIP] Failed to load {json_path}: {e}")
        return []

    skeleton = data.get("skeleton", {})
    eye = data.get("eye_gaze", {})
    head = data.get("head_gaze", {})
    frame_rate = data.get("frame_rate", 25.0)

    # ---------- 1. Determine number of frames safely ----------
    try:
        if "wrist_left" in skeleton and "x" in skeleton["wrist_left"]:
            n_frames = len(skeleton["wrist_left"]["x"])
        elif "wrist_right" in skeleton and "x" in skeleton["wrist_right"]:
            n_frames = len(skeleton["wrist_right"]["x"])
        else:
            n_frames = 0
    except Exception:
        n_frames = 0

    if n_frames == 0:
        print(f"[WARN] No valid frames in {json_path}")
        return []

    # ---------- 2. Resolve shoulder typo ----------
    if "shoulder_left" in skeleton:
        ls_key, rs_key = "shoulder_left", "shoulder_right"
    else:
        ls_key, rs_key = "sholder_left", "sholder_right"

    # ---------- 3. Universal safe getter ----------
    def get_val(source, key, idx, default=0.0):
        arr = source.get(key, [])
        if not isinstance(arr, list) or idx >= len(arr):
            return default, False
        val = arr[idx]
        if val is None:
            return default, False
        return float(val), True

    seq = []

    for i in range(n_frames):
        frame_valid = True

        # ---- Skeleton ----
        wlx, v1 = get_val(skeleton.get("wrist_left", {}), "x", i)
        wly, _  = get_val(skeleton.get("wrist_left", {}), "y", i)
        wlz, _  = get_val(skeleton.get("wrist_left", {}), "z", i)

        wrx, v2 = get_val(skeleton.get("wrist_right", {}), "x", i)
        wry, _  = get_val(skeleton.get("wrist_right", {}), "y", i)
        wrz, _  = get_val(skeleton.get("wrist_right", {}), "z", i)

        elx, v3 = get_val(skeleton.get("elbow_left", {}), "x", i)
        ely, _  = get_val(skeleton.get("elbow_left", {}), "y", i)
        elz, _  = get_val(skeleton.get("elbow_left", {}), "z", i)

        erx, v4 = get_val(skeleton.get("elbow_right", {}), "x", i)
        ery, _  = get_val(skeleton.get("elbow_right", {}), "y", i)
        erz, _  = get_val(skeleton.get("elbow_right", {}), "z", i)

        slx, v5 = get_val(skeleton.get(ls_key, {}), "x", i)
        sly, _  = get_val(skeleton.get(ls_key, {}), "y", i)
        slz, _  = get_val(skeleton.get(ls_key, {}), "z", i)

        srx, v6 = get_val(skeleton.get(rs_key, {}), "x", i)
        sry, _  = get_val(skeleton.get(rs_key, {}), "y", i)
        srz, _  = get_val(skeleton.get(rs_key, {}), "z", i)

        if not all([v1, v2, v3, v4, v5, v6]):
            frame_valid = False

        pose = {
            "left_wrist": np.array([wlx, wly, wlz], dtype=np.float32),
            "right_wrist": np.array([wrx, wry, wrz], dtype=np.float32),
            "left_elbow": np.array([elx, ely, elz], dtype=np.float32),
            "right_elbow": np.array([erx, ery, erz], dtype=np.float32),
            "left_shoulder": np.array([slx, sly, slz], dtype=np.float32),
            "right_shoulder": np.array([srx, sry, srz], dtype=np.float32),
        }

        # ---- Head ----
        hy, vh1 = get_val(head, "ry", i)
        hp, vh2 = get_val(head, "rx", i)
        hr, vh3 = get_val(head, "rz", i)

        head_pose = {"yaw": hy, "pitch": hp, "roll": hr}

        # ---- Gaze ----
        gx, vg1 = get_val(eye, "rx", i)
        gy, vg2 = get_val(eye, "ry", i)

        gaze = {"gx": gx, "gy": gy, "gz": 1.0}

        seq.append({
            "t": i / frame_rate,
            "pose": pose,
            "head": head_pose,
            "gaze": gaze,
            "valid": frame_valid
        })

    return seq

