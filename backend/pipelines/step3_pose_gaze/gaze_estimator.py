def estimate_gaze_sequence(head_seq):
    gaze_seq = []

    for head in head_seq:
        if head is None:
            gaze_seq.append(None)
            continue

        # Simple heuristic
        gx = -head["yaw"]
        gy = -head["pitch"]
        gz = 1.0  # forward bias

        gaze_seq.append({
            "gx": float(gx),
            "gy": float(gy),
            "gz": float(gz)
        })

    return gaze_seq

