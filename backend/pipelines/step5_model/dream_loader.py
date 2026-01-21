import os

def load_user_sessions(root_dir):
    """
    Returns:
        dict[user_id] -> list of session json file paths
    """
    users = {}

    for entry in sorted(os.listdir(root_dir)):
        if not entry.lower().startswith("user"):
            continue

        user_path = os.path.join(root_dir, entry)
        if not os.path.isdir(user_path):
            continue

        sessions = sorted([
            os.path.join(user_path, f)
            for f in os.listdir(user_path)
            if f.endswith(".json")
        ])

        if sessions:
            users[entry] = sessions

    return users

