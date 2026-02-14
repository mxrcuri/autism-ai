import os
import pickle

CACHE_DIR = "backend/cache/step4_features"


def _cache_path(session_path):
    name = os.path.basename(session_path)
    return os.path.join(CACHE_DIR, name.replace(".json", ".pkl"))


def load_step4(session_path):
    path = _cache_path(session_path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_step4(session_path, features):
    path = _cache_path(session_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)


