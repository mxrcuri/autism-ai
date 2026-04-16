"""
model_loader.py

Loads the TCNAutoencoder checkpoint exactly once and caches it.
Subsequent calls to get_model() return the same (model, device) tuple instantly.

The model lives at:
    ml/models/tcn_autoencoder.pth   — PyTorch state dict
    ml/models/calibration.pkl       — used inside score_sequence()

IMPORTANT: Run uvicorn from the /backend/ directory so that relative
paths resolve correctly.
"""

import os
import torch

from app.config import FEATURE_KEYS
from pipelines.step5_model.autoencoder import TCN_VAE

# Absolute paths so this works regardless of CWD.
# model_loader.py lives at: backend/app/model_loader.py
# Models live at:           backend/ml/models/
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
_MODEL_PATH = os.path.join(_HERE, "ml", "models", "tcn_autoencoder.pth")
_CALIB_PATH = os.path.join(_HERE, "ml", "models", "calibration.pkl")

# Cached singletons
_model = None
_device = None


def get_model():
    """
    Returns (model, device). Loads from disk on first call only.
    Thread-safe enough for FastAPI's single-process dev server.
    """
    global _model, _device

    if _model is not None:
        return _model, _device

    # --- Device ---
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ModelLoader] Using device: {_device}")

    # --- Resolve feature_dim from calibration (avoids hard-coding) ---
    # calibration.pkl stores {"mu": array, "sigma": array} over window scores;
    # the feature_dim is the number of features in FEATURE_KEYS (always 9).
    FEATURE_DIM = len(FEATURE_KEYS)

    # --- Build model architecture (must match training config) ---

    _model = TCN_VAE(
        feature_dim=FEATURE_DIM,
        hidden_ch=128,
        emb_dim=64,
        levels=5,
        dropout=0.3,
    )

    # --- Load weights ---
    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"[ModelLoader] Checkpoint not found at '{_MODEL_PATH}'. "
            "Run uvicorn from the /backend/ directory."
        )

    state = torch.load(_MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.to(_device)
    _model.eval()

    print(f"[ModelLoader] Loaded checkpoint from '{_MODEL_PATH}'")
    return _model, _device
