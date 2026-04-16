import os
import torch
import numpy as np
import pickle
from scipy.special import expit  # sigmoid
from scipy.spatial.distance import mahalanobis

# Build absolute path so this works regardless of CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))           # backend/ml/inference
_CALIB_PATH = os.path.join(_HERE, "..", "models", "calibration.pkl")

# Load calibration once at import time.
# The calibration pkl contains 'mu' (64,) and 'cov' (64, 64) for the latent embeddings.
with open(_CALIB_PATH, "rb") as _f:
    _calib = pickle.load(_f)

_MU  = np.array(_calib["mu"])
_COV = np.array(_calib["cov"])

# Precompute the inverse covariance matrix for Mahalanobis filtering
# Add small identity to diagonal for numerical stability before auto-inversion
_INV_COV = np.linalg.pinv(_COV + np.eye(_COV.shape[0]) * 1e-6)

def score_sequence(model, features, device):
    """
    Compute screening confidence score for a session based on the 
    TCNAutoencoder's latent embeddings using Mahalanobis distance 
    from the normative/calibration distribution.
    """
    model.eval()
    x = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        # TCN_VAE returns: recon, mu, logvar, z
        recon, mu_embed, logvar, z_embed = model(x)
        mu_embed = mu_embed.cpu().numpy()  # shape: [B, emb_dim] e.g. [1, 64]

        # Also get basic sequence reconstruction error for logging
        errors = torch.mean((x - recon) ** 2, dim=(1, 2)).cpu().numpy()

    # The model processes the whole sequence into a single embedding vector 
    # per batch item. Since batch_size=1, mu_embed is shape (1, 64).
    # We compute the Mahalanobis distance of this sequence's embedding from the norm.
    z_dist = mahalanobis(mu_embed[0], _MU, _INV_COV)

    # Convert the distance into a confidence score
    # A larger distance means it's an anomaly (further from neurotypical norm).
    # Since Mahalanobis distance is positive, we can map it to probabilities.
    # An acceptable rough map is using a shifted sigmoid.
    # Expected distance for a 64-dim standard normal is around sqrt(64) = 8.
    dist_shift = z_dist - np.sqrt(len(_MU))
    confidence = float(expit(dist_shift))

    return {
        "confidence":     confidence,
        "mean_deviation": float(z_dist),    # Report actual distance
        "window_scores":  errors.tolist(),  # Report raw reconstruction errors
    }
