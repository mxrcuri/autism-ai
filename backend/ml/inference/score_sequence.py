import torch
import numpy as np
import pickle
from scipy.special import expit  # sigmoid


def score_sequence(model, features, device):
    """
    Compute screening confidence score for a session.
    """
    # Load calibration stats
    with open("ml/models/calibration.pkl", "rb") as f:
        calib = pickle.load(f)

    mu, sigma = calib["mu"], calib["sigma"]

    model.eval()
    x = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(x)
        errors = torch.mean((x - recon) ** 2, dim=(1, 2)).cpu().numpy()

    # z-score normalization
    z = (errors - mu) / sigma

    # aggregate over session
    z_mean = float(np.mean(z))

    # screening confidence in [0, 1]
    confidence = float(expit(z_mean))

    return {
        "confidence": confidence,
        "mean_deviation": z_mean,
        "window_scores": z.tolist()
    }
