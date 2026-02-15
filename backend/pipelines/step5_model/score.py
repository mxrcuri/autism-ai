import torch
import numpy as np


def reconstruction_error(model, dataset):
    device = next(model.parameters()).device
    model.eval()

    errors = []

    with torch.no_grad():
        for x in dataset:

            # Convert numpy array to torch tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            # Ensure float, add batch dimension, move to device
            x = x.float().unsqueeze(0).to(device)

            # Forward pass
            recon = model(x)

            # Compute MSE reconstruction error
            err = torch.mean((recon - x) ** 2).item()
            errors.append(err)

    return np.array(errors)

def score_sequence(features, model):
    """
    Bridge function to match the API expectation.
    Takes features, runs reconstruction, and returns a single score.
    """
    errors = reconstruction_error(model, features)
    # Return the average error across all windows in the sequence
    return float(np.mean(errors))
