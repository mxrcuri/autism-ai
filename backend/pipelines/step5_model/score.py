import torch
import numpy as np


def reconstruction_error(model, dataset):
    device = next(model.parameters()).device
    model.eval()

    errors = []

    with torch.no_grad():
        for x in dataset:
            x = x.unsqueeze(0).to(device)
            recon = model(x)
            err = torch.mean((recon - x) ** 2).item()
            errors.append(err)

    return np.array(errors)

