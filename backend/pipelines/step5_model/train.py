import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path

from .autoencoder import TCNAutoencoder
from .augmentations import augment   # <-- ONLY import you actually need


# --------------------------------------------------
# Calibration: embedding distribution
# --------------------------------------------------
def calibrate_embeddings(model, dataloader, device):
    model.eval()
    Z = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            _, z = model(batch, return_embedding=True)
            Z.append(z.cpu())

    Z = torch.cat(Z, dim=0)

    mu = Z.mean(dim=0)
    cov = torch.cov(Z.T)
    cov += 1e-5 * torch.eye(Z.shape[1], device=cov.device)

    return mu.numpy(), cov.numpy()


# --------------------------------------------------
# Training
# --------------------------------------------------
def train_autoencoder(
    dataset,
    epochs=10,
    batch_size=16,
    lr=1e-3,
    lambda_contrastive=0.1
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    calib_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = TCNAutoencoder(
        feature_dim=dataset.X.shape[-1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        last_z = None

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)

            # ---- two augmented views ----
            x1 = augment(batch)
            x2 = augment(batch)

            recon1, z1 = model(x1, return_embedding=True)
            recon2, z2 = model(x2, return_embedding=True)

            # ---- losses ----
            recon_loss = recon_loss_fn(recon1, x1)
            contrastive = -F.cosine_similarity(z1, z2, dim=1).mean()

            loss = recon_loss + lambda_contrastive * contrastive

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            last_z = z1.detach()

        # ---- diagnostic ----
        print(
            f"[Epoch {ep+1:03d}] "
            f"Loss={total_loss/len(train_loader):.4f} | "
            f"Embedding std={last_z.std(dim=0).mean().item():.4f}"
        )

    # --------------------------------------------------
    # Calibration (POST-TRAINING)
    # --------------------------------------------------
    mu, cov = calibrate_embeddings(model, calib_loader, device)

    calib_path = Path("ml/models/calibration.pkl")
    calib_path.parent.mkdir(parents=True, exist_ok=True)

    with open(calib_path, "wb") as f:
        pickle.dump({"mu": mu, "cov": cov}, f)

    print(f"[CALIBRATION] Saved embedding stats to {calib_path}")

    return model

