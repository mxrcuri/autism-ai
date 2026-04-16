import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle
from pathlib import Path

from .autoencoder import TCN_VAE
from .augmentations import augment


# --------------------------------------------------
# Calibration: embedding distribution
# --------------------------------------------------
def calibrate_embeddings(model, dataloader, device):
    model.eval()
    Z_mu = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)
            _, mu, _, _ = model(batch)
            Z_mu.append(mu.cpu())

    Z_mu = torch.cat(Z_mu, dim=0)

    # For VAEs, the mu distribution is close to standard normal
    # We calibrate exact covariance to track the true learned boundary
    mu_center = Z_mu.mean(dim=0)
    cov = torch.cov(Z_mu.T)
    cov += 1e-5 * torch.eye(Z_mu.shape[1], device=cov.device)

    return mu_center.numpy(), cov.numpy()


# --------------------------------------------------
# Loss function for VAE
# --------------------------------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Beta-VAE loss function
    Reconstruction loss + Beta * KL Divergence
    """
    # MSE Reconstruction
    recon_loss = F.mse_loss(recon_x, x)

    # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    loss = recon_loss + beta * kl_divergence
    return loss, recon_loss, kl_divergence


# --------------------------------------------------
# Training
# --------------------------------------------------
def train_autoencoder(
    dataset,
    epochs=15,
    batch_size=16,
    lr=1e-3,
    beta=0.1  # Start with light KL penalization
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

    model = TCN_VAE(
        feature_dim=dataset.X.shape[-1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        last_mu = None

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]

            batch = batch.to(device)

            x = augment(batch)

            recon, mu, logvar, z = model(x)

            # Calculate VAE loss
            # Anneal beta from 0 to target_beta over the first few epochs (optional, but robust)
            current_beta = beta * (min(1.0, (ep + 1) / max(1, epochs * 0.5)))
            
            loss, r_loss, k_loss = vae_loss(recon, x, mu, logvar, beta=current_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += r_loss.item()
            total_kl += k_loss.item()
            last_mu = mu.detach()

        # ---- diagnostic ----
        batches = len(train_loader)
        print(
            f"[Epoch {ep+1:03d}] "
            f"Loss={total_loss/batches:.4f} "
            f"(Recon={total_recon/batches:.4f}, KL={total_kl/batches:.4f}) | "
            f"Embed std={last_mu.std(dim=0).mean().item():.4f}"
        )

    # --------------------------------------------------
    # Calibration (POST-TRAINING)
    # --------------------------------------------------
    mu_c, cov_c = calibrate_embeddings(model, calib_loader, device)

    calib_path = Path("ml/models/calibration.pkl")
    calib_path.parent.mkdir(parents=True, exist_ok=True)

    with open(calib_path, "wb") as f:
        pickle.dump({"mu": mu_c, "cov": cov_c}, f)

    print(f"[CALIBRATION] Saved embedding stats to {calib_path}")
    model_path = Path("ml/models/tcn_autoencoder.pth")
    torch.save(model.state_dict(), model_path)

    print(f"[MODEL] Saved trained model to {model_path}")
    return model
