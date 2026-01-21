import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .autoencoder import TCNAutoencoder


def train_autoencoder(
    dataset,
    epochs=60,
    batch_size=16,
    lr=1e-3
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TCNAutoencoder(
        feature_dim=dataset.X.shape[-1]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total = 0.0

        for batch in loader:
            batch = batch.to(device)

            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"[Epoch {ep+1:03d}] Loss: {total / len(loader):.6f}")

    return model

