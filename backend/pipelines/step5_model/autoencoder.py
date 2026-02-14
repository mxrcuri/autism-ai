import torch
import torch.nn as nn
import torch.nn.functional as F
from .tcn import TCN   # the TCN we finalized earlier


class TCNAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_ch=128,
        emb_dim=64,
        levels=5,
        dropout=0.3
    ):
        super().__init__()

        # -------- Encoder --------
        self.encoder = TCN(
            in_features=feature_dim,
            hidden_ch=hidden_ch,
            emb_dim=emb_dim,
            levels=levels,
            dropout=dropout
        )

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, feature_dim)
        )

    def forward(self, x, return_embedding=False):
        """
        x: [B, T, F]
        """
        z = self.encoder(x)             # [B, emb_dim]
        recon = self.decoder(z)         # [B, F]

        # expand recon across time (simple but stable)
        recon = recon.unsqueeze(1).expand(-1, x.shape[1], -1)

        if return_embedding:
            return recon, z
        return recon

