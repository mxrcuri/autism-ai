import torch
import torch.nn as nn
from .tcn import TemporalBlock


class TCNAutoencoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            TemporalBlock(feature_dim, hidden_dim, dilation=1),
            TemporalBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalBlock(hidden_dim, hidden_dim, dilation=4),
        )

        self.decoder = nn.Sequential(
            TemporalBlock(hidden_dim, hidden_dim, dilation=4),
            TemporalBlock(hidden_dim, hidden_dim, dilation=2),
            TemporalBlock(hidden_dim, feature_dim, dilation=1),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.transpose(1, 2)

