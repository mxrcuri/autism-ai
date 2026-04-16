import torch
import torch.nn as nn
from .tcn import TCN

class TCN_VAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_ch=128,
        emb_dim=64,
        levels=5,
        dropout=0.3
    ):
        super().__init__()

        # -------- Encoder (Temporal Extraction) --------
        self.encoder_base = TCN(
            in_features=feature_dim,
            hidden_ch=hidden_ch,
            emb_dim=hidden_ch,  # Output hidden state, not final embedding
            levels=levels,
            dropout=dropout
        )

        # -------- Variational Bottleneck --------
        self.fc_mu = nn.Linear(hidden_ch, emb_dim)
        self.fc_logvar = nn.Linear(hidden_ch, emb_dim)

        # -------- Decoder --------
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_ch),
            nn.ReLU(),
            nn.Linear(hidden_ch, feature_dim)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic output for inference

    def forward(self, x):
        """
        x: [B, T, F]
        """
        # 1. Temporal encoding
        hidden = self.encoder_base(x)       # [B, hidden_ch]
        
        # 2. Extract latent distributions
        mu = self.fc_mu(hidden)             # [B, emb_dim]
        logvar = self.fc_logvar(hidden)     # [B, emb_dim]
        
        # 3. Reparameterization trick
        z = self.reparameterize(mu, logvar) # [B, emb_dim]

        # 4. Decode
        recon = self.decoder(z)             # [B, F]

        # expand recon across time assuming time-invariant bottleneck
        recon = recon.unsqueeze(1).expand(-1, x.shape[1], -1)

        return recon, mu, logvar, z
