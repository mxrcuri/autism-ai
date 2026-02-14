import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Temporal Residual Block (collapse-resistant)
# --------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        dilation=1,
        dropout=0.3
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else None
        )

        # residual scaling (VERY important)
        self.res_scale = 0.5

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # causal trim
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + self.res_scale * res


# --------------------------------------------------
# Multi-Scale Temporal Encoder
# --------------------------------------------------
class TemporalEncoder(nn.Module):
    def __init__(
        self,
        in_ch,
        hidden_ch=128,
        levels=5,
        kernel_size=3,
        dropout=0.3
    ):
        super().__init__()

        blocks = []
        ch = in_ch

        for i in range(levels):
            dilation = 2 ** i
            blocks.append(
                TemporalBlock(
                    ch,
                    hidden_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            ch = hidden_ch

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        # x: [B, C, T]
        return self.network(x)


# --------------------------------------------------
# Final TCN Model (Embedding Generator)
# --------------------------------------------------
class TCN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_ch=128,
        emb_dim=64,
        levels=5,
        dropout=0.3,
        last_k=5
    ):
        super().__init__()

        self.encoder = TemporalEncoder(
            in_ch=in_features,
            hidden_ch=hidden_ch,
            levels=levels,
            dropout=dropout
        )

        self.last_k = last_k

        # projection head (NOT classifier)
        self.proj = nn.Sequential(
            nn.Linear(hidden_ch, hidden_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ch, emb_dim)
        )

        # temperature for contrastive / cosine use (optional)
        self.temperature = 0.1

    def forward(self, x):
        """
        x: [B, T, F]
        returns: embedding [B, emb_dim]
        """

        # convert to Conv1D format
        x = x.transpose(1, 2)  # [B, F, T]

        h = self.encoder(x)    # [B, hidden_ch, T]

        # ----- IMPORTANT: no mean over entire time -----
        k = min(self.last_k, h.shape[-1])
        h = h[:, :, -k:]       # last-k frames
        h = h.mean(dim=2)      # [B, hidden_ch]

        z = self.proj(h)       # [B, emb_dim]

        return z


# --------------------------------------------------
# Optional: Contrastive helper (no labels)
# --------------------------------------------------
def contrastive_loss(z1, z2, temperature=0.1):
    """
    z1, z2: embeddings of two augmentations of same sequence
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return -F.cosine_similarity(z1, z2).mean() / temperature

