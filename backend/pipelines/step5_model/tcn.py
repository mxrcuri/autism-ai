import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
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

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else None
        )

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # causal trim
        out = self.norm(out)
        out = self.relu(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res

