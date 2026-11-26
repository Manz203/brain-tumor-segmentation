"""Attention gates on encoder skip connections (2-D binary segmentation)."""
import torch
from torch import nn
from .model import UNet

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.skip = nn.Conv2d(channels, hidden, 1)
        self.gate = nn.Conv2d(channels, hidden, 1)
        self.psi = nn.Conv2d(hidden, 1, 1)

    def forward(self, skip, gate):
        alpha = torch.sigmoid(self.psi(torch.relu(self.skip(skip) + self.gate(gate))))
        return skip * alpha

class AttentionUNet(UNet):
    def __init__(self, width=32):
        super().__init__(width)
        self.attention = nn.ModuleList(AttentionGate(width * k) for k in (8, 4, 2, 1))

    def forward(self, x):
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise ValueError("Image dimensions must be divisible by 16")
        skips = []
        for layer in self.enc:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        x = self.mid(x)
        for up, dec, attention, skip in zip(self.up, self.dec, self.attention, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, attention(skip, x)], dim=1))
        return self.head(x)
