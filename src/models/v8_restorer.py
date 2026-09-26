"""Small codec/QP-conditioned decoder restorer; one output for every analyzer."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pointwise(self.activation(self.depthwise(x)))


class V8Restorer(nn.Module):
    """Predict a bounded 128-pixel RGB residual from three decoded frames."""

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(11, channels, 3, stride=2, padding=1),
                                  nn.SiLU(), _Block(channels), _Block(channels))
        self.head = nn.Conv2d(channels, 12, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, decoded: torch.Tensor, qp: int, native_size: int) -> torch.Tensor:
        if decoded.ndim != 5 or decoded.shape[1] != 3 or decoded.shape[2] < 3:
            raise ValueError("expected [B,3,T,H,W] with T>=3")
        if not 0 <= qp <= 51 or native_size not in (96, 112, 128):
            raise ValueError("invalid QP or native resolution")
        b, c, t, h, w = decoded.shape
        x = decoded.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if (h, w) != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        frames = x.reshape(b, t, c, 128, 128)
        previous = torch.cat((frames[:, :1], frames[:, :-1]), dim=1)
        following = torch.cat((frames[:, 1:], frames[:, -1:]), dim=1)
        temporal = torch.cat((previous, frames, following), dim=2).reshape(b * t, 9, 128, 128)
        conditioning = torch.stack((torch.full_like(x[:, :1], qp / 51),
                                    torch.full_like(x[:, :1], native_size / 128)), dim=1)
        conditioning = conditioning.reshape(b * t, 2, 128, 128)
        residual = self.shuffle(self.head(self.stem(torch.cat((temporal, conditioning), 1))))
        restored = (x + .1 * torch.tanh(residual)).clamp(0, 1)
        return restored.reshape(b, t, 3, 128, 128).permute(0, 2, 1, 3, 4)
