"""A²FM — Adaptive Attenuation-compensated Fusion Module.

Beer-Lambert inspired: estimates per-channel transmission coefficients,
applies learnable inverse attenuation compensation, fuses encoder/decoder
features via spatially-adaptive gating.  λ=0 at init → identity, progressively
activates as training proceeds.
"""

import torch
import torch.nn as nn


class TransmissionEstimator(nn.Module):
    """Per-channel transmission coefficient estimator: GAP → FC-ReLU → FC-Sigmoid."""

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        hidden = max(c // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.gap(x))  # (B, C, 1, 1)


class AttenuationCompensation(nn.Module):
    """x' = x * (1 + λ * (1 - t)),  λ init 0 → progressive compensation."""

    def __init__(self):
        super().__init__()
        self.lmbda = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.lmbda * (1.0 - t))


class FusionGate(nn.Module):
    """Spatially-adaptive gate: Concat → DWConv → BN → ReLU → Conv1×1 → Sigmoid."""

    def __init__(self, c: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_comp: torch.Tensor, y_comp: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat([x_comp, y_comp], dim=1))  # (B, 1, H, W)


class A2FM(nn.Module):
    """Adaptive Attenuation-compensated Fusion Module.

    Pipeline:
        t = TransEstim(x), TransEstim(y)          — per-channel transmission
        x', y' = Compensate(x,t), Compensate(y,t) — inverse attenuation
        weight = FusionGate(x', y')               — per-pixel fusion weight
        output = y + proj(weight*x' + (1-weight)*y')  — decoder-primary residual
    """

    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        self.trans_estim = TransmissionEstimator(c, reduction)
        self.compensation = AttenuationCompensation()
        self.fusion_gate = FusionGate(c)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, data):
        """Args: data = (x, y) — x=encoder, y=decoder.  Returns fused feature map."""
        x, y = data

        t_enc = self.trans_estim(x)
        t_dec = self.trans_estim(y)

        x_comp = self.compensation(x, t_enc)
        y_comp = self.compensation(y, t_dec)

        weight = self.fusion_gate(x_comp, y_comp)
        fused = weight * x_comp + (1.0 - weight) * y_comp

        return y + self.out_conv(fused)
