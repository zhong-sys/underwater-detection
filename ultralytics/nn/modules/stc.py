"""STC: Spectral Transmission Compensator.

Lightweight channel-wise compensation module for underwater object detection.
Reinterprets channel attention through the Beer-Lambert law: instead of
learning arbitrary importance weights (SE, ECA), STC estimates per-channel
optical transmission coefficients and applies physically-motivated inverse
compensation.

Core idea:
    t = sigma(FC(GAP(x)))       — transmission coefficient (physical meaning)
    alpha = 1 + lambda * (1-t)  — Beer-Lambert inverse compensation
    output = alpha * x           — compensate attenuated channels

Unlike SE/ECA which learn "this channel is important" without knowing why,
STC learns "this channel is attenuated by X, compensate by Y" — the
compensation follows the exponential attenuation model.

At init (lambda=0): alpha=1, output=x (identity, safe integration).
lambda gets gradients immediately; transmission estimator activates
progressively as lambda grows (ControlNet pattern).

Reference:
    Beer-Lambert law: I_d(lambda) = I_0(lambda) * exp(-beta(lambda) * d)
    SE-Net (Hu et al., 2018 CVPR): channel attention via global pooling
    ECA-Net (Wang et al., 2020 CVPR): 1D cross-channel interaction
    FcaNet (Qin et al., 2021 ICCV): multi-spectral channel attention
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.a2fm import TransmissionEstimator


class SpectralTransmissionCompensator(nn.Module):
    """Spectral Transmission Compensator (STC).

    Estimates per-channel optical transmission and compensates for
    wavelength-dependent underwater attenuation.

    Pipeline:
        1. t = TransmissionEstimator(x)        — per-channel transmission
        2. alpha = 1 + lambda * (1 - t)        — compensation factor
        3. output = alpha * x                  — apply compensation

    At init (lambda=0): alpha=1, output=x (identity).
    lambda (per-channel, init 0) learns which channels need compensation.
    As lambda grows, transmission estimator receives gradients and learns
    to predict physically meaningful transmission coefficients.
    """

    def __init__(self, c: int, reduction: int = 4):
        """Args: c = input channels, reduction = TE bottleneck ratio."""
        super().__init__()

        # Per-channel transmission estimator (reuses A2FM component)
        self.trans_estim = TransmissionEstimator(c, reduction)

        # Per-channel compensation strength (init 0 -> identity)
        # Shape (1, C, 1, 1) for broadcast with (B, C, H, W)
        self.lmbda = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Returns compensated features (B, C, H, W)."""
        # 1. Estimate per-channel transmission
        t = self.trans_estim(x)  # (B, C, 1, 1)

        # 2. Beer-Lambert inverse compensation
        # t close to 0 (high attenuation) -> alpha > 1 (compensate)
        # t close to 1 (clear)           -> alpha = 1 (preserve)
        # At init (lambda=0): alpha=1 -> output=x (identity)
        alpha = 1.0 + self.lmbda * (1.0 - t)  # (1|B, C, 1, 1)

        # 3. Apply compensation
        return alpha * x
