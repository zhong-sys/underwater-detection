"""CFD — Contrast-Aware Foreground Discriminator for underwater object detection.

Problem (from error analysis on RUOD):
    yolo11n baseline: 42364 FP vs 7010 TP (6×), Precision as low as 0.09.
    The detection head cannot distinguish foreground from background in low-contrast
    underwater scenes — turbidity patches and suspended particles are falsely
    detected as objects.

Root cause:
    In underwater images, contrast degradation blurs the boundary between object
    edges and background noise. The detection head receives features where
    background regions have similar activation patterns to foreground regions.

Key insight:
    Local contrast (spatial mean + variance of channel activations) is a reliable
    foreground/background cue:
    - Object boundaries and textured surfaces → high local variance
    - Turbid water, suspended particles     → low local variance, uniform mean

    CFD computes a per-position foreground probability from local contrast
    statistics and gates the feature map accordingly. Background-like positions
    are suppressed; foreground-like positions pass through.

Architecture:
    X → [μ(X), σ(X)]_spatial → 1×1 Conv (2→hidden→1) → Sigmoid → F ∈ [0,1]
    γ  = sigmoid(gate), gate init -5 → γ ≈ 0 at epoch 0
    output = X × (1 - γ + γ·F)

    Progressive activation (gate init -5):
        epoch 0:  γ ≈ 0.007 → output ≈ X (identity, safe cold start)
        training: γ increases → output transitions toward X·F
    This prevents the random-initialized fg_map from destroying feature signal
    in early epochs, avoiding the slow loss convergence problem.

    - 2-channel input: channel-mean and channel-std per spatial position
    - ~0.001M parameters (virtually zero cost)
    - Applied on P3/P4/P5 before Detect head

Unlike CAFE/DRU which operate on backbone/neck and rely on indirect gradient:
    CFD operates directly on detection-head inputs and uses spatial (not channel)
    statistics — targeting the actual FG/BG decision point.
"""

import torch
import torch.nn as nn


class CFD(nn.Module):
    """Contrast-Aware Foreground Discriminator.

    Computes a per-position foreground probability from local channel statistics
    (mean, std) and gates the feature map to suppress background activations.

    Args:
        channels: input/output channel count.
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 16, 8)

        # 2-channel input: (mean, std) per spatial position → FG probability
        self.fg = nn.Sequential(
            nn.Conv2d(2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )

        # Progressive gate: init -5 → sigmoid ≈ 0.007 → CFD ≈ identity at epoch 0
        # As training progresses, gate activates → output transitions toward x * fg_map
        self.gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: gated features (B, C, H, W)."""
        # Local contrast statistics
        mean = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        std = x.std(dim=1, keepdim=True)      # (B, 1, H, W)

        # Foreground probability map
        fg_map = self.fg(torch.cat([mean, std], dim=1))  # (B, 1, H, W)

        # Progressive blend: identity → gated
        # epoch 0: gate ≈ 0 → output ≈ x (safe, no signal loss)
        # training: gate ↑ → output → x * fg_map
        gamma = self.gate.sigmoid()
        return x * (1 - gamma + gamma * fg_map)
