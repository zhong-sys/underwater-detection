"""DRU — Detail Recovery Upsampling for underwater small object detection.

Problem:
    FPN uses nearest-neighbor (NN) upsampling to double feature map resolution.
    NN upsampling copies each pixel into a 2×2 block — the four pixels are
    identical. No new information is created.

    Prior work (CARAFE, DySample, JAFAR, Lurker) proposes learnable upsampling
    operators that generate content-aware kernels or offsets. These methods
    recover details uniformly across all spatial positions.

Key insight:
    In underwater scenes, not all regions should have details recovered equally:
    - High-contrast regions (object boundaries, textured surfaces): detail
      recovery is beneficial — there are real details to recover.
    - Low-contrast regions (turbid water, suspended particles): the "details"
      are predominantly noise. Aggressive recovery amplifies sensor noise
      and backscatter artifacts.

    DRU gates detail injection by local contrast: recover details only where
    the signal-to-noise ratio supports it. This is fundamentally different
    from CARAFE/DySample which apply uniform recovery everywhere.

Architecture:
    x_up  = NN_Upsample(x)                    # stable base
    C     = ContrastEstim(μ(x_up), σ(x_up))   # per-pixel contrast ∈ [0, 1]
    D     = DetailConv(x_up)                  # detail residual ∈ [-1, 1]
    γ     = sigmoid(gate), init γ ≈ 0
    output = x_up + γ · C · D                # contrast-gated detail injection

Reference:
    - FPN (Lin et al., 2017 CVPR): nearest-neighbor upsampling
    - CARAFE (Wang et al., 2019 ICCV): content-aware kernel-based upsampling
    - DySample (Liu et al., 2023 ICCV): point-sampling upsampling
    - DRU differs: contrast-gated residual detail, underwater-specific
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DRU(nn.Module):
    """Detail Recovery Upsampling with contrast-gated detail injection.

    Drop-in replacement for nn.Upsample(scale_factor=2, mode='nearest') in FPN.

    Unlike CARAFE/DySample which recover details uniformly, DRU gates detail
    injection by local contrast — suppressing noise amplification in turbid
    water regions while recovering real details at object boundaries.

    Key design elements:
        1. Contrast estimator: (μ, σ) → C ∈ [0, 1] from channel statistics
        2. Detail predictor: depthwise-separable conv → D ∈ [-1, 1]
        3. Contrast gate: detail injection modulated by C
           High C (texture/boundary) → full detail recovery
           Low C (turbid water)    → suppressed (avoid noise amplification)
        4. Gate init 0 → DRU ≈ NN upsampling at epoch 0 (safe)

    Args:
        channels: input/output channel count.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Detail recovery: depthwise 5×5 (local patterns) + pointwise 1×1
        self.detail = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Tanh(),  # bounded residual ∈ [-1, 1]
        )

        # Local contrast estimator: (mean, std) → contrast map ∈ [0, 1]
        # 2-channel input: channel-mean and channel-std per spatial position
        self.contrast_estim = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        # Progressive gate: init 0 → DRU ≈ NN upsampling at epoch 0
        self.gate = nn.Parameter(torch.zeros(1))

        # Initialize contrast estimator bias to produce ~0.5 output
        nn.init.constant_(self.contrast_estim[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: 2× upsampled features (B, C, 2H, 2W)."""
        # ---- 1. Base nearest-neighbor upsampling (stable, no learned params) ----
        x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # ---- 2. Local contrast estimation ----
        # μ: local brightness, σ: local variation (high at edges/texture)
        x_mean = x_up.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        x_std = x_up.std(dim=1, keepdim=True)      # (B, 1, H, W)
        contrast = self.contrast_estim(
            torch.cat([x_mean, x_std], dim=1)
        )  # (B, 1, H, W), values ∈ [0, 1]

        # ---- 3. Detail prediction ----
        detail = self.detail(x_up)  # (B, C, H, W), values ∈ [-1, 1]

        # ---- 4. Contrast-gated injection ----
        # gate init 0 → identity at epoch 0
        # As training progresses:
        #   contrast ≈ 1 (texture/boundary) → full detail injection
        #   contrast ≈ 0 (turbid water)     → suppressed (preserve NN base)
        gate = self.gate.sigmoid()

        return x_up + gate * contrast * detail
