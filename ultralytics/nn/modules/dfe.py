# ultralytics/nn/modules/dfe.py
"""
DFE — Deformable Feature Enhancement module for underwater deformable organisms.

Target problem:
    Underwater organisms such as jellyfish (Scyphozoa) and cuttlefish (Sepiida) exhibit
    (1) highly deformable morphologies that fixed square convolutions struggle to align with,
    and (2) low optical contrast due to tissue translucency, producing weak features easily
    suppressed by background context.

Key references:
    - DCNv2 (Zhu et al., 2019, ICCV): Modulated deformable convolution.
        Deformable ConvNets v2: More Deformable, Better Results.
    - RepLKNet (Ding et al., 2022, CVPR): Large-kernel attention shows that sparse spatial
        sampling can outperform dense square grids.
    - Li et al. (2025, Scientific Reports): SPPFMS — parallel multi-scale processing
        with feature-preserving fusion. DOI: 10.1038/s41598-025-85961-9

Design philosophy (consistent with MDCA and MCRF):
    - Two complementary branches: Deformable Sampling + Contrast Enhancement
    - Progressive fusion via zero-initialized learnable weights
    - Lightweight design: depth-wise / separable convolutions for offset prediction
    - Placed between neck output and detection head (pre-head refinement stage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableSampling2D(nn.Module):
    """
    Lightweight deformable sampling via learned 2D offsets with bilinear interpolation.

    Unlike full DCNv2 which predicts per-kernel-position offsets (2k² channels),
    this separable design predicts a single (dx, dy) offset map per spatial position
    and applies it via grid_sample — reducing offset parameters by ~9× while retaining
    the core benefit of adaptive spatial sampling.

    This design choice is motivated by the observation that underwater deformable
    organisms primarily exhibit smooth, continuous deformations (e.g. jellyfish bell
    contraction) rather than independent per-kernel-element offsets, making a single
    offset map per position sufficient.
    """

    def __init__(self, c: int, reduction: int = 8):
        """
        Args:
            c: input channel count
            reduction: bottleneck ratio for offset prediction
        """
        super().__init__()
        hidden = max(c // reduction, 8)

        # Lightweight offset prediction: group conv + pointwise
        # Predicts offset_x and offset_y separately then concatenates
        self.offset_net = nn.Sequential(
            # First capture spatial context
            nn.Conv2d(c, hidden, 3, padding=1, groups=min(hidden, 4), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            # Then predict offsets
            nn.Conv2d(hidden, 2, 3, padding=1, bias=True),
            nn.Tanh(),  # bound offsets to [-1, 1] in normalised space
        )

        # Small learnable scale factor for offset magnitude
        self.offset_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input feature map

        Returns:
            (B, C, H, W) deformably-sampled feature map
        """
        B, C, H, W = x.shape

        # Predict offsets in normalised coordinates [-1, 1]
        offsets = self.offset_net(x)  # (B, 2, H, W)

        # Build reference grid
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype),
            indexing='ij',
        )
        base_grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
        base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

        # Apply offsets (scale controls magnitude)
        offset_dx = offsets[:, 0:1, :, :].permute(0, 2, 3, 1)  # (B, H, W, 1)
        offset_dy = offsets[:, 1:2, :, :].permute(0, 2, 3, 1)  # (B, H, W, 1)
        sampling_grid = base_grid + self.offset_scale * torch.cat([offset_dx, offset_dy], dim=-1)

        # Bilinear sampling with border padding (stable for edge cells)
        return F.grid_sample(x, sampling_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


class ContrastEnhancement(nn.Module):
    """
    Local contrast amplification module.

    Computes a per-pixel contrast weight by comparing each pixel to its local
    neighbourhood.  This amplifies weak but structured features (semi-transparent
    jellyfish tissue) while suppressing unstructured background noise.

    Mechanism:
        local_mean = AvgPool(3x3)(x)
        deviation  = |x - local_mean|           ← how much does this pixel stand out?
        weight     = σ(Conv([x, local_mean, deviation]))
        output     = x + weight * x             ← amplify weak-but-structured regions
    """

    def __init__(self, c: int, kernel_size: int = 3):
        """
        Args:
            c: input channel count
            kernel_size: local pooling window size
        """
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size // 2)

        # Combine original, local mean, and deviation to predict contrast weight
        self.weight_net = nn.Sequential(
            nn.Conv2d(c * 3, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = self.pool(x)
        deviation = torch.abs(x - local_mean)
        stats = torch.cat([x, local_mean, deviation], dim=1)
        weight = self.weight_net(stats)
        return x * (1.0 + weight)


class DFE(nn.Module):
    """
    DFE — Deformable Feature Enhancement.

    A pre-head refinement module that addresses two failure modes common to
    underwater deformable organisms: irregular shape alignment and low contrast.

    Architecture:
        Input x (from neck / after MCRF fusion)
          │
          ├── DeformableSampling2D → deform_proj → α-gated
          │    Adapts spatial sampling to irregular object boundaries
          │
          ├── ContrastEnhancement → β-gated
          │    Amplifies weak features in low-contrast regions
          │
          └── output = x + α·x_deform + β·x_contrast
               α=0, β=0 at init → safe progressive activation

    Placement in the model:
        Backbone → Neck (MCRF fusion) → DFE → Detect
        (P3, P4, P5 each get their own DFE instance)

    Parameters:
        - Offset network: ~(c²/32 + 2c) additional params — negligible
        - Contrast network: ~(3c²/4) additional params — lightweight
        - Deform projection: c² params (1×1 conv)
        - Total per DFE instance ~5-10% overhead vs the MCRF it refines

    References:
        DCNv2 (Zhu et al., ICCV 2019): Modulated deformable convolution.
        Li et al. (2025, Scientific Reports): Feature preservation for small objects.
    """

    def __init__(self, c: int, offset_reduction: int = 8):
        """
        Args:
            c: input/output channel count (residual design)
            offset_reduction: bottleneck ratio for offset prediction
        """
        super().__init__()

        # ---- Deformable Sampling branch ----
        self.deform_sampler = DeformableSampling2D(c, offset_reduction)
        # Lightweight projection after deformable sampling
        self.deform_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),  # depth-wise
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, 1, bias=False),                        # point-wise
            nn.BatchNorm2d(c),
        )

        # ---- Contrast Enhancement branch ----
        self.contrast_enhance = ContrastEnhancement(c)

        # ---- Progressive fusion weights ----
        # α: deformable branch contribution (init 0 → safe)
        # β: contrast branch contribution (init 0 → safe)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map from neck / MCRF fusion

        Returns:
            (B, C, H, W) refined feature map
        """
        # Deformable sampling: adapt spatial grid to object boundaries
        x_deform = self.deform_sampler(x)
        x_deform = self.deform_proj(x_deform)

        # Contrast enhancement: amplify weak-but-structured regions
        x_contrast = self.contrast_enhance(x)

        # Progressive fusion
        # α=0, β=0 at init → output = x (identity, safe start)
        return x + self.alpha * x_deform + self.beta * x_contrast
