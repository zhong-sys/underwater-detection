
"""MSRF -- Multi-Scale Receptive Field for P3 small-object detection head.

Problem:
    YOLO11 P3 (stride=8): a 15x15px object ~2x2 cells in feature space.
    Standard 3x3 kernel ~1 cell radius -- cannot see the full object contour
    across adjacent grid cells.  Small objects with centers near cell boundaries
    are split across multiple cells; the responsible cell sees only a fragment.

Solution:
    Three parallel depthwise branches with different kernel sizes expand the
    effective receptive field BEFORE detection:
    - 3x3 DWConv:   local texture (original FoV)
    - 7x7 DWConv:   neighborhood (2-3 adjacent cells)
    - d=3 DWConv:   distant context (~5 cell span)

    Branches are concatenated and fused via linear 1x1 conv + BN, then added
    as a residual.  NO progressive gate -- when training from scratch, all
    weights are random anyway; the gate only hurts by desynchronizing module
    activation from backbone convergence.

Complementarity to DCAv2:
    DCAv2: feature QUALITY (channel-wise, photometric)
    MSRF:  feature COVERAGE (spatial receptive field, geometric)

Reference:
    - Liu et al. (ECCV 2018): RFBNet -- receptive field blocks
    - MSRF differs: depthwise-separable multi-scale (vs Inception-style),
      no gate (full-activation from epoch 0), P3-targeted for small objects.
"""

import torch
import torch.nn as nn


class MSRF(nn.Module):
    """Multi-Scale Receptive Field for P3 detection head.

    Expands effective receptive field via three parallel depthwise branches,
    helping small objects be fully visible across neighboring grid cells.

    Args:
        channels: input/output channel count.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Three parallel receptive-field branches (depthwise, minimal params)
        self.branch_local = nn.Conv2d(channels, channels, 3, padding=1,
                                       groups=channels, bias=False)
        self.branch_neighbor = nn.Conv2d(channels, channels, 7, padding=3,
                                          groups=channels, bias=False)
        self.branch_dilated = nn.Conv2d(channels, channels, 3, padding=3,
                                         dilation=3, groups=channels, bias=False)

        # Linear fusion: 3*C -> C (nonlinearity from downstream C3k2)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: context-enriched features (B, C, H, W)."""
        b_local = self.branch_local(x)
        b_neighbor = self.branch_neighbor(x)
        b_dilated = self.branch_dilated(x)

        fused = self.fuse(torch.cat([b_local, b_neighbor, b_dilated], dim=1))
        return x + fused
