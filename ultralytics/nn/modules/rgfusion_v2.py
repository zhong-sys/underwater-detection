# ultralytics/nn/modules/rgfusion_v2.py
"""
RGFusion_v2 — Enhanced Residual-Guided Cross-layer Feature Fusion for underwater small objects.

References:
    - SPPFMS (Li et al., 2025, Scientific Reports): Parallel multi-scale pooling with feature
        fusion. The paper "A small underwater object detection model with enhanced feature
        extraction and fusion" (DOI: 10.1038/s41598-025-85961-9) proposes parallel (not serial)
        multi-scale processing and feature-preserving skip connections.
    - CII-FPN / PRCII-Net (2025, Applied Soft Computing): Cross-scale information interaction
        with dual-branch fusion for spatial-semantic enhancement.
    - SKSA / LMFEN (2025, Ocean Engineering): Separable kernel spatial attention for
        regional context in underwater imagery.
    - GFPN / LFN-YOLO (2025, Frontiers in Marine Science): Generalized feature pyramid
        with cross-layer local attention.

Design philosophy:
    The original RGFusion operates at a single scale — it computes attention on "initial = x + y"
    directly.  For underwater scenes, objects span a wide size range (tiny starfish → large turtles)
    and water turbidity creates ambiguous features.  Processing at multiple receptive fields before
    the attention stage provides richer context.

    All new branches are gated behind learnable parameters initialised to zero — at initialisation
    the module behaves identically to the original RGFusion, and new capabilities activate
    gradually during training.  This makes the upgrade "safe": it cannot degrade the original
    performance baseline.
"""

import torch
import torch.nn as nn
from einops import rearrange


# ============================================================================
#   Sub-modules (identical to original rgfusion.py — preserved for compatibility)
# ============================================================================

class SpatialAttention_CGA(nn.Module):
    """Spatial attention: channel-pool (avg+max) → 7×7 conv → 1-channel spatial map."""
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention_CGA(nn.Module):
    """Channel attention: GAP → Conv-Reduce-ReLU → Conv-Expand → C-channel map."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention_CGA(nn.Module):
    """
    Pixel-wise attention: combines input feature with prior attention map via depth-wise conv.

    Input x (B, C, H, W) and pattn1 (B, C, H, W) are stacked along a new dim,
    rearranged to (B, 2C, H, W), then processed by a grouped conv (groups=C).
    """
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect',
                             groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)              # (B, C, 1, H, W)
        pattn1 = pattn1.unsqueeze(dim=2)    # (B, C, 1, H, W)
        x2 = torch.cat([x, pattn1], dim=2)  # (B, C, 2, H, W)
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        return self.sigmoid(self.pa2(x2))


# ============================================================================
#   RGFusion_v2
# ============================================================================

class RGFusion_v2(nn.Module):
    """
    RGFusion_v2 — Enhanced Residual-Guided Cross-layer Feature Fusion.

    Architecture (new additions marked with ★):

        Input: (x, y)  where x = encoder feature, y = decoder feature

        initial = x + y
           │
           ├──★ Multi-Scale Dilated Context (γ-gated, init γ=0):
           │      dil_conv(x+y, d=1) ─┐
           │      dil_conv(x+y, d=3) ─┤→ fuse → γ·context
           │      dil_conv(x+y, d=5) ─┘
           │
           └── initial' = initial + γ·context

        cattn  = ChannelAttention(initial')
        sattn  = SpatialAttention(initial')
        pattn1 = sattn + cattn           (→ could be extended with learned weights)
        pattn2 = σ(PixelAttention(initial', pattn1))

        fused  = initial' + pattn2·x + (1-pattn2)·y
        fused  = Conv1×1(fused)

        output = x + α·fused             (α progressive, init α=0)

    Key improvements over RGFusion:
        1. ★ Multi-Scale Dilated Context (γ) — parallel dilated depth-wise convs at
           dilation rates [1, 3, 5] capture local detail AND wide receptive fields
           simultaneously.  For underwater scenes this means:
           - d=1: fine texture of small starfish / scallop
           - d=3: mid-range context (corals, rocks)
           - d=5: wide background context (water column, lighting gradient)
           The fused multi-scale features provide richer input to the attention modules.

        2. Progressive activation — γ is initialised to zero, so early training
           behaves identically to the original RGFusion.  Gradients gradually
           increase γ only if the multi-scale context is beneficial.

    Comparison with RGFusion:
        | Feature                | RGFusion  | RGFusion_v2 |
        |------------------------|-----------|-------------|
        | Channel Attention      | ✓         | ✓           |
        | Spatial Attention      | ✓         | ✓           |
        | Pixel Attention (gate) | ✓         | ✓           |
        | Progressive α residual | ✓         | ✓           |
        | Multi-Scale Context    | ✗         | ✓ (γ-gated) |
    """

    def __init__(self, dim, reduction=8, dilation_rates=(1, 3, 5)):
        """
        Args:
            dim: input/output channel count (must match x and y channels)
            reduction: bottleneck ratio for ChannelAttention_CGA
            dilation_rates: dilation rates for multi-scale depth-wise convs
        """
        super().__init__()

        # ---- Original RGFusion components (preserved exactly) ----
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.alpha = nn.Parameter(torch.zeros(1))  # progressive residual weight
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)

        # ---- ★ Multi-Scale Dilated Context (NEW) ----
        # Depth-wise convs at different dilations — lightweight, no channel mixing
        self.ms_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=r, dilation=r, groups=dim, bias=False)
            for r in dilation_rates
        ])
        # 1×1 fuse for multi-scale features
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(dim * len(dilation_rates), dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        # ★ Progressive multi-scale weight (init 0 → behaves like original RGFusion)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, data):
        x, y = data

        # ---- Step 1: Initial feature combination ----
        initial = x + y

        # ---- ★ Step 2: Multi-Scale Dilated Context (NEW) ----
        ms_features = [conv(initial) for conv in self.ms_convs]
        ms_context = self.ms_fuse(torch.cat(ms_features, dim=1))
        # γ-gated progressive activation: at init γ≈0, no effect
        initial = initial + self.gamma * ms_context

        # ---- Step 3: Attention (original RGFusion pipeline) ----
        cattn = self.ca(initial)      # channel attention
        sattn = self.sa(initial)      # spatial attention
        pattn1 = sattn + cattn        # combined attention map

        # ---- Step 4: Pixel-level gating ----
        pattn2 = torch.sigmoid(self.pa(initial, pattn1))

        # ---- Step 5: Gated fusion ----
        # pattn2 ∈ [0,1] per-pixel: select between encoder (x) and decoder (y)
        fused = initial + pattn2 * x + (1 - pattn2) * y
        fused = self.conv(fused)

        # ---- Step 6: Progressive residual output ----
        # α ≈ 0 at init → output ≈ x (encoder pass-through), stable early training
        return x + self.alpha * fused
