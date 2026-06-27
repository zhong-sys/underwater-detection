# ultralytics/nn/modules/dca_conv_v2.py
"""
DCAConv_v2 — Enhanced Dual Convolution + Attention module for underwater small object detection.

References:
    - CSPSL (Li et al., 2025, Scientific Reports): Feature preservation through split processing.
        A small underwater object detection model with enhanced feature extraction and fusion.
        DOI: 10.1038/s41598-025-85961-9
    - MSDA / YOLOv11-MSE (2025, JMSE): Multi-Scale Dilated Attention for variable-sized objects.
    - CBAM (Woo et al., 2018, ECCV): Sequential Channel + Spatial attention.
"""

import torch
import torch.nn as nn


class ChannelAttention_SE(nn.Module):
    """Squeeze-and-Excitation style channel attention with reduction ratio."""
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention_Simple(nn.Module):
    """Lightweight spatial attention: channel-pool → 7×7 conv → sigmoid."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DualConv(nn.Module):
    """Dual-branch convolution: grouped 3×3 + pointwise 1×1, summed."""
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        super().__init__()
        self.gc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=g, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.gc(x) + self.pwc(x)


class DCAConv_v2(nn.Module):
    """
    DCAConv_v2 — Enhanced backbone feature enhancement for underwater small objects.

    Architecture:
        Input x:
          ├── Original DCAConv path (DualConv×2 → ChannelAttention)
          ├── Spatial Attention gate on original features (→ SA_gate * DualConv(x))
          ├── Multi-Scale Dilated Conv branch (dilation=2,3 → fuse)
          └── Progressive learnable fusion: output = x + original + β·spatial + γ·multi_scale

    Key improvements over DCAConv:
        1. Spatial Attention — highlights regions of interest in low-contrast underwater scenes,
           helping localize small objects (starfish, scallop, echinus).
        2. Multi-Scale Dilated Convolution — captures wider receptive fields at different
           dilation rates (2, 3), providing context for objects at varying distances/depths.
        3. Progressive Fusion — β and γ initialised to 0, so early training behaves
           identically to original DCAConv; gradients gradually activate the new branches.

    Inspired by:
        - CSPSL (Li et al., 2025): Feature preservation philosophy — the original
          residual path + spatial gate preserve fine-grained details for small objects.
        - MSDA (YOLOv11-MSE, 2025): Multi-scale dilated attention captures
          variable-sized underwater objects without extra parameters.
        - CBAM: Sequential attention mechanism shown to improve detection.
    """

    def __init__(self, c, g=4, dilation_rates=(2, 3), se_ratio=16):
        """
        Args:
            c: input/output channels (residual design maintains channel count)
            g: groups for DualConv grouped convolution
            dilation_rates: dilation rates for multi-scale branch
            se_ratio: reduction ratio for SE channel attention
        """
        super().__init__()

        # --- Original DCAConv path (preserved exactly) ---
        self.original = nn.Sequential(
            DualConv(c, c, 1, g),
            DualConv(c, c, 1, g),
            ChannelAttention_SE(c, se_ratio),
        )

        # --- Spatial Attention branch ---
        # Lightweight: 2-channel → 1-channel spatial weight map
        self.sa = SpatialAttention_Simple()
        # After SA gating, a single DualConv processes the attended features
        self.sa_proj = DualConv(c, c, 1, g)

        # --- Multi-Scale Dilated Convolution branch ---
        # Depth-wise dilated convs capture wider context without extra params
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(c, c, 3, padding=r, dilation=r, groups=c, bias=False)
            for r in dilation_rates
        ])
        # 1×1 fuse for multi-scale features
        self.ms_fuse = nn.Sequential(
            nn.Conv2d(c * len(dilation_rates), c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        # --- Progressive fusion weights ---
        # Initialized to 0: training starts with behaviour ≈ original DCAConv
        # Gradients gradually activate these branches as training progresses
        self.beta = nn.Parameter(torch.zeros(1))   # spatial attention branch weight
        self.gamma = nn.Parameter(torch.zeros(1))  # multi-scale branch weight

    def forward(self, x):
        # Original DCAConv path
        original_out = self.original(x)

        # Spatial attention: generate spatial weight map and apply gating
        sa_map = self.sa(x)                     # (B, 1, H, W)
        sa_out = self.sa_proj(x * sa_map)       # (B, C, H, W)

        # Multi-scale dilated: parallel depth-wise convs at different dilations
        ms_features = [conv(x) for conv in self.dilated_convs]  # list of (B, C, H, W)
        ms_out = self.ms_fuse(torch.cat(ms_features, dim=1))     # (B, C, H, W)

        # Progressive fusion
        # β ≈ 0, γ ≈ 0 at init → output ≈ x + original_out (same as original DCAConv)
        return x + original_out + self.beta * sa_out + self.gamma * ms_out
