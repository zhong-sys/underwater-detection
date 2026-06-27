"""DCAConv_v3 -- DCAv2 with MSDC removed for multi-module compatibility.

Motivation:
    DCAv2's Multi-Scale Dilated Convolution (MSDC) branch expands receptive
    field in the backbone. When co-deployed with MSRF (which also expands
    receptive field at P3), the two compete for the same gradient space.

    DCAv3 removes MSDC, keeping channel attention and spatial attention.
    All multi-scale receptive-field work is delegated to MSRF.

Architecture (vs DCAv2):
    DCAv2:  x + original + beta*SA + gamma*MSDC
    DCAv3:  x + original + beta*SA
"""

import torch
import torch.nn as nn


class ChannelAttention_SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention_Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        super().__init__()
        self.gc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, groups=g, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.gc(x) + self.pwc(x)


class DCAConv_v3(nn.Module):
    """DCAv2 minus MSDC -- channel + spatial attention only.

    Keeps:
        - Channel attention (SE-style, core Beer-Lambert mechanism)
        - Spatial attention (mean+max -> 7x7 conv -> sigmoid)
        - Progressive beta gate for SA branch
    Removes:
        - Multi-Scale Dilated Convolution (MSDC) branch (MSRF handles this)
        - gamma parameter (no longer needed)

    Args:
        c: input/output channels.
        g: groups for DualConv.
        se_ratio: reduction ratio for SE channel attention.
    """

    def __init__(self, c, g=4, se_ratio=16):
        super().__init__()

        self.original = nn.Sequential(
            DualConv(c, c, 1, g),
            DualConv(c, c, 1, g),
            ChannelAttention_SE(c, se_ratio),
        )

        self.sa = SpatialAttention_Simple()
        self.sa_proj = DualConv(c, c, 1, g)

        # Progressive gate for SA branch (init 0 -> identity at epoch 0)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        original_out = self.original(x)

        sa_map = self.sa(x)
        sa_out = self.sa_proj(x * sa_map)

        return x + original_out + self.beta * sa_out
