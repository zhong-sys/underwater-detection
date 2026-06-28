"""DCAConv_v3 lightweight channel enhancement.

DCAConv_v3 is a lightweight channel-enhancement module derived from DCAv2.
It removes both spatial attention and multi-scale dilated convolution to avoid
functional overlap with MSRF.
DCAv3 focuses on photometric/channel degradation caused by underwater absorption.
MSRF handles spatial receptive-field compensation for small objects.
"""

import torch
import torch.nn as nn


class ChannelAttention_SE(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        hidden = max(channel // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        super().__init__()
        g = min(g, in_channels, out_channels)
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
    """DCAv2-derived channel enhancement without spatial branches."""
    def __init__(self, c, g=4, se_ratio=16):
        super().__init__()
        self.original = nn.Sequential(
            DualConv(c, c, 1, g),
            DualConv(c, c, 1, g),
            ChannelAttention_SE(c, se_ratio),
        )

    def forward(self, x):
        original_out = self.original(x)
        return x + original_out
