import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
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

class DualConv(nn.Module):
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

class DCAConv(nn.Module):
    def __init__(self, c, g=4):
        super().__init__()
        self.m = nn.Sequential(
            DualConv(c, c, 1, g),
            DualConv(c, c, 1, g),
            ChannelAttention(c)
        )

    def forward(self, x):
        return x + self.m(x)