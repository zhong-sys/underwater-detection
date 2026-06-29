import torch
import torch.nn as nn


class LCFE(nn.Module):
    """
    Local Contrast Foreground Enhancement.

    LCFE is designed for underwater object detection where weak boundaries,
    low contrast, turbidity, and background similarity often cause missed
    detections. It enhances local contrast cues and weak foreground regions
    with a lightweight residual design.
    """

    def __init__(self, channels: int, k: int = 7):
        super().__init__()

        self.local_smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.contrast_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=k, padding=k // 2, bias=True),
            nn.Sigmoid(),
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        smooth = self.local_smooth(x)
        contrast_feat = x - smooth

        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        contrast_map = torch.mean(torch.abs(contrast_feat), dim=1, keepdim=True)

        gate = self.spatial_gate(torch.cat([avg_map, max_map, contrast_map], dim=1))

        enhanced = self.contrast_proj(contrast_feat)

        return x + self.scale * enhanced * gate
