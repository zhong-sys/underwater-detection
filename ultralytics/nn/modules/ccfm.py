"""CCFM: Clarity-Confidence Fusion Module for underwater FPN skips.

The module is a drop-in replacement for Concat at top-down FPN skip
connections. It estimates a local clarity map from shallow backbone features
using mean, standard deviation, and fixed Sobel edge magnitude. The clarity map
then guides a two-path fidelity fusion:

    shallow_out = x + alpha * s * detail - beta * (1 - s) * noise

where detail is a fixed high-frequency residual and noise is a fixed local
low-frequency residual. This keeps reliable edges/textures while suppressing
smooth turbid interference before cross-scale concatenation.

Complementarity to DCAv2:
    DCAv2 enhances backbone feature quality in the channel domain.
    CCFM controls cross-scale skip reliability in the neck, reducing noisy
    shallow-feature injection while preserving clear edges and textures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCFM(nn.Module):
    """Clarity-confidence fusion for FPN skip connections.

    Args:
        c_shallow: channels of the shallow backbone feature.
        c_deep: channels of the deep neck feature.
        max_strength: maximum detail/noise residual strength.
    """

    def __init__(self, c_shallow: int, c_deep: int, max_strength: float = 0.5):
        super().__init__()
        self.max_strength = max_strength

        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        blur = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        self.register_buffer("blur", blur)

        self.clarity = nn.Sequential(
            nn.Conv2d(3, 8, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 1, bias=True),
            nn.Sigmoid(),
        )

        # Clarity starts at 0.5 everywhere. The residual strengths below start
        # small but non-zero, so the module stays near Concat while gradients
        # can still reach the clarity predictor.
        nn.init.zeros_(self.clarity[2].weight)
        nn.init.zeros_(self.clarity[2].bias)

        # sigmoid(-3.891) ~= 0.02, so each branch initially contributes about
        # 1% of its residual when max_strength=0.5 and clarity=0.5.
        self.alpha_logit = nn.Parameter(torch.tensor(-3.891))
        self.beta_logit = nn.Parameter(torch.tensor(-3.891))

    def forward(self, data):
        """Fuse (x_deep, x_shallow) and return a concatenated feature map."""
        x_deep, x_shallow = data

        if x_deep.shape[-2:] != x_shallow.shape[-2:]:
            x_deep = F.interpolate(
                x_deep, size=x_shallow.shape[-2:], mode="nearest"
            )

        gray = x_shallow.mean(dim=1, keepdim=True)
        mean = gray
        std = x_shallow.std(dim=1, keepdim=True, unbiased=False)

        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)
        edge = torch.log1p(edge)

        clarity = self.clarity(torch.cat([mean, std, edge], dim=1))

        # Fixed detail/noise decomposition:
        # - detail: high-frequency residual, expected to preserve object edges.
        # - noise: local low-frequency deviation, expected to capture smooth
        #   scattering/background fluctuation for conservative suppression.
        shallow_blur = F.avg_pool2d(x_shallow, kernel_size=3, stride=1, padding=1)
        detail = x_shallow - shallow_blur
        noise = shallow_blur - x_shallow.mean(dim=(2, 3), keepdim=True)

        alpha = self.max_strength * self.alpha_logit.sigmoid()
        beta = self.max_strength * self.beta_logit.sigmoid()
        shallow_out = x_shallow + alpha * clarity * detail - beta * (1.0 - clarity) * noise

        return torch.cat([x_deep, shallow_out], dim=1)
