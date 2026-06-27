"""SEE — Scattering Edge Enhancement for underwater object detection.

Physical motivation:
    Underwater light degradation has TWO independent mechanisms:
    1. Absorption (Beer-Lambert law): wavelength-dependent attenuation
       → channel responses selectively degraded → DCAv2 (channel domain)
    2. Scattering (Mie/Rayleigh): direction-dependent blur
       → spatial edges and textures lost → SEE (spatial domain)

    DCAv2 and SEE are orthogonal: DCAv2 selects which channels to emphasize,
    SEE enhances spatial structure regardless of channel selection.

Why fixed Sobel (not learned):
    Learned edge detectors can "forget" to detect edges when gradients are weak
    (the same failure mode as CAFE/CFD). Fixed Sobel kernels are deterministic
    — they always provide edge information. The learnable part only decides
    how much to enhance at each position, not whether edges exist.

Architecture:
    X → mean(dim=1) → x_gray                    (pseudo-intensity)
    x_gray → Sobel_x, Sobel_y → [ex, ey]          (fixed, 0 params)
    [ex, ey] → Conv1x1(2→8→1) → Sigmoid → E      (learnable refinement)
    γ = sigmoid(gate), gate init -5
    output = X × (1 + γ·E)                        (progressive enhancement)

    - ~0.001M parameters
    - Applied on P3/P4/P5 in backbone (same positions as DCAv2)
    - Progressive gate ensures safe cold start (epoch 0: ≈ identity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEE(nn.Module):
    """Scattering Edge Enhancement — spatial edge sharpening via fixed Sobel.

    Complements DCAv2 (channel attention, Beer-Lambert physics) by addressing
    the second component of underwater light degradation: scattering-induced
    edge blur. Operates purely in the spatial domain with fixed edge detectors.

    Args:
        channels: input/output channel count.
    """

    def __init__(self, channels: int):
        super().__init__()

        # Fixed 3×3 Sobel kernels — deterministic, never degenerate
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("sx", sx.view(1, 1, 3, 3))
        self.register_buffer("sy", sy.view(1, 1, 3, 3))

        # Edge refinement: 2-ch (edge_x, edge_y) → 1-ch edge attention map
        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

        # Progressive gate: init -5 → sigmoid ≈ 0.007 → SEE ≈ identity at epoch 0
        self.gate = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: edge-enhanced features (B, C, H, W)."""
        # Pseudo-intensity: channel-mean
        x_gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Fixed Sobel edge detection (no learnable params)
        ex = F.conv2d(x_gray, self.sx, padding=1)  # horizontal edges
        ey = F.conv2d(x_gray, self.sy, padding=1)  # vertical edges

        # Edge attention map ∈ [0, 1]
        e = self.refine(torch.cat([ex, ey], dim=1))  # (B, 1, H, W)

        # Progressive enhancement
        # epoch 0:  gate ≈ 0 → output ≈ x (safe, identity)
        # training: gate ↑ → output = x × (1 + e) at edge positions
        gamma = self.gate.sigmoid()
        return x * (1.0 + gamma * e)
