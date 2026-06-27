"""CS_HFCM -- Cross-Scale High-Frequency Compensation Module.

Replaces standard Concat at FPN skip connections with edge-aware cross-scale
detail injection.

Motivation:
    FPN uses nearest-neighbor upsampling to coarsen deep semantic features, then
    concatenates them with backbone shallow features via skip connections.  But
    standard Concat treats ALL spatial positions in the shallow features equally
    -- including turbid-water noise and backscatter artifacts.

    CS_HFCM extracts high-frequency edges from backbone shallow features using a
    FIXED Laplacian kernel (deterministic, no gradient competition) and injects
    only the edge-rich regions into the neck. Turbid-water smooth regions in the
    shallow features are attenuated; object boundaries are preserved.

    This is TRUE cross-scale: shallow backbone features (high resolution, rich
    texture) are filtered and injected into deep neck features (low resolution,
    semantic but blurred).

Architecture:
    Input: (x_deep, x_shallow) -- neck upsampled + backbone skip
    1. Laplacian on channel-mean of x_shallow -> edge map (fixed, non-learnable)
    2. edge map -> spatial gate (1ch, channel-shared) -> g in [0,1]
    3. detail = x_shallow * g  (edge regions pass, smooth regions suppressed)
    4. alpha = sigmoid(gate_param), init alpha ~ 0.007
    5. shallow_out = (1-alpha)*x_shallow + alpha*detail  (progressive)
    6. output = concat(x_deep, shallow_out)  (same shape as standard Concat)

Key properties:
    - Laplacian kernel: fixed, center-negative, natural high-pass -- no BN.
    - Spatial gate: 1-channel, shared across all channels -- ~0.001M params.
    - Progressive alpha: epoch 0 -> CS_HFCM = standard Concat (safe cold start).
    - Drop-in replacement for Concat at FPN skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CS_HFCM(nn.Module):
    """Cross-Scale High-Frequency Compensation Module.

    Drop-in replacement for Concat at FPN skip connections. Enhances backbone
    shallow features with fixed Laplacian edge extraction before concatenation
    with neck deep features. Turbid-water smooth regions are suppressed via
    a learned spatial gate; object boundaries pass through.

    Args:
        c_shallow: channels of backbone shallow feature (second input).
        c_deep:    channels of neck deep feature (first input).
    """

    def __init__(self, c_shallow: int, c_deep: int):
        super().__init__()

        # Fixed 3x3 Laplacian kernel (center-negative, sum=0, perfect high-pass)
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("lap", lap.view(1, 1, 3, 3))

        # Spatial gate: edge magnitude -> reliability (channel-shared, 1ch in/out)
        self.gate = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

        # Progressive injection: epoch 0 -> pure shallow (identity = standard Concat)
        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, data):
        """Args: data = (x_deep, x_shallow). Returns concatenated features."""
        x_deep, x_shallow = data

        # 1. Fixed Laplacian edge extraction (no learnable params, no BN)
        x_gray = x_shallow.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        edge = F.conv2d(x_gray, self.lap, padding=1)     # (B, 1, H, W)

        # 2. Spatial gate: where are the reliable edges?
        g = self.gate(edge.abs())                         # (B, 1, H, W)

        # 3. Edge-filtered shallow features
        detail = x_shallow * g

        # 4. Progressive blend: identity -> edge-filtered
        #    epoch 0: gamma ~ 0.007 -> shallow_out ~ x_shallow (standard Concat)
        #    trained: gamma ~ 1    -> shallow_out ~ detail   (edge-injected)
        gamma = self.alpha.sigmoid()
        shallow_out = (1.0 - gamma) * x_shallow + gamma * detail

        # 5. Concatenate (same output shape as standard Concat)
        return torch.cat([x_deep, shallow_out], dim=1)
