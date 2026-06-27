"""CAG-DCN -- Contrast-Aware Gated Deformable Convolution for underwater detection.

Problem with standard DCN underwater:
    Standard DCN computes offsets purely from features: offset = f_theta(x).  In
    underwater scenes, turbid regions produce noisy features from backscatter
    and suspended particles -- these generate SPURIOUS offsets that warp the
    sampling grid toward noise patterns rather than object boundaries.

Key insight:
    Local contrast (spatial mean/std of channel activations) is a proxy for
    water clarity at each position:
    - HIGH local contrast -> likely object boundary or textured surface -> offsets reliable
    - LOW  local contrast -> likely turbid water or uniform background -> offsets unreliable

    CAG-DCN uses a contrast clarity estimator to GATE the learned offsets:
    - Clear regions: full offset range -> aggressive deformable sampling
    - Turbid regions: suppressed offsets -> conservative regular sampling

Complementarity:
    DCAv2:  photometric degradation -> channel attention (intensity domain)
    CAG-DCN: geometric degradation with contrast awareness (spatial domain)
    CAL:     optimization bias -> contrast-weighted loss (gradient domain)

    Three modules, three dimensions, one root cause: underwater contrast degradation.

Architecture:
    X -> OffsetFeat(3x3dw+1x1) -> offset              (standard DCN)
    X -> ModFeat(3x3dw+1x1) -> mask                    (DCNv2 modulation)
    X -> DeformConv(offset, mask) -> expand            (pure DCN, no gate)
    X -> mean/std -> ClarityEst(2->8->1) -> s in [0,1] (local clarity)
    scale = 0.5 + 0.5*s in [0.5, 1.0]
    output = X + DCN(X) * scale

Reference:
    - Dai et al. (2017 ICCV): DCNv1
    - Zhu et al. (2019 CVPR): DCNv2 -- modulated deformable convolution
    - CAG-DCN differs: contrast-gated offset modulation for underwater robustness
"""

import torch
import torch.nn as nn
import torchvision.ops


class CAGDCN(nn.Module):
    """Contrast-Aware Gated Deformable Convolution.

    Extends standard DCN with a local contrast clarity gate that suppresses
    unreliable offsets in turbid underwater regions. The clarity estimator
    uses local mean and standard deviation of channel activations as a proxy
    for water quality at each spatial position.

    Args:
        channels: input/output channel count.
    """

    def __init__(self, channels: int):
        super().__init__()
        reduced = max(channels // 2, 128)

        # Channel reduction/expansion
        self.reduce = nn.Conv2d(channels, reduced, 1, bias=False)
        self.expand = nn.Sequential(
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        # Offset predictor: depthwise 3x3 (local context) -> 1x1 (cross-channel)
        self.offset_feat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, 18, 1, bias=True),
        )

        # Modulation predictor (DCNv2-style)
        self.mod_feat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, 9, 1, bias=True),
        )

        # Contrast clarity estimator (the innovation)
        # 2-ch input: local channel-mean and channel-std -> clarity in [0, 1]
        self.clarity = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

        # Deformable conv weight
        self.deform_weight = nn.Parameter(torch.randn(reduced, reduced, 3, 3) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: geometry-enhanced features (B, C, H, W)."""
        x_red = self.reduce(x)

        # 1. Standard DCN offset and modulation (UNCHANGED)
        offset = self.offset_feat(x)            # (B, 18, H, W)
        mask = self.mod_feat(x).sigmoid()        # (B,  9, H, W)

        # 2. Deformable convolution (pure DCN, no interference)
        x_deform = torchvision.ops.deform_conv2d(
            input=x_red,
            offset=offset,
            weight=self.deform_weight,
            mask=mask,
            stride=1,
            padding=1,
        )

        x_deform = self.expand(x_deform)

        # 3. Contrast-aware residual scale (AFTER DCN, mild modulation)
        #    Clear region:  scale ~ 1.0  -> full DCN contribution
        #    Turbid region: scale ~ 0.5  -> half DCN contribution (not zero!)
        #    Init:          scale ~ 0.75 -> DCN retains 75% effectiveness
        mean = x.mean(dim=1, keepdim=True)       # (B, 1, H, W)
        std = x.std(dim=1, keepdim=True)         # (B, 1, H, W)
        s = self.clarity(torch.cat([mean, std], dim=1))  # (B, 1, H, W)
        scale = 0.5 + 0.5 * s                    # range [0.5, 1.0]

        return x + x_deform * scale
