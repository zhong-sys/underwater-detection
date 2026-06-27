"""BL-ASFF: Beer-Lambert Adaptive Spatial Feature Fusion.

Extends ASFF (Liu et al. 2019, 2000+ citations) with per-level transmission
estimation and Beer-Lambert channel compensation before cross-scale fusion.

Key design (v3 — lightweight anti-overfitting):
- Residual: output = aligned_target + fusion_scale * cross_fused
  Target level features preserved via residual; cross-scale info added as
  a gated refinement (fusion_scale init 0 → identity at epoch 0).
- Cross-fusion only on NON-target levels (2 levels), preventing double-counting
  with the residual path. weight_conv predicts 2-channel softmax weights.
- Scalar gate replaces heavy out_conv (DW+PW+BN): reduces ~92K params (13%)
  and constrains model capacity to prevent overfitting.
- lambda init 0 → Beer-Lambert compensation activates progressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.a2fm import TransmissionEstimator


class BLASFF(nn.Module):
    """Beer-Lambert Adaptive Spatial Feature Fusion (lightweight v3).

    Pipeline per level i in {0,1,2} (P3,P4,P5):
        1. t_i = TransEstim(feat_i)               — per-channel transmission
        2. feat_i' = feat_i * (1 + lambda_i * (1-t_i)) — BL compensation
        3. Resize to target spatial size + channel-align (1x1 Conv)
        4. Predict 2-channel spatial softmax weights (non-target levels only)
        5. Cross-fused = weighted sum of non-target aligned features
        6. Residual: output = aligned_target + fusion_scale * cross_fused
           fusion_scale init 0 → output = aligned_target at epoch 0
    """

    def __init__(self, in_channels: list[int], target_level: int = 0, init_gate: float = -1.0):
        """Args:
            in_channels: [c3, c4, c5]
            target_level: 0=P3, 1=P4, 2=P5
            init_gate: initial value for fusion gate logit.
                       -3.0 (sigmoid≈0.05): conservative, preserve fine details (P3)
                       -1.0 (sigmoid≈0.27): moderate cross-scale fusion (P4, P5)
        """
        super().__init__()
        assert len(in_channels) == 3
        assert 0 <= target_level <= 2

        self.target_level = target_level
        c_out = in_channels[target_level]

        # Indices of the OTHER two levels (for cross-fusion)
        self.cross_indices = [i for i in range(3) if i != target_level]

        # Per-level transmission estimators
        self.trans_estim = nn.ModuleList([
            TransmissionEstimator(c) for c in in_channels
        ])

        # Per-level Beer-Lambert compensation strength (init 0 -> identity)
        self.lmbda = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in in_channels
        ])

        # Per-level channel alignment (-> c_out)
        self.align = nn.ModuleList([
            nn.Conv2d(c, c_out, 1) for c in in_channels
        ])

        # Spatial weight predictor for non-target levels only
        # Input: concat of 3 aligned features. Output: 2-channel softmax.
        self.weight_conv = nn.Sequential(
            nn.Conv2d(c_out * 3, c_out, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, 2, 1),
        )
        # Soft prior: uniform weights across the 2 non-target levels
        nn.init.constant_(self.weight_conv[-1].bias, 0.0)

        # Learnable gate for cross-scale fusion strength (sigmoid → [0,1])
        # init_gate=-3: sigmoid≈0.05 (conservative, for P3 fine details)
        # init_gate=-1: sigmoid≈0.27 (moderate, for P4/P5)
        self.fusion_gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """x = [P3, P4, P5] from FPN+PAN neck. Returns fused feature at target scale."""
        assert len(x) == 3

        # 1. Per-level transmission estimation + Beer-Lambert compensation
        compensated = []
        for i, feat in enumerate(x):
            t = self.trans_estim[i](feat)
            compensated.append(feat * (1.0 + self.lmbda[i] * (1.0 - t)))

        # 2. Resize to target spatial size + channel alignment
        target_size = compensated[self.target_level].shape[-2:]
        aligned = []
        for i, feat in enumerate(compensated):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(self.align[i](feat))

        # 3. Weight prediction for non-target levels only -> 2-channel softmax
        cross_weights = self.weight_conv(torch.cat(aligned, dim=1))  # (B, 2, H, W)
        cross_weights = torch.softmax(cross_weights, dim=1)

        # 4. Cross-fusion: weighted sum of non-target aligned features only
        cross_fused = sum(
            cross_weights[:, j:j + 1] * aligned[i]
            for j, i in enumerate(self.cross_indices)
        )

        # 5. Residual: target-level feature + gated cross-scale refinement
        #    sigmoid(fusion_gate) → [0,1] controls cross-scale contribution
        gate = torch.sigmoid(self.fusion_gate)
        return aligned[self.target_level] + gate * cross_fused
