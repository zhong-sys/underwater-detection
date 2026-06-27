"""CAFE — Contrast-Aware Feature Enhancement for underwater object detection.

Problem:
    Standard channel attention (SE: GAP→FC→Sigmoid) compresses H×W into a single
    scalar per channel. In underwater scenes, light scattering degrades feature
    contrast — activations become nearly uniform spatially. GAP loses the
    information that distinguishes object channels from background channels.
    The sigmoid output tends toward 0.5 for all channels, providing no discrimination.

    Prior work (MCA, FcaNet, GSoP) proposes generic alternatives to GAP (moments,
    DCT, covariance), but these are task-agnostic — they use the same statistic
    regardless of scene conditions.

Key insight:
    The degradation severity varies across underwater scenes (clear oceanic vs
    turbid coastal). A fixed fusion of GAP and variance is suboptimal:
    - Clear scenes: GAP already works well, variance adds noise
    - Turbid scenes: GAP fails, variance is essential

    CAFE estimates per-scene degradation from the GAP and variance statistics
    themselves, then dynamically adjusts the fusion ratio. This is fundamentally
    different from fixed-ratio approaches (MCA, FcaNet) which treat all scenes
    identically.

Architecture:
    X → GAP → fc_se ──────────→ w_se
    X → Var  → fc_var ─────────→ w_var
    X → [GAP, Var] → degrad_est → d ∈ [-1, 1]  (degradation score)
    α = sigmoid(α_base + β·d)                   (dynamic gate)
    w = (1-α)·w_se + α·w_var
    output = X · sigmoid(w)

Reference:
    - Hu et al. (2018 CVPR): Squeeze-and-Excitation Networks
    - Qin et al. (2021 ICCV): FcaNet — DCT-based channel attention
    - Gao et al. (2019 CVPR): GSoP — second-order pooling
    - Lee et al. (2024): MCA — Moment Channel Attention
    - CAFE differs: scene-adaptive fusion via learned degradation estimator
"""

import torch
import torch.nn as nn


class CAFE(nn.Module):
    """Contrast-Aware Feature Enhancement with dynamic degradation estimation.

    Unlike MCA/FcaNet which use a fixed fusion of GAP and higher-order statistics,
    CAFE estimates per-scene degradation severity from feature statistics and
    dynamically adjusts the GAP-variance fusion ratio.

    Key design elements:
        1. Degradation estimator: (GAP, Var) → d ∈ [-1, 1]
           d > 0 → turbid scene → more variance reliance
           d < 0 → clear scene → more GAP reliance (standard SE)
        2. β init 0 → degradation modulation activates progressively
        3. α_base init -5 → sigmoid ≈ 0.007 → CAFE ≈ SE at epoch 0

    Args:
        channels: input/output channel count.
        reduction: bottleneck ratio for FC layers.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Standard SE path
        self.fc_se = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

        # Variance path
        self.fc_var = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

        # Base gate (init ≈ 0 → CAFE ≈ SE at epoch 0)
        self.alpha_base = nn.Parameter(torch.tensor(-5.0))

        # Degradation estimator: [GAP_vec, Var_vec] → degradation score ∈ [-1, 1]
        # The linear layers map 2C → hidden → 1
        self.degrad_estim = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

        # Degradation modulation strength (init 0 → safe, no modulation at start)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, C, H, W). Returns: channel-recalibrated features (B, C, H, W)."""
        B, C, H, W = x.shape

        # ---- 1. Standard SE: global average pooling ----
        gap = self.gap(x)          # (B, C, 1, 1)
        w_se = self.fc_se(gap)     # (B, C, 1, 1)

        # ---- 2. Variance: Var[X] = E[X²] - E[X]² ----
        x_mean = x.mean(dim=[-2, -1], keepdim=True)
        x_sq_mean = (x * x).mean(dim=[-2, -1], keepdim=True)
        var = x_sq_mean - x_mean * x_mean
        w_var = self.fc_var(var)   # (B, C, 1, 1)

        # ---- 3. Scene-level degradation estimation ----
        # GAP captures DC component, variance captures AC spread.
        # Their joint distribution encodes degradation severity:
        #   clear scene:  distinct GAP per channel + moderate variance
        #   turbid scene: compressed GAP range  + suppressed variance
        d = self.degrad_estim(
            torch.cat([gap.view(B, C), var.view(B, C)], dim=1)
        )  # (B, 1)

        # ---- 4. Dynamic gate: degradation-adaptive fusion ----
        # alpha_base init -5 → sigmoid ≈ 0.007 (SE-dominant)
        # beta init 0         → no modulation (safe start)
        # As beta activates: d > 0 (turbid) → alpha increases → more variance
        #                   d < 0 (clear)  → alpha decreases → more GAP
        alpha = (self.alpha_base + self.beta * d).sigmoid()  # (B, 1)
        alpha = alpha.view(B, 1, 1, 1)

        # ---- 5. Fused channel weights ----
        w = (1.0 - alpha) * w_se + alpha * w_var
        w = w.sigmoid()

        return x * w
