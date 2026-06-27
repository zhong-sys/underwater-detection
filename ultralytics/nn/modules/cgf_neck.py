"""CGFusion + MKConv — Channel-Gated Fusion Neck (v2, with cross-channel mixing).

Replaces YOLO's standard FPN+PAN (Upsample+Concat+C3k2) with channel-gated
fusion and multi-kernel adaptive convolution.

v2 修复 (v1 跑不过 baseline 的根因):
    v1 为追求极致轻量，砍掉了 C3k2 的 Bottleneck 变换结构，导致每个融合点
    仅有 gate + 1×1 proj + DW conv — 无跨通道信息交换，无非线性深度。
    v2 在保留两个创新点的基础上，给 MKConv 加回 1×1 夹心结构（cv1→DW核选择→cv2），
    恢复 C3k2 级别的跨通道混合能力。

Innovations:
    1. CGFusion: 逐通道源门控（原始通道空间，内容感知）
       - 不同于 BiFPN: per-CHANNEL（不是 per-source scalar）
       - 不同于 SE: 门控在融合前（source-channel-aware）
    2. MKConv: 逐通道核选择（2核: 3×3 + dilated 3×3≈7×7）+ 1×1 夹心
       - 不同于标准 conv: 每通道自选感受野
       - 不同于 DCAv2 SA: 通道级核选择，非空间注意力

Orthogonality to DCAv2:
    DCAv2 CA:  WHICH channels to emphasize    (B, C, 1, 1)
    DCAv2 SA:  WHERE to enhance               (B, 1, H, W)
    CGFusion:  per-CHANNEL source gating      (B, total_C, 1, 1)  [pre-fusion]
    MKConv:    per-CHANNEL kernel selection   (B, K, 4, 1, 1)     [post-fusion]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


# ══════════════════════════════════════════════════════════════════════════════
# CGFusion — Per-channel source gating
# ══════════════════════════════════════════════════════════════════════════════

class CGFusion(nn.Module):
    """Lightweight Channel-Gated Fusion.

    Replaces YOLO's Concat. Learns per-channel gates that selectively weight
    each source channel before 1×1 fusion.

    v2 fix: gate initialized near-identity (bias=2 → sigmoid≈0.88),
    not 0.5× attenuation that starves downstream gradients.
    """

    def __init__(self, in_channels_list, out_channels: int):
        super().__init__()
        self.num_sources = len(in_channels_list)
        self.out_channels = out_channels
        total_in = sum(in_channels_list)

        # Gate: pool → MLP → sigmoid per-channel
        gate_hidden = max(total_in // 32, 16)
        self.gate_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_in, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, total_in),
        )
        # Near-identity init: sigmoid(2.0) ≈ 0.88, mild gating at start
        nn.init.constant_(self.gate_mlp[-1].bias, 2.0)

        self.sigmoid = nn.Sigmoid()

        # 1×1 projection: concat → output channels
        self.proj = Conv(total_in, out_channels, 1, 1)

    def forward(self, xs):
        """xs: list of feature maps, each (B, C_i, H_i, W_i)."""
        target_shape = xs[0].shape[-2:]

        aligned = []
        for x in xs:
            if x.shape[-2:] != target_shape:
                x = F.interpolate(x, size=target_shape, mode='bilinear',
                                  align_corners=False)
            aligned.append(x)

        concat = torch.cat(aligned, dim=1)           # (B, ΣC_i, H, W)

        gates = self.gate_mlp(concat)                # (B, ΣC_i)
        gates = self.sigmoid(gates)
        gates = gates.unsqueeze(-1).unsqueeze(-1)    # (B, ΣC_i, 1, 1)

        return self.proj(concat * gates)


# ══════════════════════════════════════════════════════════════════════════════
# MKConv — Per-channel kernel selection with cross-channel mixing
# ══════════════════════════════════════════════════════════════════════════════

class MKConv(nn.Module):
    """Multi-Kernel Convolution with CHANNEL-wise kernel selection.

    Replaces C3k2 blocks. Uses a 1×1 bottleneck sandwich (like C3k2) with
    the multi-kernel DW selection replacing the standard Bottleneck.

    v2 fix: added cv1(compress) + cv2(expand) 1×1 convs for cross-channel
    mixing. v1 had only DW conv (channel-isolated), which can't exchange
    information across channels — losing C3k2's core transformation capacity.
    """

    def __init__(self, channels: int, e: float = 0.5, act: bool = True,
                 reduction: int = 16):
        super().__init__()
        c_ = max(int(channels * e), 16)   # hidden channels (bottleneck)
        K = 2                              # kernel variants: small + large

        # 1×1 bottleneck: cross-channel mixing (same role as C3k2's cv1/cv2)
        self.cv1 = Conv(channels, c_, 1, 1)       # compress
        self.cv2 = Conv(c_, channels, 1, 1, act=False)  # expand (no act yet)

        # DW kernel branches (innovation — operates in bottleneck space)
        self.dw_small = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 1, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
        )
        self.dw_large = nn.Sequential(
            nn.Conv2d(c_, c_, 3, 1, 3, dilation=3, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
        )

        # Kernel selector (on bottleneck channels, further reducing cost)
        hidden = max(c_ // reduction, K * 4)
        self.kernel_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_ * K),
        )

        self.act_out = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        """x (B, C, H, W) → processed (B, C, H, W)."""
        identity = x
        x = self.cv1(x)                              # compress: C → c_

        # Channel-wise kernel selection (K=2: small + large)
        B, C_bn = x.shape[:2]
        K = 2
        kernel_w = self.kernel_selector(x)           # (B, c_*K)
        kernel_w = kernel_w.view(B, K, C_bn, 1, 1)   # (B, K, c_, 1, 1)
        kernel_w = F.softmax(kernel_w, dim=1)

        # DW processing with kernel selection
        dw_out = (kernel_w[:, 0] * self.dw_small(x) +
                  kernel_w[:, 1] * self.dw_large(x))
        x = x + dw_out                               # residual in bottleneck

        x = self.cv2(x)                              # expand: c_ → C
        return self.act_out(x + identity)             # residual at output
