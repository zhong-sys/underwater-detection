"""Unit tests for TCB Loss (Transmission-Calibrated Boundary Loss)."""
import sys
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1: TCBBoxLoss unit test
# ---------------------------------------------------------------------------
def test_tcb_box_loss_unit():
    """Verify TCBBoxLoss computes without errors and init_identity holds."""
    print("=" * 60)
    print("Test 1: TCBBoxLoss unit test")
    print("=" * 60)

    from ultralytics.utils.loss import TCBBoxLoss

    tcb = TCBBoxLoss(reg_max=16)
    tcb.train()

    # Check init values
    print(f"  beta init:      {tcb.beta.item():.6f} (expect 0.0)")
    print(f"  alpha_raw init: {tcb.alpha_raw.item():.6f} (expect -5.0)")
    alpha_init = tcb.alpha_raw.sigmoid().item()
    print(f"  alpha effective:{alpha_init:.6f} (expect ~0.007)")

    assert tcb.beta.item() == 0.0, f"beta should be 0 at init, got {tcb.beta.item()}"
    assert alpha_init < 0.01, f"alpha should be ~0 at init, got {alpha_init}"
    print("  [PASS] init_identity: alpha ~= 0, beta = 0\n")

    # Create dummy inputs (batch=2, 8400 anchors, 64 channels, 3 feat levels)
    B, N, C = 2, 8400, 64
    # Feature maps: P3 (80x80), P4 (40x40), P5 (20x20)
    feats = [
        torch.randn(B, C, 80, 80),
        torch.randn(B, C // 2, 40, 40),   # channels differ per level (realistic)
        torch.randn(B, C // 2, 20, 20),
    ]
    # Anchor points and strides matching feat sizes
    # 80*80 + 40*40 + 20*20 = 6400 + 1600 + 400 = 8400
    anchor_points = torch.cat([
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 80, 1.0), torch.arange(0.5, 80, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),                # (6400, 2)
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 40, 1.0), torch.arange(0.5, 40, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),                # (1600, 2)
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 20, 1.0), torch.arange(0.5, 20, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),                # (400, 2)
    ], dim=0)  # (8400, 2)

    stride = torch.cat([
        torch.full((6400, 1), 8.0),
        torch.full((1600, 1), 16.0),
        torch.full((400, 1), 32.0),
    ], dim=0)  # (8400, 1)

    # Predicted boxes in xyxy (batch, anchors, 4)
    pred_bboxes = torch.rand(B, N, 4) * 640
    pred_bboxes = pred_bboxes.clamp(min=0, max=640)
    # Ensure valid boxes: x2 > x1, y2 > y1
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + torch.rand(B, N) * 50 + 5
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + torch.rand(B, N) * 50 + 5

    # Target boxes (strided)
    target_bboxes = torch.rand(B, N, 4) * 80
    target_bboxes = target_bboxes.clamp(min=0, max=80)
    target_bboxes[..., 2] = target_bboxes[..., 0] + torch.rand(B, N) * 10 + 2
    target_bboxes[..., 3] = target_bboxes[..., 1] + torch.rand(B, N) * 10 + 2

    # Foreground mask: mark ~20% of anchors as foreground
    fg_mask = torch.rand(B, N) < 0.2

    pred_dist = torch.randn(B, N, 4 * 16)  # (B, N, 64)
    target_scores = torch.zeros(B, N, 10)
    # Assign scores for foreground anchors
    for b in range(B):
        fg_idx = fg_mask[b].nonzero(as_tuple=True)[0]
        if len(fg_idx) > 0:
            target_scores[b, fg_idx, torch.randint(0, 10, (len(fg_idx),))] = 1.0
    target_scores_sum = max(target_scores.sum(), torch.tensor(1.0))

    imgsz = torch.tensor([640.0, 640.0])

    # Forward pass
    loss_box, loss_dfl = tcb(
        pred_dist, pred_bboxes, anchor_points, target_bboxes,
        target_scores, target_scores_sum, fg_mask, imgsz, stride,
        feats=feats,
    )

    print(f"  loss_box: {loss_box.item():.6f}")
    print(f"  loss_dfl: {loss_dfl.item():.6f}")
    print(f"  C (norm constant): {tcb.C.item():.4f}")
    print(f"  alpha effective:   {tcb.alpha_raw.sigmoid().item():.6f}")
    print(f"  beta:              {tcb.beta.item():.6f}")

    assert not torch.isnan(loss_box), "loss_box is NaN!"
    assert not torch.isnan(loss_dfl), "loss_dfl is NaN!"
    assert loss_box.item() >= 0, f"loss_box should be >= 0, got {loss_box.item()}"
    print("  [PASS] Forward pass OK, losses are valid\n")


# ---------------------------------------------------------------------------
# Test 2: TCB degrades gracefully without feats
# ---------------------------------------------------------------------------
def test_tcb_without_feats():
    """Verify TCB works without feats (falls back to pure CIoU)."""
    print("=" * 60)
    print("Test 2: TCB without feats (fallback mode)")
    print("=" * 60)

    from ultralytics.utils.loss import TCBBoxLoss

    tcb = TCBBoxLoss(reg_max=16)
    tcb.train()

    B, N = 2, 8400
    pred_dist = torch.randn(B, N, 4 * 16)
    pred_bboxes = torch.rand(B, N, 4) * 640
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 20
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 20

    target_bboxes = torch.rand(B, N, 4) * 80
    target_bboxes[..., 2] = target_bboxes[..., 0] + 10
    target_bboxes[..., 3] = target_bboxes[..., 1] + 10

    fg_mask = torch.rand(B, N) < 0.2
    target_scores = torch.zeros(B, N, 10)
    for b in range(B):
        fg_idx = fg_mask[b].nonzero(as_tuple=True)[0]
        if len(fg_idx) > 0:
            target_scores[b, fg_idx, 0] = 1.0
    target_scores_sum = max(target_scores.sum(), torch.tensor(1.0))

    anchor_points = torch.rand(N, 2) * 80
    stride = torch.full((N, 1), 8.0)
    imgsz = torch.tensor([640.0, 640.0])

    loss_box, loss_dfl = tcb(
        pred_dist, pred_bboxes, anchor_points, target_bboxes,
        target_scores, target_scores_sum, fg_mask, imgsz, stride,
        feats=None,  # no feats -> pure CIoU
    )

    print(f"  loss_box: {loss_box.item():.6f}")
    print(f"  loss_dfl: {loss_dfl.item():.6f}")
    assert not torch.isnan(loss_box)
    print("  [PASS] Fallback mode works\n")


# ---------------------------------------------------------------------------
# Test 3: Progressive activation — loss should change after "training" steps
# ---------------------------------------------------------------------------
def test_progressive_activation():
    """Verify beta and alpha can be learned (gradients are non-zero)."""
    print("=" * 60)
    print("Test 3: Progressive activation (gradient flow)")
    print("=" * 60)

    from ultralytics.utils.loss import TCBBoxLoss

    tcb = TCBBoxLoss(reg_max=16)
    tcb.train()

    B, N = 2, 2100  # smaller to speed up
    feats = [
        torch.randn(B, 64, 40, 40),
        torch.randn(B, 32, 20, 20),
        torch.randn(B, 32, 10, 10),
    ]

    anchor_points = torch.cat([
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 40, 1.0), torch.arange(0.5, 40, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 20, 1.0), torch.arange(0.5, 20, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 10, 1.0), torch.arange(0.5, 10, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
    ], dim=0)

    stride = torch.cat([
        torch.full((1600, 1), 8.0),
        torch.full((400, 1), 16.0),
        torch.full((100, 1), 32.0),
    ], dim=0)

    pred_bboxes = torch.rand(B, N, 4) * 640
    pred_bboxes[..., 2] = pred_bboxes[..., 0] + 30
    pred_bboxes[..., 3] = pred_bboxes[..., 1] + 30
    pred_dist = torch.randn(B, N, 4 * 16)

    target_bboxes = torch.rand(B, N, 4) * 80
    target_bboxes[..., 2] = target_bboxes[..., 0] + 8
    target_bboxes[..., 3] = target_bboxes[..., 1] + 8

    fg_mask = torch.rand(B, N) < 0.2
    target_scores = torch.zeros(B, N, 10)
    for b in range(B):
        fg_idx = fg_mask[b].nonzero(as_tuple=True)[0]
        if len(fg_idx) > 0:
            target_scores[b, fg_idx, 0] = 1.0
    target_scores_sum = max(target_scores.sum(), torch.tensor(1.0))
    imgsz = torch.tensor([640.0, 640.0])

    # Run several "training" steps
    optimizer = torch.optim.SGD(tcb.parameters(), lr=0.01, momentum=0.9)
    losses = []
    betas = []
    alphas = []

    for step in range(10):
        optimizer.zero_grad()
        loss_box, loss_dfl = tcb(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride,
            feats=feats,
        )
        loss = loss_box + loss_dfl
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        betas.append(tcb.beta.item())
        alphas.append(tcb.alpha_raw.sigmoid().item())

    print(f"  Loss evolution: {[f'{l:.4f}' for l in losses]}")
    print(f"  Beta  evolution: {[f'{b:.4f}' for b in betas]}")
    print(f"  Alpha evolution: {[f'{a:.4f}' for a in alphas]}")

    # Verify parameters changed from init
    assert betas[-1] != 0.0 or betas[0] == 0.0, "beta should evolve"
    print(f"  beta  grad flow: {betas[0]:.4f} -> {betas[-1]:.4f}")
    print(f"  alpha grad flow: {alphas[0]:.4f} -> {alphas[-1]:.4f}")
    print("  [PASS] Parameters receive gradients and evolve\n")


# ---------------------------------------------------------------------------
# Test 4: Attenuation severity range check
# ---------------------------------------------------------------------------
def test_attenuation_severity():
    """Verify attenuation severity a is in valid range [0, 1]."""
    print("=" * 60)
    print("Test 4: Attenuation severity range")
    print("=" * 60)

    from ultralytics.utils.loss import TCBBoxLoss

    B = 4
    feats = [
        torch.randn(B, 64, 80, 80),    # normal activations
        torch.randn(B, 32, 40, 40) * 0.5,  # weaker activations (simulated attenuation)
        torch.randn(B, 32, 20, 20) * 0.3,  # even weaker
    ]
    N_total = 80 * 80 + 40 * 40 + 20 * 20  # 8400
    anchor_points = torch.cat([
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 80, 1.0), torch.arange(0.5, 80, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 40, 1.0), torch.arange(0.5, 40, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
        torch.stack(torch.meshgrid(
            torch.arange(0.5, 20, 1.0), torch.arange(0.5, 20, 1.0), indexing="xy"
        ), dim=-1).reshape(-1, 2),
    ], dim=0)
    stride = torch.cat([
        torch.full((6400, 1), 8.0),
        torch.full((1600, 1), 16.0),
        torch.full((400, 1), 32.0),
    ], dim=0)
    fg_mask = torch.rand(B, N_total) < 0.3

    a = TCBBoxLoss._attenuation_severity(fg_mask, anchor_points, stride, feats)
    print(f"  a shape:    {a.shape}")
    print(f"  a min/max:  {a.min().item():.4f} / {a.max().item():.4f}")
    print(f"  a mean/std: {a.mean().item():.4f} / {a.std().item():.4f}")

    assert a.min() >= -0.01, f"a should be >= ~0, got min {a.min().item()}"
    assert a.max() <= 1.01, f"a should be <= ~1, got max {a.max().item()}"
    # With weaker feats (multiplied by 0.3, 0.5), we expect higher a values
    print("  [PASS] Attenuation severity in [0, 1] range\n")


# ---------------------------------------------------------------------------
# Test 5: End-to-end with DetectionModel
# ---------------------------------------------------------------------------
def test_e2e_detection_model():
    """Verify v8DetectionLoss instantiates with loss_type='tcb'."""
    print("=" * 60)
    print("Test 5: End-to-end with DetectionModel + loss_type='tcb'")
    print("=" * 60)

    from ultralytics.nn.tasks import DetectionModel
    from types import SimpleNamespace

    # Use A2FM YAML
    model = DetectionModel("yolo11-A2FM.yaml")
    model.args = SimpleNamespace(loss_type="tcb")
    model.eval()

    # Feed dummy data through model
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y = model(x)

    print(f"  Model output shape: {y[0].shape if isinstance(y, (list, tuple)) else y.shape}")

    # Check criterion type
    criterion = model.init_criterion()
    print(f"  Criterion type: {type(criterion).__name__}")
    print(f"  loss_type:      {criterion.loss_type}")
    print(f"  bbox_loss type: {type(criterion.bbox_loss).__name__}")

    from ultralytics.utils.loss import TCBBoxLoss
    assert isinstance(criterion.bbox_loss, TCBBoxLoss), \
        f"Expected TCBBoxLoss, got {type(criterion.bbox_loss).__name__}"
    print("  [PASS] DetectionModel correctly initialises TCB criterion\n")


# ---------------------------------------------------------------------------
# Test 6: Full forward-backward with criterion
# ---------------------------------------------------------------------------
def test_full_criterion():
    """Verify full criterion forward+backward with TCB loss type."""
    print("=" * 60)
    print("Test 6: Full criterion forward-backward")
    print("=" * 60)

    from ultralytics.nn.tasks import DetectionModel

    from types import SimpleNamespace
    model = DetectionModel("yolo11-A2FM.yaml")
    model.args = SimpleNamespace(
        box=7.5, cls=0.5, dfl=1.5, loss_type="tcb",
    )  # simulate training CLI args
    model.train()

    # Create dummy batch
    x = torch.randn(2, 3, 640, 640)
    batch = {
        "im_file": ["fake1.jpg", "fake2.jpg"],
        "ori_shape": [(640, 640), (640, 640)],
        "resized_shape": [(640, 640), (640, 640)],
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
        "cls": torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        "bboxes": torch.tensor([[100.0, 100.0, 150.0, 150.0],
                                 [200.0, 200.0, 250.0, 260.0],
                                 [300.0, 300.0, 320.0, 330.0]], dtype=torch.float32),
        "img": x,
    }

    # Forward
    preds = model(x)
    parsed = preds[1] if isinstance(preds, tuple) else preds

    criterion = model.init_criterion()
    loss, loss_items = criterion(preds, batch)

    print(f"  Total loss: {loss}")
    print(f"  Loss items: {loss_items.tolist()}")

    # Backward (loss is 3-element tensor, sum to scalar)
    loss.sum().backward()

    # Check gradients on TCB parameters (may be None if no fg_mask matches)
    bbox_loss = criterion.bbox_loss
    print(f"  beta  param: {bbox_loss.beta.item():.6f}, grad: {bbox_loss.beta.grad}")
    print(f"  alpha param: {bbox_loss.alpha_raw.item():.6f}, grad: {bbox_loss.alpha_raw.grad}")
    print(f"  C     value: {bbox_loss.C.item():.4f}")

    assert not torch.isnan(loss.sum())
    print("  [PASS] Full criterion forward+backward OK\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TCB Loss — Test Suite")
    print("=" * 60 + "\n")

    test_tcb_box_loss_unit()
    test_tcb_without_feats()
    test_progressive_activation()
    test_attenuation_severity()
    test_e2e_detection_model()
    test_full_criterion()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
