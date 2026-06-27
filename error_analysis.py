#!/usr/bin/env python3
"""
RUOD 错误分析 — 诊断 yolo11n 到底弱在哪。

输出:
    1. 混淆矩阵（类间互错）
    2. 按尺寸分层的 Recall（小/中/大目标各自漏检率）
    3. FP 分布（误检集中在哪些类）
    4. FN 分布（漏检集中在哪些类 + 什么尺寸）
    5. 定位质量分析（匹配到的框 IoU 分布）

Usage (服务器):
    python error_analysis.py --data D:/ruod_split_811 --model baseline.pt --device 0
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ── COCO 尺寸阈值 ──────────────────────────────────────────
SMALL_THRESH = 32 * 32       # area < 1024 px²
MEDIUM_THRESH = 96 * 96      # area < 9216 px²
IOU_THRESH = 0.5

# RUOD 类别
CLASS_NAMES = {
    0: "holothurian", 1: "echinus", 2: "scallop",
    3: "starfish", 4: "fish", 5: "corals",
    6: "diver", 7: "cuttlefish", 8: "turtle", 9: "jellyfish",
}


def compute_iou(box1, box2):
    """box: (x1, y1, x2, y2) in pixels."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def get_size_label(area):
    if area < SMALL_THRESH:
        return "small"
    elif area < MEDIUM_THRESH:
        return "medium"
    return "large"


def load_ground_truth(label_dir: str, image_dir: str, img_size: int = 640):
    """加载所有测试集 ground truth。

    Returns:
        gt: dict[image_filename] = list of (cls, x1, y1, x2, y2, area)
    """
    gt = {}
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    for lf in tqdm(label_files, desc="Loading GT"):
        base = os.path.splitext(lf)[0]
        # 找对应图片
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            ip = os.path.join(image_dir, base + ext)
            if os.path.exists(ip):
                img = cv2.imread(ip)
                break
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes = []
        lp = os.path.join(label_dir, lf)
        if os.path.getsize(lp) == 0:
            gt[base] = boxes
            continue

        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                # YOLO norm → pixel
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                area = (x2 - x1) * (y2 - y1)
                boxes.append((cls_id, x1, y1, x2, y2, area))
        gt[base] = boxes

    return gt


def run_inference(model_path: str, image_dir: str, device: str, conf: float = 0.001):
    """用 ultralytics 跑推理，返回所有预测。

    Returns:
        preds: dict[image_filename] = list of (cls, x1, y1, x2, y2, score)
    """
    from ultralytics import YOLO
    model = YOLO(model_path)

    preds = {}
    result = model.predict(
        source=image_dir,
        device=device,
        conf=conf,
        iou=0.65,
        verbose=False,
        stream=True,
    )
    for r in tqdm(result, desc="Inference"):
        base = Path(r.path).stem
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                boxes.append((
                    cls_ids[i],
                    float(xyxy[i][0]), float(xyxy[i][1]),
                    float(xyxy[i][2]), float(xyxy[i][3]),
                    float(confs[i]),
                ))
        preds[base] = boxes

    return preds


def match_predictions(gt: dict, preds: dict, iou_thresh: float = IOU_THRESH):
    """匹配预测框和 GT 框。

    Returns:
        matches: list of (image_name, gt_cls, pred_cls, gt_size, iou, pred_score)
        false_positives: list of (image_name, pred_cls, pred_score, pred_size)
        false_negatives: list of (image_name, gt_cls, gt_size)
    """
    matches = []
    false_positives = []
    false_negatives = []

    for img_name, gt_boxes in tqdm(gt.items(), desc="Matching"):
        if img_name not in preds:
            continue
        pred_boxes = preds[img_name]

        matched_gt = set()
        matched_pred = set()

        # 贪心匹配：按 score 降序，每个 pred 找最佳 GT
        sorted_preds = sorted(
            enumerate(pred_boxes),
            key=lambda x: x[1][5],  # score
            reverse=True,
        )

        for pi, pbox in sorted_preds:
            p_cls, px1, py1, px2, py2, p_score = pbox
            best_iou = 0.0
            best_gi = -1

            for gi, gbox in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                g_cls, gx1, gy1, gx2, gy2, g_area = gbox
                iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
                    best_g_cls = g_cls
                    best_g_size = get_size_label(g_area)

            if best_iou >= iou_thresh and best_gi >= 0:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                matches.append((
                    img_name, best_g_cls, p_cls, best_g_size, best_iou, p_score
                ))
            else:
                p_area = (px2 - px1) * (py2 - py1)
                false_positives.append((
                    img_name, p_cls, p_score, get_size_label(max(p_area, 1))
                ))

        # 未匹配的 GT → FN
        for gi, gbox in enumerate(gt_boxes):
            if gi not in matched_gt:
                g_cls = gbox[0]
                g_area = gbox[5]
                false_negatives.append((
                    img_name, g_cls, get_size_label(g_area)
                ))

    return matches, false_positives, false_negatives


def analyze(matches, false_positives, false_negatives):
    """汇总分析并打印报告。"""
    n_cls = 10

    # ── 1. 混淆矩阵 ──
    conf_matrix = np.zeros((n_cls, n_cls), dtype=int)  # [gt_cls, pred_cls]
    for _, gt_cls, pred_cls, _, _, _ in matches:
        conf_matrix[gt_cls, pred_cls] += 1

    # ── 2. 每类 TP / FP / FN ──
    tp_per_cls = np.zeros(n_cls, dtype=int)
    fp_per_cls = np.zeros(n_cls, dtype=int)
    fn_per_cls = np.zeros(n_cls, dtype=int)
    for _, gt_cls, pred_cls, _, _, _ in matches:
        tp_per_cls[gt_cls] += 1
    for _, pred_cls, _, _ in false_positives:
        fp_per_cls[pred_cls] += 1
    for _, gt_cls, _ in false_negatives:
        fn_per_cls[gt_cls] += 1

    # ── 3. 尺寸分层 Recall ──
    tp_size = {"small": np.zeros(n_cls, int), "medium": np.zeros(n_cls, int), "large": np.zeros(n_cls, int)}
    fn_size = {"small": np.zeros(n_cls, int), "medium": np.zeros(n_cls, int), "large": np.zeros(n_cls, int)}
    for _, gt_cls, _, gt_size, _, _ in matches:
        tp_size[gt_size][gt_cls] += 1
    for _, gt_cls, gt_size in false_negatives:
        fn_size[gt_size][gt_cls] += 1

    # ── 4. FP 的尺寸分布 ──
    fp_size = {"small": np.zeros(n_cls, int), "medium": np.zeros(n_cls, int), "large": np.zeros(n_cls, int)}
    for _, pred_cls, _, pred_size in false_positives:
        fp_size[pred_size][pred_cls] += 1

    # ── 5. 定位 IoU 分布 ──
    iou_per_cls = defaultdict(list)
    for _, gt_cls, _, _, iou, _ in matches:
        iou_per_cls[gt_cls].append(iou)

    # ── 6. FP 的 score 分布 ──
    fp_score_per_cls = defaultdict(list)
    for _, pred_cls, score, _ in false_positives:
        fp_score_per_cls[pred_cls].append(score)

    # ════════════════════════════════════════════════════════════
    # 打印报告
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 85)
    print("  RUOD ERROR ANALYSIS — yolo11n Baseline")
    print("=" * 85)

    # ── 统计总量 ──
    total_gt = sum(tp_per_cls) + sum(fn_per_cls)
    total_pred = sum(tp_per_cls) + sum(fp_per_cls)
    print(f"\n  GT 目标数: {total_gt}  |  预测数: {total_pred}")
    print(f"  TP: {sum(tp_per_cls)}  |  FP: {sum(fp_per_cls)}  |  FN: {sum(fn_per_cls)}\n")

    # ── 逐类详细表 ──
    header = f"  {'Class':<16} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'小R':>7} {'中R':>7} {'大R':>7} {'IoU_avg':>7}"
    print(header)
    print("  " + "-" * len(header))
    for cls_id in range(n_cls):
        name = CLASS_NAMES[cls_id]
        gt_n = tp_per_cls[cls_id] + fn_per_cls[cls_id]
        tp = tp_per_cls[cls_id]
        fp = fp_per_cls[cls_id]
        fn = fn_per_cls[cls_id]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(gt_n, 1)

        # 尺寸 Recall
        r_small = tp_size["small"][cls_id] / max(tp_size["small"][cls_id] + fn_size["small"][cls_id], 1)
        r_med = tp_size["medium"][cls_id] / max(tp_size["medium"][cls_id] + fn_size["medium"][cls_id], 1)
        r_large = tp_size["large"][cls_id] / max(tp_size["large"][cls_id] + fn_size["large"][cls_id], 1)

        iou_avg = np.mean(iou_per_cls[cls_id]) if iou_per_cls[cls_id] else 0

        print(f"  {name:<16} {gt_n:>5} {tp:>5} {fp:>5} {fn:>5} "
              f"{prec:>6.3f} {rec:>6.3f} "
              f"{r_small:>6.3f} {r_med:>6.3f} {r_large:>6.3f} "
              f"{iou_avg:>6.3f}")

    # ── 尺寸汇总 ──
    print(f"\n  {'─'*60}")
    print(f"  {'Size':<12} {'TP':>6} {'FN':>6} {'Recall':>8}")
    for sz in ["small", "medium", "large"]:
        tps = sum(tp_size[sz])
        fns = sum(fn_size[sz])
        r = tps / max(tps + fns, 1)
        print(f"  {sz:<12} {tps:>6} {fns:>6} {r:>8.3f}")

    # ── 混淆矩阵（上三角 = 类别混淆） ──
    print(f"\n  {'─'*60}")
    print("  混淆矩阵 (行=GT, 列=Pred)")
    print(f"  {'':>16}", end="")
    for i in range(n_cls):
        print(f"  {CLASS_NAMES[i][:4]:>4}", end="")
    print()
    for i in range(n_cls):
        print(f"  {CLASS_NAMES[i]:>16}", end="")
        for j in range(n_cls):
            n = conf_matrix[i, j]
            if n > 0:
                print(f"  {n:>4}", end="")
            else:
                print(f"  {'·':>4}", end="")
        print()

    # ── 关键发现 ──
    print(f"\n  {'='*85}")
    print("  关键发现")
    print(f"  {'='*85}")

    # 找出 Recall 最差的类
    worst_rec = sorted(
        [(fn_per_cls[i] / max(tp_per_cls[i] + fn_per_cls[i], 1), CLASS_NAMES[i])
         for i in range(n_cls)
    ], reverse=True)[:3]  # worst = highest miss rate

    # 找出 FP 最多的类
    most_fp = sorted(
        [(fp_per_cls[i], CLASS_NAMES[i]) for i in range(n_cls)],
        reverse=True
    )[:3]

    # 找出最常见混淆对
    confusion_pairs = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i != j and conf_matrix[i, j] > 0:
                confusion_pairs.append((conf_matrix[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
    confusion_pairs.sort(reverse=True)
    top_confusions = confusion_pairs[:5]

    print(f"\n  🔴 漏检率最高 (FN 多):")
    for rate, name in worst_rec:
        print(f"      {name}: 漏检率 {rate:.1%} ({fn_per_cls[[k for k,v in CLASS_NAMES.items() if v==name][0]]}/{tp_per_cls[[k for k,v in CLASS_NAMES.items() if v==name][0]]+fn_per_cls[[k for k,v in CLASS_NAMES.items() if v==name][0]]})")

    print(f"\n  ⚪ 误检最多 (FP 多):")
    for n, name in most_fp:
        print(f"      {name}: FP={n} ({n/sum(fp_per_cls)*100:.0f}% of all FP)")

    print(f"\n  🔀 最常见类别混淆:")
    for n, gt, pred in top_confusions:
        print(f"      {gt} → {pred}: {n} 次")

    # 小目标集中在大类？
    small_fn_dist = sorted(
        [(fn_size["small"][i], CLASS_NAMES[i]) for i in range(n_cls)],
        reverse=True
    )
    print(f"\n  📐 小目标漏检分布:")
    for n, name in small_fn_dist[:5]:
        total_fn_cls = fn_per_cls[[k for k,v in CLASS_NAMES.items() if v==name][0]]
        pct = n / max(total_fn_cls, 1) * 100
        print(f"      {name}: {n} 个小目标漏检 ({pct:.0f}% of total FN for this class)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=r"D:/ruod_split_811")
    parser.add_argument("--model", default=r"runs/detect/train/abl_yolo11/weights/best.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    args = parser.parse_args()

    data_root = Path(args.data)
    img_dir = str(data_root / "images" / "test")
    label_dir = str(data_root / "labels" / "test")

    print(f"Data:   {args.data}")
    print(f"Model:  {args.model}")
    print(f"Device: {args.device}\n")

    # 1. 加载 GT
    gt = load_ground_truth(label_dir, img_dir)

    # 2. 推理
    preds = run_inference(args.model, img_dir, args.device, args.conf)

    # 3. 匹配
    matches, fps, fns = match_predictions(gt, preds)

    # 4. 分析
    analyze(matches, fps, fns)


if __name__ == "__main__":
    main()
