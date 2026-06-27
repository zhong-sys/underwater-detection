#!/usr/bin/env python3
"""
使用训练好的模型评估测试集，并打印总体和每类指标。

示例：
    python test.py --model D:/yolo-ce/runs/detect/runs/train/baseline_exp-14/weights/best.pt --data D:/yolo-ce/ruod.yaml --split test
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
if str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO

# ------------------ Configuration (edit paths here) ------------------
# Edit these variables directly in the script instead of using CLI args
MODEL_PATH = "D:/yolo-ce/runs/detect/runs/train/baseline_exp_25-7/weights/best.pt"
DATA_PATH = "D:/yolo-ce/ruod.yaml"
SPLIT = "test"  # one of 'train','val','test'
IMG_SIZE = 640
BATCH = 16
DEVICE = 0
WORKERS = 8  # 0 is safest on Windows
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO model on a dataset split.")
    parser.add_argument("--model", default=MODEL_PATH, help="trained weight file")
    parser.add_argument("--data", default=DATA_PATH, help="dataset YAML file")
    parser.add_argument("--split", default=SPLIT, choices=["train", "val", "test"], help="dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="inference image size")
    parser.add_argument("--batch", type=int, default=BATCH, help="evaluation batch size")
    parser.add_argument("--device", default=DEVICE, help="device id, e.g. 0, 0,1, or cpu")
    parser.add_argument("--workers", type=int, default=WORKERS, help="dataloader workers; 0 is safest on Windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
    )

    overall = metrics.results_dict

    print("\n" + "=" * 72)
    print(f"整体指标（{args.split} 集）")
    print("=" * 72)
    print(f"Precision:  {overall['metrics/precision(B)']:.4f}")
    print(f"Recall:     {overall['metrics/recall(B)']:.4f}")
    print(f"mAP50:      {overall['metrics/mAP50(B)']:.4f}")
    print(f"mAP50-95:   {overall['metrics/mAP50-95(B)']:.4f}")

    print("\n" + "=" * 72)
    print("各类别指标（按 AP50-95 从高到低）")
    print("=" * 72)
    print(f"{'Class':>15s}  {'P':>8s}  {'R':>8s}  {'AP50':>8s}  {'AP50-95':>8s}")
    print("-" * 72)

    per_class = []
    for cls_id, name in model.names.items():
        p, r, ap50, ap50_95 = metrics.box.class_result(cls_id)
        per_class.append(
            {
                "class": name,
                "P": float(p),
                "R": float(r),
                "AP50": float(ap50),
                "AP50-95": float(ap50_95),
            }
        )

    per_class.sort(key=lambda item: item["AP50-95"], reverse=True)
    for item in per_class:
        print(f"{item['class']:>15s}  {item['P']:8.4f}  {item['R']:8.4f}  {item['AP50']:8.4f}  {item['AP50-95']:8.4f}")

    print("\n✅ 评估完成。")


if __name__ == "__main__":
    main()