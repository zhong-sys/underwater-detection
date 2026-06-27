#!/usr/bin/env python3
"""
消融实验训练脚本 — CAFE & DRU for Underwater Small Object Detection

实验矩阵:
    python train_ablation.py --model yolo11.yaml              # E1: baseline
    python train_ablation.py --model yolo11n-CAFE.yaml        # E2: +CAFE
    python train_ablation.py --model yolo11n-DRU.yaml         # E3: +DRU
    python train_ablation.py --model yolo11n-CAFE-DRU.yaml    # E4: +CAFE+DRU

Usage:
    python train_ablation.py --model yolo11n-CAFE-DRU.yaml --epochs 300 --name exp_cafe_dru
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="CAFE & DRU Ablation Training")
    parser.add_argument("--model", type=str, default="yolo11n-CAFE-DRU.yaml",
                        help="Model YAML config")
    parser.add_argument("--data", type=str, default="D:/yolo-ce/ruod.yaml",
                        help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device")
    parser.add_argument("--workers", type=int, default=8,
                        help="Data loader workers")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--name", type=str, default=None,
                        help="Experiment name (auto-generated if not set)")
    parser.add_argument("--project", type=str, default="runs/train",
                        help="Output project directory")
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help="Optimizer: SGD, Adam, AdamW")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable AMP")
    return parser.parse_args()


def main():
    args = parse_args()

    # Auto-generate experiment name from model config
    if args.name is None:
        model_stem = Path(args.model).stem  # e.g. "yolo11n-CAFE-DRU"
        args.name = f"abl_{model_stem}"

    print(f"{'='*60}")
    print(f"Model:   {args.model}")
    print(f"Exp:     {args.name}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"{'='*60}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=0.937,
        weight_decay=0.0005,
        amp=args.amp,
        project=args.project,
        patience=args.patience,
        name=args.name,
        exist_ok=False,
        pretrained=False,
        cache=False,
    )
    return results


if __name__ == "__main__":
    main()
