#!/usr/bin/env python3
"""
RUOD small object analysis — motivation for CAFE & DRU paper.

Computes object size distribution and identifies how many objects fall below
the "small" threshold (area < 32×32 pixels) where FPN upsampling information
loss becomes critical.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# COCO-inspired size thresholds (pixels, at typical 640×640 input)
SMALL_THRESH = 32 * 32   # area < 1024 px²  → small
MEDIUM_THRESH = 96 * 96  # area < 9216 px² → medium, above → large

# RUOD class names
CLASS_NAMES = {
    0: "holothurian",
    1: "echinus",
    2: "scallop",
    3: "starfish",
    4: "fish",
    5: "corals",
    6: "diver",
    7: "cuttlefish",
    8: "turtle",
    9: "jellyfish",
}


def analyze_dataset(label_dir: str, image_dir: str) -> dict:
    """Analyse object size distribution in a dataset split."""
    stats = {
        "total_images": 0,
        "total_objects": 0,
        "small": 0,       # area < 1024
        "medium": 0,      # 1024 ≤ area < 9216
        "large": 0,       # area ≥ 9216
        "per_class": defaultdict(lambda: {"total": 0, "small": 0, "medium": 0, "large": 0}),
        "areas": [],
    }

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    stats["total_images"] = len(label_files)

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        if os.path.getsize(label_path) == 0:
            continue

        # Get image dimensions
        base_name = os.path.splitext(label_file)[0]
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            img_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                break

        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                # YOLO format: class cx cy w h (normalised)
                _, _, bw, bh = map(float, parts[1:5])
                obj_w, obj_h = bw * w, bh * h
                area = obj_w * obj_h

                stats["total_objects"] += 1
                stats["areas"].append(area)

                if area < SMALL_THRESH:
                    stats["small"] += 1
                    stats["per_class"][cls_id]["small"] += 1
                elif area < MEDIUM_THRESH:
                    stats["medium"] += 1
                    stats["per_class"][cls_id]["medium"] += 1
                else:
                    stats["large"] += 1
                    stats["per_class"][cls_id]["large"] += 1

                stats["per_class"][cls_id]["total"] += 1

    return stats


def print_report(name: str, stats: dict):
    """Print formatted analysis report."""
    total = max(stats["total_objects"], 1)
    areas = np.array(stats["areas"])

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Images:        {stats['total_images']}")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Area stats:    min={areas.min():.0f}  median={np.median(areas):.0f}  "
          f"mean={areas.mean():.0f}  max={areas.max():.0f}")
    print(f"  {'─'*50}")
    print(f"  Small  (< 32×32):  {stats['small']:5d}  ({100*stats['small']/total:5.1f}%)")
    print(f"  Medium (32-96²):   {stats['medium']:5d}  ({100*stats['medium']/total:5.1f}%)")
    print(f"  Large  (> 96×96):  {stats['large']:5d}  ({100*stats['large']/total:5.1f}%)")
    print(f"  {'─'*50}")
    print(f"  {'Class':<16} {'Total':>6} {'Small':>6} {'Small%':>7} {'Medium':>6} {'Large':>6}")
    for cls_id in sorted(stats["per_class"].keys()):
        c = stats["per_class"][cls_id]
        name = CLASS_NAMES.get(cls_id, f"cls_{cls_id}")
        ct = max(c["total"], 1)
        print(f"  {name:<16} {c['total']:>6} {c['small']:>6} {100*c['small']/ct:>6.1f}% "
              f"{c['medium']:>6} {c['large']:>6}")
    print()


def main():
    parser = argparse.ArgumentParser(description="RUOD small object analysis")
    parser.add_argument("--data_root", type=str, default="D:/ruod_split_811",
                        help="Root of the RUOD 8:1:1 split")
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        label_dir = os.path.join(args.data_root, "labels", split)
        image_dir = os.path.join(args.data_root, "images", split)
        if os.path.isdir(label_dir):
            stats = analyze_dataset(label_dir, image_dir)
            print_report(split.upper(), stats)
        else:
            print(f"\n[{split}] Directory not found: {label_dir}")


if __name__ == "__main__":
    main()
