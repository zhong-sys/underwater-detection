#!/usr/bin/env python3
"""
位置先验模块 · 阶段一训练（冻结 backbone + neck）
"""

import torch
from pathlib import Path
import sys
from multiprocessing import freeze_support

sys.path.insert(0, str(Path(__file__).resolve().parent / "ultralytics"))
from ultralytics import YOLO

# ======================== 配置 ========================
# 新创建的 YAML 路径
MODEL_YAML = "yolo11n-pp.yaml"   # 如果你用的是 EDFF 版本，改成对应 yaml 名字

# 基线权重路径（请确认实际路径）
BEST_PT = "D:/yolo-ce/runs/detect/runs/train/baseline_exp_25/weights/best.pt"
# 如果上面路径不对，试试 "runs/train/baseline_exp_25/weights/best.pt"

DATA = "D:/yolo-ce/ruod.yaml"

EPOCHS = 10
IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 8

LR0 = 0.001               # 低学习率，保护预训练权重
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

PROJECT = "runs/stage1_pp"
NAME = "exp1"
EXIST_OK = True


def main():
    # ======================== 1. 用新 YAML 构建模型 ========================
    model = YOLO(MODEL_YAML, task="detect")

    # ======================== 2. 加载基线权重，只迁移匹配层 ========================
    ckpt = torch.load(BEST_PT, map_location="cpu")
    pretrained_dict = ckpt["model"].float().state_dict()

    model_dict = model.model.state_dict()
    matched = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model.model.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)} / {len(model_dict)} layers from baseline.")

    # ======================== 3. 冻结 backbone + neck ========================
    # 打印层名称，帮你确定冻结范围
    print("======= 模型参数列表（前 30 层） =======")
    for i, (name, _) in enumerate(model.model.named_parameters()):
        if i >= 30:
            break
        print(f"{i:3d}  {name}")

    # 冻结前 20 层（你需要根据打印结果微调，确保 PositionPriorModule 和 Detect 不被冻结）
    FREEZE_LAYERS = 20
    for layer in list(model.model.children())[:FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False

    # 确认哪些层需要梯度
    for i, (name, p) in enumerate(model.model.named_parameters()):
        if p.requires_grad:
            print(f"TRAINABLE: {i} {name}")

    # ======================== 4. 训练 ========================
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="SGD",
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        amp=True,
        project=PROJECT,
        name=NAME,
        exist_ok=EXIST_OK,
        pretrained=False,          # 已手动加载权重
    )
    return results


if __name__ == "__main__":
    freeze_support()
    main()