#!/usr/bin/env python3
"""
YOLO11n + EPP 两阶段训练脚本

目标：
1. 保持与 train.py 相同的训练风格和参数组织方式
2. 使用 yolo11n-EPP.yaml 构建新模型
3. 从基线 checkpoint 中仅迁移形状匹配的参数，保证公平初始化
4. 阶段一冻结原始 backbone + neck，只微调 EPP 和 Detect
5. 阶段二解冻全模型，在阶段一结果上进一步微调
"""

import os
from pathlib import Path
import sys
from multiprocessing import freeze_support

import torch

# Windows/Conda 下常见的 OpenMP 运行时冲突会直接阻止程序启动。
# 这里先放宽限制，保证训练脚本能继续跑起来；如需更干净的环境，可后续再统一依赖版本。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent
LOCAL_ULTRALYTICS = ROOT / "ultralytics"
if str(LOCAL_ULTRALYTICS) not in sys.path:
    sys.path.insert(0, str(LOCAL_ULTRALYTICS))

from ultralytics import YOLO

# ======================== 基础配置 ========================
# 数据集配置文件 (YAML 格式，需要包含 train/val 路径和类别)
DATA = "D:/yolo-ce/ruod.yaml"

# 模型配置
MODEL_YAML = "yolo11n-EPP.yaml"

# 基线权重路径：建议使用同一套 baseline 训练出来的 best.pt
# 也可以先用官方 yolov11n.pt 作为初始化起点
BASELINE_CKPT = "D:/yolo-ce/runs/detect/runs/train/baseline_exp_25-7/weights/best.pt"
# BASELINE_CKPT = "D:/yolo-ce/yolo11n.pt"

# 训练超参数
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 280
IMGSZ = 640
BATCH = 16                    # 根据显存调整，如 32/16
DEVICE = 0                    # GPU 编号，"0,1" 多卡，"cpu"
WORKERS = 8                   # 数据加载线程数

# 优化器与学习率
OPTIMIZER = "SGD"
STAGE1_LR0 = 0.001
STAGE2_LR0 = 0.0005
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# 输出与项目设置
PROJECT = "runs/train"
NAME = "epp_two_stage_exp"
EXIST_OK = False

# 诊断阶段：先用 freeze=0 快速测试，看是否是冻结策略的问题
# 如果 freeze=0 效果还差，说明是权重/架构问题
# 如果 freeze=0 效果好，再改回 freeze=11 只冻 backbone
FREEZE = 23

STAGE1_PROJECT = PROJECT
STAGE1_NAME = f"{NAME}_stage1"
STAGE2_PROJECT = PROJECT
STAGE2_NAME = f"{NAME}_stage2"


def _extract_state_dict(checkpoint):
    """从 Ultralytics / PyTorch checkpoint 中尽量稳健地取出 state_dict。"""
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and checkpoint["ema"] is not None:
            return checkpoint["ema"].float().state_dict()
        if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
            return checkpoint["model"].float().state_dict()
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def load_partial_weights(model, ckpt_path):
    """仅加载与当前模型 key + shape 完全匹配的权重。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_state = _extract_state_dict(ckpt)
    model_state = model.model.state_dict()

    matched = {
        key: value
        for key, value in pretrained_state.items()
        if key in model_state and value.shape == model_state[key].shape
    }

    model.model.load_state_dict(matched, strict=False)
    print(f"\n===== Weight Loading Info =====")
    print(f"Loaded {len(matched)} / {len(model_state)} tensors from {ckpt_path}")
    print(f"Match ratio: {100 * len(matched) / len(model_state):.1f}%")
    if len(matched) < len(model_state) * 0.8:
        print("WARNING: Less than 80% weights matched! Check architecture compatibility.")
    print(f"==============================\n")


# ======================== 训练入口 ========================
if __name__ == "__main__":
    freeze_support()

    # ======================== 阶段一：冻结 backbone + neck ========================
    model = YOLO(MODEL_YAML, task="detect")

    # 从基线 checkpoint 中做公平初始化：只迁移匹配的层
    load_partial_weights(model, BASELINE_CKPT)

    print("\n===== Stage 1: freeze=0 diagnostic test =====")
    results_stage1 = model.train(
        data=DATA,
        epochs=STAGE1_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=STAGE1_LR0,
        momentum=MOMENTUM,
        amp=True,
        weight_decay=WEIGHT_DECAY,
        freeze=FREEZE,
        project=STAGE1_PROJECT,
        patience=30,
        name=STAGE1_NAME,
        exist_ok=EXIST_OK,
        pretrained=False,       # 已手动加载权重
        cache=False,
    )

    stage1_best = Path(model.trainer.best)
    print(f"Stage 1 done. save_dir: {model.trainer.save_dir}")
    print(f"Stage 1 done. best.pt: {stage1_best}")

    if not stage1_best.exists():
        raise FileNotFoundError(f"Stage 1 best checkpoint not found: {stage1_best}")

    # ======================== 阶段二：解冻全模型微调 ========================
    print("\n===== Stage 2: unfreeze all and fine-tune =====")
    model_stage2 = YOLO(str(stage1_best), task="detect")
    results_stage2 = model_stage2.train(
        data=DATA,
        epochs=STAGE2_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=STAGE2_LR0,
        momentum=MOMENTUM,
        amp=True,
        weight_decay=WEIGHT_DECAY,
        freeze=0,
        project=STAGE2_PROJECT,
        patience=30,
        name=STAGE2_NAME,
        exist_ok=EXIST_OK,
        pretrained=False,
        cache=False,
    )

    print("Training finished.")
    print(f"Stage 1 best: {stage1_best}")
