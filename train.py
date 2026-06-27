#!/usr/bin/env python3
"""
常规 YOLO 训练脚本 (基于 Ultralytics)
Usage:
    python train_baseline.py
"""

import os
from pathlib import Path
import sys

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
DATA = "D:/yolo-ce/ruod.yaml"   # 请替换为你的数据集路径

# 训练超参数
EPOCHS = 300
IMGSZ = 640
BATCH = 16                    # 根据显存调整，如 32/16
DEVICE = 0                   # GPU 编号，"0,1" 多卡，"cpu"
WORKERS = 8                   # 数据加载线程数

# 优化器与学习率
OPTIMIZER = "SGD"
LR0 = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005

# 输出与项目设置
PROJECT = "runs/train"        # 保存根目录
NAME = "baseline_exp_25"  # 实验名称 (会自动创建子文件夹)
EXIST_OK = False               # 若存在同名目录，是否覆盖

# ======================== 训练入口 ========================
if __name__ == "__main__":
    # 加载模型
    model = YOLO("yolo11n-DCA-CGA.yaml")   # 先根据 yaml 构建网络
    # model = YOLO("yolo11n.pt")       # 直接加载预训练权重（会自动构建网络）
    # model.load("yolo11n.pt")            # 手动加载指定权重
    # 开始训练
    results = model.train(
        data=DATA,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        loss_type="tcb",          # 启用 TCB 损失函数 (Beer-Lambert 传输校准边界损失)
        momentum=MOMENTUM,
        amp=True,                # 是否使用混合精度训练 (自动根据硬件支持选择)
        weight_decay=WEIGHT_DECAY,
        project=PROJECT,
        patience=30,           # 早停轮数，若 val mAP 不提升则提前结束训练
        name=NAME,
        exist_ok=EXIST_OK,
        pretrained=False,       # 若为 .yaml 且设为 True，会自动下载官方预训练权重并迁移
        cache=False,              # 是否缓存数据集到内存，提升小数据集训练速度
    )