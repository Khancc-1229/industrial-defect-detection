"""
03_train.py
训练 YOLOv8n-seg 分割模型

使用 Ultralytics 框架，配置来自 config.py
训练期间 GPU 全速运行，预计 1.5-2 小时（50 epochs on RTX 4060）
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# 禁用 W&B 等第三方日志
os.environ["WANDB_MODE"] = "disabled"
os.environ["COMET_MODE"] = "disabled"


def main():
    print("=" * 60)
    print("  03_train.py - 训练 YOLOv8n-seg")
    print("=" * 60)
    print()

    # ---- 检查数据 ----
    dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"✗ 未找到 dataset.yaml: {dataset_yaml}")
        print(f"  请先运行 02_prepare_dataset.py")
        return

    n_train = len(list(YOLO_IMAGES_TRAIN.glob("*.jpg")))
    n_val = len(list(YOLO_IMAGES_VAL.glob("*.jpg")))
    print(f"训练集: {n_train} 张, 验证集: {n_val} 张")
    print(f"类别: {CLASS_NAMES}")
    print(f"模型: {TRAIN_CONFIG['model']}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}, Batch: {TRAIN_CONFIG['batch']}, ImgSz: {TRAIN_CONFIG['imgsz']}")
    print()

    # ---- 开始训练 ----
    from ultralytics import YOLO

    print("加载预训练权重...")
    model = YOLO(TRAIN_CONFIG["model"])

    print(f"\n开始训练 ({datetime.now().strftime('%H:%M:%S')})...")
    print("-" * 60)

    # 训练配置
    train_args = {
        "data": str(dataset_yaml),
        "epochs": TRAIN_CONFIG["epochs"],
        "imgsz": TRAIN_CONFIG["imgsz"],
        "batch": TRAIN_CONFIG["batch"],
        "device": TRAIN_CONFIG["device"],
        "workers": TRAIN_CONFIG["workers"],
        "optimizer": TRAIN_CONFIG["optimizer"],
        "lr0": TRAIN_CONFIG["lr0"],
        "lrf": TRAIN_CONFIG["lrf"],
        "momentum": TRAIN_CONFIG["momentum"],
        "weight_decay": TRAIN_CONFIG["weight_decay"],
        "warmup_epochs": TRAIN_CONFIG["warmup_epochs"],
        "warmup_momentum": TRAIN_CONFIG["warmup_momentum"],
        "cos_lr": TRAIN_CONFIG["cos_lr"],
        "close_mosaic": TRAIN_CONFIG["close_mosaic"],
        # 数据增强
        "hsv_h": TRAIN_CONFIG["hsv_h"],
        "hsv_s": TRAIN_CONFIG["hsv_s"],
        "hsv_v": TRAIN_CONFIG["hsv_v"],
        "degrees": TRAIN_CONFIG["degrees"],
        "translate": TRAIN_CONFIG["translate"],
        "scale": TRAIN_CONFIG["scale"],
        "fliplr": TRAIN_CONFIG["fliplr"],
        "mosaic": TRAIN_CONFIG["mosaic"],
        # 保存设置
        "project": str(RESULTS_DIR),
        "name": "train",
        "exist_ok": True,
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
    }

    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n✗ 训练出错: {e}")
        print("  尝试减小 batch size 到 8...")
        train_args["batch"] = 8
        try:
            results = model.train(**train_args)
        except Exception as e2:
            print(f"\n✗ 再次失败: {e2}")
            return

    print("-" * 60)
    print(f"\n训练完成! ({datetime.now().strftime('%H:%M:%S')})")

    # ---- 保存最佳权重到 outputs/weights/ ----
    best_pt = RESULTS_DIR / "train" / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dst = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_pt, dst)
        print(f"✓ 最佳权重: {dst}")

    # 训练结果摘要
    results_csv = RESULTS_DIR / "train" / "results.csv"
    if results_csv.exists():
        print(f"✓ 训练曲线: {results_csv}")

    print()
    print("=" * 60)
    print("  训练完成！下一步: 运行 04_evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
