"""项目一：工业缺陷检测与分割系统 - 全局配置"""

import os
from pathlib import Path

# ========== 路径配置 ==========
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
COCO_DIR = ROOT / "steel.coco"           # 主数据集 (COCO格式)
COCO_TRAIN_DIR = COCO_DIR / "train"
COCO_VALID_DIR = COCO_DIR / "valid"

YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"
YOLO_IMAGES_TRAIN = YOLO_DATASET_DIR / "images" / "train"
YOLO_IMAGES_VAL = YOLO_DATASET_DIR / "images" / "val"
YOLO_LABELS_TRAIN = YOLO_DATASET_DIR / "labels" / "train"
YOLO_LABELS_VAL = YOLO_DATASET_DIR / "labels" / "val"

SCRIPTS_DIR = ROOT / "scripts"
OUTPUTS_DIR = ROOT / "outputs"
WEIGHTS_DIR = OUTPUTS_DIR / "weights"
RESULTS_DIR = OUTPUTS_DIR / "results"
MASKS_CACHE_DIR = OUTPUTS_DIR / "masks_cache"
DEMO_DIR = ROOT / "demo"

# 确保输出目录存在
for d in [WEIGHTS_DIR, RESULTS_DIR, MASKS_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ========== 数据集配置 ==========
# 6个缺陷类别 (YOLO内部ID 0-5)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# COCO JSON 中真正的缺陷类别 ID (id 0-5是Roboflow元数据垃圾)
COCO_VALID_CAT_IDS = {6, 7, 8, 9, 10, 11}
COCO_CAT_ID_TO_NAME = {
    6: "crazing",
    7: "inclusion",
    8: "patches",
    9: "pitted_surface",
    10: "rolled-in_scale",
    11: "scratches",
}

# 合并训练+验证后重新划分
TRAIN_RATIO = 0.8
# 背景图(无缺陷)保留比例, 设为None则全部丢弃
BACKGROUND_RATIO = 0.1  # 10%背景图用于减少误检
RANDOM_SEED = 42

# 训练/验证划分
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ========== 训练配置 ==========
TRAIN_CONFIG = {
    "model": "yolov8n-seg.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": 16,
    "device": 0,
    "workers": 4,
    "optimizer": "auto",  # AdamW
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "cos_lr": True,
    "close_mosaic": 10,  # 最后10个epoch关闭mosaic增强
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

# ========== MobileSAM 配置 ==========
MOBILE_SAM_MODEL = "mobile_sam.pt"  # ultralytics自动下载
SAM_CONFIDENCE_THRESHOLD = 0.5
USE_ELLIPSE_FALLBACK = True  # SAM下载失败时用椭圆mask兜底

# ========== 导出配置 ==========
ONNX_CONFIG = {
    "opset": 12,
    "simplify": True,
    "imgsz": 640,
    "half": False,  # ONNX用FP32，TRT用FP16
}

TENSORRT_CONFIG = {
    "workspace": 4,  # GB
    "half": True,    # FP16
    "imgsz": 640,
}

# ========== Benchmark 配置 ==========
BENCHMARK_NUM_IMAGES = 100
BENCHMARK_WARMUP = 10
BENCHMARK_REPEAT = 3

# ========== Gradio 配置 ==========
GRADIO_TITLE = "工业缺陷检测与分割系统"
GRADIO_DESCRIPTION = """
基于 YOLOv8n-seg 的钢铁表面缺陷检测与实例分割系统。
支持 PyTorch / ONNX / TensorRT 三种推理后端。
数据集: NEU-DET (6类缺陷, 1800张图像)
"""
GRADIO_SERVER_PORT = 7860
