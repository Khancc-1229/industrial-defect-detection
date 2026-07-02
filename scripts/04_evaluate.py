"""
04_evaluate.py
评估训练好的 YOLOv8n-seg 模型

生成:
  - mAP50, mAP50-95 指标
  - 每类 Precision / Recall / AP
  - 混淆矩阵
  - 验证集预测可视化（随机采样拼图）
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_per_class_metrics(metrics_dict, save_path):
    """每类 AP 柱状图"""
    classes = list(metrics_dict.keys())
    ap50 = [metrics_dict[c]["ap50"] for c in classes]
    ap50_95 = [metrics_dict[c]["ap50_95"] for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, ap50, width, label="mAP50", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + width/2, ap50_95, width, label="mAP50-95", color="#FF9800", edgecolor="white")

    # 数值标注
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel("类别")
    ax.set_ylabel("Average Precision")
    ax.set_title("Per-Class AP - YOLOv8n-seg on NEU-DET")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Per-class AP 柱状图: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    """混淆矩阵可视化"""
    # 归一化
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")

    # 标注数值
    thresh = cm_norm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 混淆矩阵: {save_path}")


def visualize_predictions(model, val_img_dir, num_samples=20, save_path=None):
    """随机采样验证集图片并可视化预测结果"""
    img_paths = sorted(list(val_img_dir.glob("*.jpg")))
    if len(img_paths) > num_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(img_paths), num_samples, replace=False)
        img_paths = [img_paths[i] for i in indices]

    cols = 5
    rows = (len(img_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes]

    for idx, img_path in enumerate(tqdm(img_paths, desc="  可视化预测")):
        ax = axes[idx]

        # 推理
        results = model(str(img_path), verbose=False)
        img_rgb = cv2.cvtColor(results[0].orig_img, cv2.COLOR_BGR2RGB)

        # 绘制分割 overlay
        if results[0].masks is not None:
            for mask_tensor, cls_id in zip(results[0].masks.data, results[0].boxes.cls):
                mask = mask_tensor.cpu().numpy()
                mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))
                # 彩色叠加
                color = plt.cm.tab10(int(cls_id) % 10)
                colored_mask = np.zeros_like(img_rgb, dtype=np.float32)
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c] * 0.5
                img_rgb = img_rgb.astype(np.float32)
                img_rgb = img_rgb * (1 - mask[:, :, None] * 0.5) + colored_mask * 255
                img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

        ax.imshow(img_rgb)
        ax.set_title(img_path.stem[:20], fontsize=7)
        ax.axis("off")

    # 隐藏多余的 subplot
    for idx in range(len(img_paths), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 预测可视化: {save_path}")


def main():
    print("=" * 60)
    print("  04_evaluate.py - 模型评估")
    print("=" * 60)
    print()

    # ---- 加载模型 ----
    best_pt = WEIGHTS_DIR / "best.pt"
    if not best_pt.exists():
        best_pt = RESULTS_DIR / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"✗ 未找到模型权重: {best_pt}")
        print(f"  请先运行 03_train.py")
        return

    from ultralytics import YOLO
    model = YOLO(str(best_pt))

    dataset_yaml = YOLO_DATASET_DIR / "dataset.yaml"

    # ---- 验证集评估 ----
    print(f"[1/4] 验证集评估...")
    metrics = model.val(data=str(dataset_yaml), split="val", verbose=True, plots=True)

    # 提取指标
    results_summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": {}
    }

    # 每类指标
    if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap is not None:
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = CLASS_NAMES[int(cls_idx)]
            results_summary["per_class"][cls_name] = {
                "ap50": float(metrics.box.ap50[i]) if metrics.box.ap50 is not None else 0.0,
                "ap50_95": float(metrics.box.ap[i]) if metrics.box.ap is not None else 0.0,
            }

    print(f"  mAP50: {results_summary['mAP50']:.4f}")
    print(f"  mAP50-95: {results_summary['mAP50_95']:.4f}")
    for cls_name, ap in results_summary["per_class"].items():
        print(f"    {cls_name}: AP50={ap['ap50']:.4f}, AP50-95={ap['ap50_95']:.4f}")

    # 保存JSON
    json_path = RESULTS_DIR / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 指标JSON: {json_path}")

    # ---- Per-class AP 图 ----
    print(f"\n[2/4] 绘制 Per-class AP 图...")
    if results_summary["per_class"]:
        plot_per_class_metrics(results_summary["per_class"],
                               RESULTS_DIR / "per_class_ap.png")

    # ---- 混淆矩阵 ----
    print(f"\n[3/4] 混淆矩阵...")
    # Ultralytics 自动生成了混淆矩阵，我们复制过来
    cm_src = RESULTS_DIR / "train" / "confusion_matrix.png"
    if cm_src.exists():
        import shutil
        shutil.copy2(cm_src, RESULTS_DIR / "confusion_matrix.png")
        print(f"  ✓ 混淆矩阵 (来自训练): {RESULTS_DIR / 'confusion_matrix.png'}")
    else:
        print(f"  ⚠ 未找到自动生成的混淆矩阵")

    # ---- 预测可视化 ----
    print(f"\n[4/4] 验证集预测可视化...")
    visualize_predictions(
        model,
        YOLO_IMAGES_VAL,
        num_samples=20,
        save_path=RESULTS_DIR / "prediction_samples.png"
    )

    # ---- 打印完整汇总表格 ----
    print(f"\n{'='*60}")
    print(f"  评估汇总")
    print(f"{'='*60}")
    print(f"  {'类别':<20s} {'AP50':>8s} {'AP50-95':>8s}")
    print(f"  {'-'*36}")
    for cls_name in CLASS_NAMES:
        ap = results_summary["per_class"].get(cls_name, {"ap50": 0, "ap50_95": 0})
        print(f"  {cls_name:<20s} {ap['ap50']:>8.4f} {ap['ap50_95']:>8.4f}")
    print(f"  {'-'*36}")
    print(f"  {'Overall':<20s} {results_summary['mAP50']:>8.4f} {results_summary['mAP50_95']:>8.4f}")

    print()
    print("=" * 60)
    print("  评估完成！下一步: 运行 05_export_onnx.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
