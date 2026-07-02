"""
02_prepare_dataset.py
加载 mask 缓存 → 合并 train+valid → 重划分 80/20 → YOLO seg 格式 → dataset.yaml

输入: outputs/masks_cache/masks_cache.pkl
      steel.coco/train/*.jpg + steel.coco/valid/*.jpg
输出: data/yolo_dataset/ (images + labels + dataset.yaml)
"""

import sys, os, pickle, json, shutil, yaml
from pathlib import Path
import numpy as np
import cv2
import random
from tqdm import tqdm
from collections import defaultdict

# Windows GBK 编码修复
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def polygon_to_yolo_line(class_id, polygon):
    """多边形顶点 → YOLO seg 行: class_id x1 y1 x2 y2 ..."""
    coords = []
    for pt in polygon:
        coords.append(f"{pt[0]:.6f}")
        coords.append(f"{pt[1]:.6f}")
    return f"{class_id} " + " ".join(coords)


def convert_to_rgb(src_path, dst_path):
    """读取图片(可能灰度) → 3通道RGB → 写入目标 (兼容中文路径)"""
    # cv2.imread 中文路径用 np.fromfile 兜底
    img = cv2.imread(str(src_path))
    if img is None:
        img = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite 中文路径用 imencode 兜底
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ret:
        buf.tofile(str(dst_path))
        return True
    return False


def stratified_split(annotated_stems, masks_cache, bg_stems=None):
    """按类别分层划分 train/val, 并分配背景图"""
    # 每张图包含的类别
    stem_classes = defaultdict(set)
    for stem in annotated_stems:
        if stem in masks_cache:
            for obj in masks_cache[stem]:
                stem_classes[stem].add(obj["class_name"])

    # 按类别分组
    class_to_stems = defaultdict(list)
    for stem, classes in stem_classes.items():
        for cls in classes:
            class_to_stems[cls].append(stem)

    # 每类分别划分
    train_set = set()
    val_set = set()
    for cls_name, stems in class_to_stems.items():
        stems = list(set(stems))
        random.shuffle(stems)
        n_train = max(1, int(len(stems) * TRAIN_RATIO))
        for s in stems[:n_train]:
            train_set.add(s)
        for s in stems[n_train:]:
            val_set.add(s)

    # 分配背景图
    if bg_stems:
        random.shuffle(bg_stems)
        n_bg_train = int(len(bg_stems) * TRAIN_RATIO)
        for s in bg_stems[:n_bg_train]:
            train_set.add(s)
        for s in bg_stems[n_bg_train:]:
            val_set.add(s)

    return list(train_set), list(val_set)


def main():
    print("=" * 60)
    print("  02_prepare_dataset.py — 准备 YOLO 分割数据集")
    print("=" * 60)
    print()
    set_seed()

    # ---- 加载 mask 缓存 ----
    print("[1/5] 加载 mask 缓存...")
    cache_file = MASKS_CACHE_DIR / "masks_cache.pkl"
    if not cache_file.exists():
        print(f"✗ 未找到: {cache_file}")
        print(f"  请先运行 01_generate_masks.py")
        return

    with open(cache_file, "rb") as f:
        masks_cache = pickle.load(f)

    annotated_stems = list(masks_cache.keys())  # 文件名 (带 .jpg)
    print(f"  ✓ 有标注的图片: {len(annotated_stems)} 张")

    # ---- 构建图片路径映射 ----
    print("\n[2/5] 构建图片索引...")
    all_image_files = {}  # fname → full_path
    background_stems = []  # 无标注的图片

    for split_dir in [COCO_TRAIN_DIR, COCO_VALID_DIR]:
        for f in split_dir.glob("*.jpg"):
            if f.name not in all_image_files:
                all_image_files[f.name] = f

    # 找出背景图
    for fname in all_image_files:
        stem = fname  # 直接用文件名
        if stem not in masks_cache:
            background_stems.append(stem)

    print(f"  总图片: {len(all_image_files)} 张")
    print(f"  有标注: {len(annotated_stems)} 张")
    print(f"  背景(无缺陷): {len(background_stems)} 张")

    # 只保留部分背景图
    if BACKGROUND_RATIO and background_stems:
        n_bg = min(len(background_stems),
                   int(len(annotated_stems) * BACKGROUND_RATIO))
        background_stems = background_stems[:n_bg]
        print(f"  保留背景图: {n_bg} 张 ({BACKGROUND_RATIO*100:.0f}%)")

    # 移除无对应图片的标注
    valid_annotated = [s for s in annotated_stems if s in all_image_files]
    if len(valid_annotated) < len(annotated_stems):
        print(f"  ⚠ {len(annotated_stems) - len(valid_annotated)} 张标注对应的图片缺失, 已移除")

    # ---- 划分 ----
    print(f"\n[3/5] 80/20 分层划分...")
    train_stems, val_stems = stratified_split(valid_annotated, masks_cache, background_stems)
    print(f"  训练: {len(train_stems)} 张  |  验证: {len(val_stems)} 张")

    # 统计每类实例
    def count_instances(stems):
        counts = defaultdict(int)
        for stem in stems:
            if stem in masks_cache:
                for obj in masks_cache[stem]:
                    counts[obj["class_name"]] += 1
        return dict(counts)

    print(f"  训练实例: {count_instances(train_stems)}")
    print(f"  验证实例: {count_instances(val_stems)}")

    # ---- 清空输出目录 ----
    print(f"\n[4/5] 写入 YOLO seg 数据集...")
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    def process_split(stems, split_name, img_dst, lbl_dst):
        count_ok = 0
        for stem in tqdm(stems, desc=f"  {split_name}"):
            if stem not in all_image_files:
                continue

            src_img = all_image_files[stem]
            dst_img = img_dst / f"{stem}"
            if not convert_to_rgb(src_img, dst_img):
                continue

            # 标签文件: 文件名替换扩展名为 .txt
            lbl_name = Path(stem).stem + ".txt"
            lbl_path = lbl_dst / lbl_name

            if stem in masks_cache:
                with open(lbl_path, "w") as f:
                    for obj in masks_cache[stem]:
                        if obj["polygon"]:
                            line = polygon_to_yolo_line(obj["class_id"], obj["polygon"])
                            f.write(line + "\n")
            else:
                # 背景图: 空标签文件
                lbl_path.touch()

            count_ok += 1
        return count_ok

    n_train = process_split(train_stems, "训练", YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN)
    n_val = process_split(val_stems, "验证", YOLO_IMAGES_VAL, YOLO_LABELS_VAL)
    print(f"\n  训练集: {n_train} 张  |  验证集: {n_val} 张")

    # ---- 生成 dataset.yaml ----
    print(f"\n[5/5] 生成 dataset.yaml...")
    yaml_path = YOLO_DATASET_DIR / "dataset.yaml"
    dataset_config = {
        "path": str(YOLO_DATASET_DIR.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  ✓ {yaml_path}")

    print()
    print("=" * 60)
    print("  数据集准备完成！下一步: python scripts/03_train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
