"""
01_generate_masks.py
从 steel.coco (COCO格式) 读取 bbox 标注 → MobileSAM 生成分割 mask → 缓存

数据源: steel.coco/train + steel.coco/valid (COCO JSON)
输出: outputs/masks_cache/masks_cache.pkl

关键: COCO JSON 中 category_id 6-11 才是真正的6类缺陷,
      id 0-5 和 12 是 Roboflow 元数据垃圾, 需要过滤掉
"""

import sys, os, pickle, json
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def load_coco_annotations(json_path, img_dir):
    """读取COCO JSON, 返回 {file_name: [{bbox_xyxy, category_id, class_name, yolo_class_id}]}"""
    with open(json_path) as f:
        data = json.load(f)

    id_to_img = {img["id"]: img["file_name"] for img in data["images"]}

    img_anns = defaultdict(list)
    skipped = 0
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id not in COCO_VALID_CAT_IDS:
            skipped += 1
            continue
        x, y, w, h = ann["bbox"]
        img_anns[ann["image_id"]].append({
            "bbox_xyxy": [x, y, x + w, y + h],
            "category_id": cat_id,
            "class_name": COCO_CAT_ID_TO_NAME[cat_id],
            "yolo_class_id": CLASS_NAME_TO_ID[COCO_CAT_ID_TO_NAME[cat_id]],
        })

    result = {}
    for img_id, anns in img_anns.items():
        if img_id in id_to_img:
            result[id_to_img[img_id]] = anns

    print(f"  有效标注: {sum(len(v) for v in result.values())} 个")
    print(f"  过滤掉(元数据): {skipped} 个")
    print(f"  有标注的图片: {len(result)} 张")
    return result


def bbox_to_ellipse_mask(bbox_xyxy, img_h, img_w):
    """bbox → 椭圆近似 mask"""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    a = max((x2 - x1) / 2.0, 1.0)
    b = max((y2 - y1) / 2.0, 1.0)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, 255, -1)
    return mask


def mask_to_polygon(mask, epsilon_factor=0.005):
    """二值 mask → 归一化多边形顶点列表"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_factor * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    h, w = mask.shape
    points = approx.reshape(-1, 2).astype(np.float32)
    points[:, 0] /= w
    points[:, 1] /= h
    return points.tolist()


def generate_masks_with_sam(sam_model, image_path, bboxes_xyxy, h, w):
    """MobileSAM: 对一张图的多个bbox生成mask"""
    masks = []
    for bbox in bboxes_xyxy:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            masks.append(bbox_to_ellipse_mask(bbox, h, w))
            continue
        try:
            results = sam_model(str(image_path), bboxes=[[x1, y1, x2, y2]], verbose=False)
            if results[0].masks is not None and len(results[0].masks.data) > 0:
                m = results[0].masks.data[0].cpu().numpy()
                m = (m > SAM_CONFIDENCE_THRESHOLD).astype(np.uint8) * 255
                masks.append(m)
            else:
                masks.append(bbox_to_ellipse_mask(bbox, h, w))
        except Exception:
            masks.append(bbox_to_ellipse_mask(bbox, h, w))
    return masks


def main():
    print("=" * 60)
    print("  01_generate_masks.py — COCO bbox → MobileSAM mask")
    print("=" * 60)
    print()

    # ---- 加载COCO标注 ----
    print("[1/4] 加载 COCO 标注...")
    all_annotations = {}

    for split_name, split_dir in [("train", COCO_TRAIN_DIR), ("valid", COCO_VALID_DIR)]:
        json_path = split_dir / "_annotations.coco.json"
        if json_path.exists():
            print(f"\n  {split_name} ({split_dir}):")
            anns = load_coco_annotations(json_path, split_dir)
            for fname, a in anns.items():
                if fname not in all_annotations:
                    all_annotations[fname] = a
        else:
            print(f"  ⚠ 未找到: {json_path}")

    if not all_annotations:
        print("✗ 没有有效标注数据!")
        return

    print(f"\n  去重后总计: {len(all_annotations)} 张图片有缺陷标注, "
          f"{sum(len(v) for v in all_annotations.values())} 个标注")

    # ---- 加载 MobileSAM ----
    print("\n[2/4] 加载 MobileSAM...")
    sam_model = None
    try:
        from ultralytics import SAM
        sam_model = SAM(MOBILE_SAM_MODEL)
        print(f"  ✓ MobileSAM 加载成功")
    except Exception as e:
        print(f"  ✗ MobileSAM 加载失败: {e}")
        print(f"  → 使用椭圆mask回退方案 (效果可接受)")

    # ---- 构建文件名→路径映射 (train + valid 两个目录) ----
    print("\n[3/4] 生成分割 mask...")
    all_image_files = {}
    for d in [COCO_TRAIN_DIR, COCO_VALID_DIR]:
        for f in d.glob("*.jpg"):
            if f.name not in all_image_files:
                all_image_files[f.name] = f

    stats = {"total_objects": 0, "total_images": 0, "missing": 0}

    masks_cache = {}
    for fname, anns in tqdm(list(all_annotations.items()), desc="  MobileSAM"):
        if fname not in all_image_files:
            stats["missing"] += 1
            continue

        img_path = all_image_files[fname]
        img = cv2.imread(str(img_path))
        if img is None:
            stats["missing"] += 1
            continue

        h, w = img.shape[:2]
        bboxes = [a["bbox_xyxy"] for a in anns]

        # SAM 或椭圆回退
        if sam_model is not None:
            masks = generate_masks_with_sam(sam_model, img_path, bboxes, h, w)
        else:
            masks = [bbox_to_ellipse_mask(b, h, w) for b in bboxes]

        # 转为 polygon
        img_masks = []
        for ann, mask in zip(anns, masks):
            polygon = mask_to_polygon(mask)
            if polygon:
                img_masks.append({
                    "class_name": ann["class_name"],
                    "class_id": ann["yolo_class_id"],
                    "polygon": polygon,
                    "bbox": ann["bbox_xyxy"],
                })
                stats["total_objects"] += 1

        if img_masks:
            masks_cache[fname] = img_masks
            stats["total_images"] += 1

    print(f"\n  有mask的图片: {stats['total_images']} 张, 总目标: {stats['total_objects']} 个")
    if stats["missing"]:
        print(f"  ⚠ 缺失图片: {stats['missing']} 张")

    # ---- 各类别统计 ----
    cls_count = Counter()
    for anns in masks_cache.values():
        for a in anns:
            cls_count[a["class_name"]] += 1
    print(f"\n  各类别数量:")
    for name in CLASS_NAMES:
        print(f"    {name}: {cls_count.get(name, 0)}")

    # ---- 保存 ----
    print("\n[4/4] 保存缓存...")
    cache_file = MASKS_CACHE_DIR / "masks_cache.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(masks_cache, f)
    print(f"  ✓ {cache_file}")
    print()
    print("=" * 60)
    print("  Mask 生成完成！下一步: python scripts/02_prepare_dataset.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
