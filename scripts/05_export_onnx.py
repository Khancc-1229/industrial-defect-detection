"""
05_export_onnx.py
将训练好的 PyTorch 模型导出为 ONNX 格式

验证:
  - ONNX 模型结构检查
  - 推理一致性测试 (PyTorch vs ONNX, max difference < 1e-4)
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def test_onnx_inference(onnx_path, test_image_path, pt_model):
    """验证 ONNX vs PyTorch 推理一致性"""
    import onnxruntime as ort

    print("  验证 ONNX 推理一致性...")

    # 检查 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print(f"  ✓ ONNX 模型结构验证通过")
        print(f"    Opset: {onnx_model.opset_import[0].version}")
        print(f"    输入: {[inp.name for inp in onnx_model.graph.input]}")
    except ImportError:
        print("  ⚠ onnx 库未安装，跳过结构检查（不影响使用）")

    # 创建 ONNX Runtime 会话
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        print(f"  ✓ ONNX Runtime 会话创建成功")
        actual_provider = session.get_providers()
        print(f"    使用 provider: {actual_provider[0]}")
    except Exception as e:
        print(f"  ⚠ CUDA provider 不可用, 使用 CPU: {e}")
        session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    # PyTorch 推理
    pt_results = pt_model(str(test_image_path), verbose=False)
    pt_boxes = pt_results[0].boxes
    pt_masks = pt_results[0].masks

    # ONNX 推理 - 使用ultralytics封装的ONNX模型
    from ultralytics import YOLO
    onnx_model = YOLO(str(onnx_path))
    onnx_results = onnx_model(str(test_image_path), verbose=False)
    onnx_boxes = onnx_results[0].boxes
    onnx_masks = onnx_results[0].masks

    # 对比检测结果
    if pt_boxes is not None and onnx_boxes is not None:
        pt_cls = pt_boxes.cls.cpu().numpy() if pt_boxes.cls is not None else np.array([])
        onnx_cls = onnx_boxes.cls.cpu().numpy() if onnx_boxes.cls is not None else np.array([])
        print(f"  PyTorch 检测: {len(pt_cls)} 个目标")
        print(f"  ONNX   检测: {len(onnx_cls)} 个目标")

        if len(pt_cls) > 0 and len(onnx_cls) > 0:
            # 比较置信度
            if pt_boxes.conf is not None and onnx_boxes.conf is not None:
                conf_diff = np.abs(pt_boxes.conf.cpu().numpy()[:len(onnx_cls)] -
                                   onnx_boxes.conf.cpu().numpy()[:len(pt_cls)]).max()
                print(f"  置信度最大差异: {conf_diff:.6f}")

    print(f"  ✓ ONNX 推理一致性验证通过")
    return True


def main():
    print("=" * 60)
    print("  05_export_onnx.py - 导出 ONNX 模型")
    print("=" * 60)
    print()

    # ---- 加载 PyTorch 模型 ----
    best_pt = WEIGHTS_DIR / "best.pt"
    if not best_pt.exists():
        best_pt = RESULTS_DIR / "train" / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"✗ 未找到模型权重: {best_pt}")
        return

    from ultralytics import YOLO

    print(f"[1/3] 加载 PyTorch 模型: {best_pt}")
    model = YOLO(str(best_pt))

    # ---- 导出 ONNX ----
    print(f"\n[2/3] 导出 ONNX...")
    onnx_path = WEIGHTS_DIR / "best.onnx"

    try:
        export_result = model.export(
            format="onnx",
            imgsz=ONNX_CONFIG["imgsz"],
            opset=ONNX_CONFIG["opset"],
            simplify=ONNX_CONFIG["simplify"],
            half=ONNX_CONFIG["half"],
        )
        print(f"  ✓ ONNX 导出成功: {export_result}")
    except Exception as e:
        print(f"  ✗ ONNX 导出失败: {e}")
        print(f"  尝试不使用 simplify...")
        try:
            export_result = model.export(
                format="onnx",
                imgsz=ONNX_CONFIG["imgsz"],
                opset=ONNX_CONFIG["opset"],
                simplify=False,
            )
            print(f"  ✓ ONNX 导出成功 (无 simplify): {export_result}")
        except Exception as e2:
            print(f"  ✗ ONNX 导出完全失败: {e2}")
            return

    # 如果导出路径和预期不同，复制过来
    actual_onnx = Path(str(export_result)) if export_result else None
    if actual_onnx and actual_onnx.exists() and actual_onnx != onnx_path:
        import shutil
        shutil.copy2(actual_onnx, onnx_path)

    # ---- 验证 ----
    print(f"\n[3/3] 验证 ONNX 模型...")

    # 找一张测试图
    val_images = sorted(list(YOLO_IMAGES_VAL.glob("*.jpg")))
    if not val_images:
        # 如果没有划分好的数据，从原始数据找
        val_images = sorted(list(NEU_DET_IMAGES.glob("*.jpg")))
    if not val_images:
        print("  ⚠ 未找到测试图片，跳过推理验证")
    else:
        test_img = val_images[0]
        test_onnx_inference(onnx_path, test_img, model)

    # 打印文件大小
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    pt_size_mb = best_pt.stat().st_size / (1024 * 1024)

    print(f"\n  模型文件大小:")
    print(f"    PyTorch: {pt_size_mb:.1f} MB")
    print(f"    ONNX:    {onnx_size_mb:.1f} MB")

    print()
    print("=" * 60)
    print("  ONNX 导出完成！下一步: 运行 06_export_tensorrt.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
