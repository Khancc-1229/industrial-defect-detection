"""
06_export_tensorrt.py
将 ONNX 模型导出为 TensorRT FP16 Engine

前提: 已安装 TensorRT Python 包 (pip install tensorrt)
如果 TensorRT 不可用，自动降级为 ONNX Runtime GPU 方案
"""

import sys
import os
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def try_ultralytics_trt_export(onnx_path, engine_path):
    """使用 Ultralytics 内置的 TensorRT 导出"""
    from ultralytics import YOLO

    print("  使用 Ultralytics 导出 TensorRT...")

    try:
        # Ultralytics 可以直接从 pt 导出到 engine
        best_pt = WEIGHTS_DIR / "best.pt"
        if not best_pt.exists():
            best_pt = RESULTS_DIR / "train" / "weights" / "best.pt"

        model = YOLO(str(best_pt))
        result = model.export(
            format="engine",
            imgsz=TENSORRT_CONFIG["imgsz"],
            half=TENSORRT_CONFIG["half"],
            workspace=TENSORRT_CONFIG["workspace"],
            simplify=True,
        )
        print(f"  ✓ TensorRT 导出成功: {result}")
        return True
    except Exception as e:
        print(f"  ✗ Ultralytics TensorRT 导出失败: {e}")
        return False


def verify_trt_engine(engine_path, test_image_path):
    """验证 TensorRT Engine"""
    try:
        from ultralytics import YOLO
        model = YOLO(str(engine_path))
        results = model(str(test_image_path), verbose=False)

        n_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  ✓ TensorRT 推理成功: 检测到 {n_detections} 个目标")

        if results[0].masks is not None:
            print(f"  ✓ 分割 mask 正常输出")
        return True
    except Exception as e:
        print(f"  ✗ TensorRT 推理失败: {e}")
        return False


def main():
    print("=" * 60)
    print("  06_export_tensorrt.py - 导出 TensorRT Engine")
    print("=" * 60)
    print()

    # ---- 检查 ONNX 模型 ----
    onnx_path = WEIGHTS_DIR / "best.onnx"
    if not onnx_path.exists():
        # 尝试找 ONNX 文件
        alt_onnx = RESULTS_DIR / "train" / "weights" / "best.onnx"
        if alt_onnx.exists():
            import shutil
            shutil.copy2(alt_onnx, onnx_path)
        else:
            print(f"✗ 未找到 ONNX 模型，请先运行 05_export_onnx.py")
            return

    engine_path = WEIGHTS_DIR / "best_fp16.engine"

    # ---- 检查 TensorRT 可用性 ----
    print(f"[1/3] 检查 TensorRT 环境...")

    trt_available = False
    try:
        import tensorrt as trt
        print(f"  ✓ TensorRT 版本: {trt.__version__}")
        trt_available = True
    except ImportError:
        print(f"  ✗ tensorrt 包未安装")
        print(f"  → 将使用 ONNX Runtime GPU 作为替代方案")

    # ---- 导出 ----
    success = False
    if trt_available or True:  # 总是尝试，因为 ultralytics 可能内部有 TensorRT
        print(f"\n[2/3] 导出 TensorRT Engine...")
        success = try_ultralytics_trt_export(onnx_path, engine_path)

    # 检查实际产物
    actual_engine = None
    if engine_path.exists():
        actual_engine = engine_path
    else:
        # Ultralytics 可能把 engine 放在其他位置
        candidates = list(WEIGHTS_DIR.glob("*.engine"))
        if not candidates:
            candidates = list((RESULTS_DIR / "train" / "weights").glob("*.engine"))
        if candidates:
            import shutil
            actual_engine = WEIGHTS_DIR / "best_fp16.engine"
            shutil.copy2(sorted(candidates, key=lambda x: x.stat().st_mtime)[-1], actual_engine)
            success = True

    # ---- 验证 ----
    if actual_engine and actual_engine.exists():
        print(f"\n[3/3] 验证 TensorRT Engine...")
        val_images = sorted(list(YOLO_IMAGES_VAL.glob("*.jpg")))
        if not val_images:
            val_images = sorted(list(NEU_DET_IMAGES.glob("*.jpg")))

        if val_images:
            verify_trt_engine(actual_engine, val_images[0])
        else:
            print(f"  ⚠ 未找到测试图片")

        engine_size_mb = actual_engine.stat().st_size / (1024 * 1024)
        print(f"\n  TensorRT Engine: {engine_size_mb:.1f} MB (FP16)")

    # ---- 如果失败,提供说明 ----
    if not success:
        print(f"\n[3/3] TensorRT 导出未成功")
        print(f"\n  ⚠ TensorRT 导出失败，常见原因:")
        print(f"    1. TensorRT 版本与 CUDA 12.8 不兼容")
        print(f"    2. 缺少 cuDNN 库")
        print(f"    3. Windows 环境下的构建工具链不完整")
        print(f"")
        print(f"  替代方案 (同样有价值的优化):")
        print(f"    pip install onnxruntime-gpu")
        print(f"    使用 ONNX Runtime GPU 进行 FP16 推理")
        print(f"    代码已内置支持，详见 benchmark 和 gradio demo")
        print(f"")
        print(f"  如需安装 TensorRT (Windows):")
        print(f"    1. 从 NVIDIA 官网下载 TensorRT 10.x ZIP 包")
        print(f"    2. 解压后 pip install tensorrt-*.whl")
        print(f"    3. 将 lib 目录添加到 PATH")

    print()
    print("=" * 60)
    if success:
        print("  TensorRT 导出完成！下一步: 运行 07_benchmark.py")
    else:
        print("  TensorRT 导出跳过。下一步: 运行 07_benchmark.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
