"""
07_benchmark.py
三后端推理速度 & 精度对比

对比对象: PyTorch (.pt) / ONNX Runtime (.onnx) / TensorRT (.engine)
指标: FPS, 平均延迟, 延迟分布, 检测数一致性
"""

import sys
import os
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def benchmark_backend(model_path, image_paths, backend_name, warmup=BENCHMARK_WARMUP):
    """
    对指定后端做 benchmark
    返回: {fps, latencies[], n_detections[]}
    """
    from ultralytics import YOLO

    if not Path(model_path).exists():
        print(f"  ✗ 模型不存在: {model_path}")
        return None

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None

    latencies = []
    n_detections = []

    # Warmup
    for _ in range(warmup):
        _ = model(str(image_paths[0]), verbose=False)

    # Benchmark
    for img_path in tqdm(image_paths, desc=f"  {backend_name}"):
        t0 = time.perf_counter()
        results = model(str(img_path), verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        latencies.append(elapsed)

        n = len(results[0].boxes) if results[0].boxes is not None else 0
        n_detections.append(n)

    latencies = np.array(latencies)
    fps = 1000.0 / latencies.mean()

    return {
        "backend": backend_name,
        "model_path": str(model_path),
        "num_images": len(image_paths),
        "fps": round(float(fps), 2),
        "latency_mean_ms": round(float(latencies.mean()), 3),
        "latency_std_ms": round(float(latencies.std()), 3),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_min_ms": round(float(latencies.min()), 3),
        "latency_max_ms": round(float(latencies.max()), 3),
        "avg_detections": round(float(np.mean(n_detections)), 1),
        "latencies": latencies.tolist(),  # 完整数据用于画图
    }


def plot_benchmark(results_list, save_dir):
    """绘制对比图"""
    valid = [r for r in results_list if r is not None]
    if not valid:
        print("  ⚠ 没有可用的 benchmark 结果")
        return

    backends = [r["backend"] for r in valid]
    colors = {"PyTorch": "#2196F3", "ONNX": "#FF9800", "TensorRT": "#4CAF50"}

    # --- FPS 柱状图 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. FPS
    ax = axes[0]
    fps_vals = [r["fps"] for r in valid]
    bars = ax.bar(backends, fps_vals, color=[colors.get(b, "#999") for b in backends], edgecolor="white")
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(fps_vals)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("FPS (越高越好)")
    ax.set_title("推理速度对比")
    ax.grid(axis='y', alpha=0.3)

    # 2. Latency
    ax = axes[1]
    lat_means = [r["latency_mean_ms"] for r in valid]
    bars = ax.bar(backends, lat_means, color=[colors.get(b, "#999") for b in backends], edgecolor="white")
    for bar, val in zip(bars, lat_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(lat_means)*0.02,
                f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("Latency (ms, 越低越好)")
    ax.set_title("平均推理延迟")
    ax.grid(axis='y', alpha=0.3)

    # 3. Latency Distribution (boxplot)
    ax = axes[2]
    lat_data = [r["latencies"] for r in valid]
    bp = ax.boxplot(lat_data, labels=backends, patch_artist=True)
    for patch, b in zip(bp['boxes'], backends):
        patch.set_facecolor(colors.get(b, "#999"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("延迟分布")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / "benchmark_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Benchmark 对比图: {save_path}")

    # --- 速度对比表 ---
    fig, ax = plt.subplots(figsize=(10, 2 + 0.3 * len(valid)))
    ax.axis("off")

    table_data = []
    headers = ["后端", "FPS↑", "平均延迟", "P95延迟", "标准差", "平均检测数"]
    for r in valid:
        table_data.append([
            r["backend"],
            f"{r['fps']:.1f}",
            f"{r['latency_mean_ms']:.1f} ms",
            f"{r['latency_p95_ms']:.1f} ms",
            f"{r['latency_std_ms']:.1f} ms",
            f"{r['avg_detections']:.1f}",
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc="center", loc="center",
                     colColours=["#E3F2FD"] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # 高亮最快行
    best_idx = np.argmax([r["fps"] for r in valid])
    for j in range(len(headers)):
        table[(best_idx + 1, j)].set_facecolor("#C8E6C9")

    save_path = save_dir / "benchmark_table.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Benchmark 表格: {save_path}")


def main():
    print("=" * 60)
    print("  07_benchmark.py - 三后端推理 Benchmark")
    print("=" * 60)
    print()

    # ---- 查找模型文件 ----
    pt_path = WEIGHTS_DIR / "best.pt"
    if not pt_path.exists():
        pt_path = RESULTS_DIR / "train" / "weights" / "best.pt"
    onnx_path = WEIGHTS_DIR / "best.onnx"
    engine_path = WEIGHTS_DIR / "best_fp16.engine"

    models_to_bench = []
    if pt_path.exists():
        models_to_bench.append((pt_path, "PyTorch"))
    else:
        print("✗ 未找到 PyTorch 模型")

    if onnx_path.exists():
        models_to_bench.append((onnx_path, "ONNX"))
    else:
        print("⚠ 未找到 ONNX 模型 (跳过)")

    if engine_path.exists():
        models_to_bench.append((engine_path, "TensorRT"))
    else:
        print("⚠ 未找到 TensorRT Engine (跳过)")

    if not models_to_bench:
        print("✗ 没有任何可用模型，请先运行训练和导出脚本")
        return

    print(f"\n将对比 {len(models_to_bench)} 个后端: {[m[1] for m in models_to_bench]}")

    # ---- 准备测试图片 ----
    val_images = sorted(list(YOLO_IMAGES_VAL.glob("*.jpg")))
    if len(val_images) < BENCHMARK_NUM_IMAGES:
        # 不够的话从训练集补
        train_images = sorted(list(YOLO_IMAGES_TRAIN.glob("*.jpg")))
        val_images = val_images + train_images[:BENCHMARK_NUM_IMAGES - len(val_images)]
    val_images = val_images[:BENCHMARK_NUM_IMAGES]

    if len(val_images) < 10:
        print(f"✗ 测试图片不足 ({len(val_images)} 张)")
        return

    print(f"测试图片: {len(val_images)} 张")
    print(f"Warmup: {BENCHMARK_WARMUP} 次\n")

    # ---- 运行 Benchmark ----
    results_list = []
    for model_path, backend_name in models_to_bench:
        print(f"Benchmark: {backend_name}")
        result = benchmark_backend(model_path, val_images, backend_name)
        if result:
            results_list.append(result)
            print(f"  FPS: {result['fps']:.1f}, 平均延迟: {result['latency_mean_ms']:.1f}ms, "
                  f"P95: {result['latency_p95_ms']:.1f}ms, 平均检测: {result['avg_detections']:.1f}")
        print()

    # ---- 保存结果 ----
    results_json = {r["backend"]: {k: v for k, v in r.items() if k != "latencies"}
                    for r in results_list}
    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果 JSON: {json_path}")

    # ---- 绘制对比图 ----
    print(f"\n绘制对比图表...")
    plot_benchmark(results_list, RESULTS_DIR)

    # ---- 汇总 ----
    print(f"\n{'='*60}")
    print(f"  Benchmark 汇总")
    print(f"{'='*60}")
    header = f"  {'后端':<12s} {'FPS':>8s} {'平均延迟':>10s} {'P95延迟':>10s} {'检测数':>8s}"
    print(header)
    print(f"  {'-'*48}")
    for r in results_list:
        print(f"  {r['backend']:<12s} {r['fps']:>8.1f} {r['latency_mean_ms']:>9.1f}ms "
              f"{r['latency_p95_ms']:>9.1f}ms {r['avg_detections']:>8.1f}")

    if len(results_list) >= 2:
        speedup = results_list[1]["fps"] / results_list[0]["fps"] if results_list[0]["fps"] > 0 else 0
        print(f"\n  ONNX vs PyTorch 加速比: {speedup:.2f}x")

    if len(results_list) >= 3:
        speedup = results_list[2]["fps"] / results_list[0]["fps"] if results_list[0]["fps"] > 0 else 0
        print(f"  TensorRT vs PyTorch 加速比: {speedup:.2f}x")

    print()
    print("=" * 60)
    print("  Benchmark 完成！下一步: 运行 gradio_app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
