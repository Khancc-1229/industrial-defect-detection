"""
Gradio Web Demo - 工业缺陷检测与分割系统

功能:
  - 图片上传 (拖拽/点击)
  - 后端选择: PyTorch / ONNX / TensorRT
  - 推理结果显示: 分割叠加图 + 检测结果表格 + 统计信息
  - 支持示例图片

用法:
  python gradio_app.py
  浏览器访问 http://localhost:7860
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

# 添加 scripts 目录
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from config import *


class DefectDetector:
    """缺陷检测器 - 管理多后端模型"""

    def __init__(self):
        self.models = {}
        self.available_backends = []

        # 尝试加载各后端
        pt_path = WEIGHTS_DIR / "best.pt"
        if not pt_path.exists():
            pt_path = RESULTS_DIR / "train" / "weights" / "best.pt"

        onnx_path = WEIGHTS_DIR / "best.onnx"
        engine_path = WEIGHTS_DIR / "best_fp16.engine"

        model_configs = [
            (pt_path, "PyTorch"),
            (onnx_path, "ONNX"),
            (engine_path, "TensorRT"),
        ]

        from ultralytics import YOLO

        for model_path, name in model_configs:
            if model_path.exists():
                try:
                    self.models[name] = YOLO(str(model_path))
                    self.available_backends.append(name)
                    print(f"  ✓ {name}: {model_path}")
                except Exception as e:
                    print(f"  ✗ {name} 加载失败: {e}")

        if not self.models:
            raise RuntimeError("没有任何可用模型！请先运行训练和导出脚本")

        print(f"\n  可用后端: {self.available_backends}")

    def predict(self, image, backend, conf_threshold=0.25, iou_threshold=0.45):
        """执行推理"""
        if backend not in self.models:
            backend = self.available_backends[0]

        model = self.models[backend]
        results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)
        return results[0]

    def get_available_backends(self):
        return self.available_backends


# 颜色映射（6类 + 背景）
CLASS_COLORS = [
    (255, 64, 64),     # crazing - 红
    (64, 255, 64),     # inclusion - 绿
    (64, 64, 255),     # patches - 蓝
    (255, 255, 64),    # pitted_surface - 黄
    (255, 64, 255),    # rolled-in_scale - 品红
    (64, 255, 255),    # scratches - 青
]


def draw_segmentation_overlay(image, result):
    """在图像上绘制分割 overlay + bbox"""
    img = np.array(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    overlay = img.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        classes = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

        for i, (mask, cls_id, conf) in enumerate(zip(masks, classes, confs)):
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]

            # Resize mask to image size
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).astype(np.uint8)

            # 绘制半透明 mask overlay
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_binary,
                    cv2.addWeighted(overlay[:, :, c], 0.5, np.full_like(overlay[:, :, c], color[c]), 0.5, 0),
                    overlay[:, :, c]
                )

            # 绘制 mask 轮廓
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # 绘制 bbox
            if boxes.xyxy is not None and i < len(boxes.xyxy):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # 标签
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1, y1 - label_h - 6), (x1 + label_w + 4, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 混合
    result_img = cv2.addWeighted(img, 0.3, overlay, 0.7, 0)
    return result_img


def process_image(image, backend, conf_threshold):
    """处理上传图片的完整流程"""
    if image is None:
        return None, "请上传图片", None

    try:
        result = detector.predict(image, backend, conf_threshold)

        # 生成分割叠加图
        overlay_img = draw_segmentation_overlay(image, result)

        # 生成检测结果文本
        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            # 统计各类别数量
            from collections import Counter
            class_counts = Counter(classes)

            lines = ["### 📊 检测结果\n"]
            lines.append(f"| 类别 | 置信度 | 数量 |")
            lines.append(f"|------|--------|------|")

            for cls_id, count in sorted(class_counts.items()):
                cls_confs = confs[classes == cls_id]
                lines.append(f"| {CLASS_NAMES[cls_id]} | "
                           f"{cls_confs.max():.3f} (最高) | {count} |")

            lines.append(f"\n**总计: {len(classes)} 个缺陷**")
            lines.append(f"**推理后端: {backend}**")

            result_text = "\n".join(lines)
        else:
            result_text = "### ✅ 未检测到缺陷\n\n该图片没有发现任何表面缺陷。"

        # 生成详细的检测表格数据
        if result.boxes is not None and len(result.boxes) > 0:
            table_data = []
            xyxy = result.boxes.xyxy.cpu().numpy()
            for i, (cls_id, conf, bbox) in enumerate(zip(classes, confs, xyxy)):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                area = w * h
                table_data.append([
                    f"#{i+1}",
                    CLASS_NAMES[cls_id],
                    f"{conf:.3f}",
                    f"({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})",
                    f"{int(w)}×{int(h)} px",
                    f"{int(area)} px²",
                ])
        else:
            table_data = []

        return overlay_img, result_text, table_data

    except Exception as e:
        import traceback
        error_msg = f"### ❌ 推理错误\n\n```\n{traceback.format_exc()}\n```"
        return None, error_msg, None


def load_example_images():
    """加载示例图片（从验证集随机选）"""
    val_dir = YOLO_IMAGES_VAL
    if not val_dir.exists():
        return []

    examples = sorted(list(val_dir.glob("*.jpg")))[:6]
    return [str(p) for p in examples]


# ========== 创建 Gradio 界面 ==========

# 初始化检测器
print("加载模型...")
detector = DefectDetector()
print(f"初始化完成，可用后端: {detector.available_backends}\n")

default_backend = detector.available_backends[0] if detector.available_backends else "PyTorch"


# 构建界面
with gr.Blocks(title=GRADIO_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""# 🔍 {GRADIO_TITLE}

{GRADIO_DESCRIPTION}

**可用后端**: {', '.join(detector.available_backends)}
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 输入区
            input_image = gr.Image(
                label="📤 上传图片",
                type="numpy",
                sources=["upload", "clipboard"],
            )

            with gr.Row():
                backend_selector = gr.Dropdown(
                    choices=detector.available_backends,
                    value=default_backend,
                    label="⚙️ 推理后端",
                    interactive=True,
                )
                conf_slider = gr.Slider(
                    minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                    label="🎯 置信度阈值",
                )

            submit_btn = gr.Button("🚀 开始检测", variant="primary", size="lg")

            # 示例图片
            example_paths = load_example_images()
            if example_paths:
                gr.Examples(
                    examples=example_paths,
                    inputs=input_image,
                    label="📸 示例图片",
                )

        with gr.Column(scale=1):
            # 输出区
            output_image = gr.Image(
                label="📊 分割结果",
                type="numpy",
                format="png",
            )
            output_text = gr.Markdown("等待输入...")

    # 检测详情表格
    gr.Markdown("---")
    gr.Markdown("### 📋 检测详情")
    output_table = gr.Dataframe(
        headers=["序号", "类别", "置信度", "位置", "尺寸", "面积"],
        label="检测目标列表",
        interactive=False,
    )

    # 绑定事件
    submit_btn.click(
        fn=process_image,
        inputs=[input_image, backend_selector, conf_slider],
        outputs=[output_image, output_text, output_table],
    )


if __name__ == "__main__":
    print(f"\n启动 Gradio 服务器...")
    print(f"访问地址: http://localhost:{GRADIO_SERVER_PORT}")
    demo.launch(
        server_port=GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
    )
