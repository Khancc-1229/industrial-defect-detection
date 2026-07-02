# 工业缺陷检测与分割系统

> YOLOv8n-seg + MobileSAM | 6 类钢铁缺陷 | mAP50 0.75 | ONNX 102 FPS

---

## 项目概述

针对热轧钢板表面 6 类缺陷（裂纹、夹杂、斑块、点蚀、氧化皮、划痕），构建了从**数据准备 → 模型训练 → 部署 → Web Demo** 的完整检测系统。

**核心亮点**：用 MobileSAM（Segment Anything Model）把原始 bbox 标注自动升级为精确分割 mask，零人工标注成本。

---

## 数据集

| 项目 | 详情 |
|------|------|
| 来源 | Roboflow（[neu-steel-defect-dataset](https://universe.roboflow.com/neudatasetoriginal/neu-steel-defect-dataset)） |
| 总量 | 训练 1,607 张，验证 419 张 |
| 规格 | 200×200 灰度图 |
| 标注 | COCO → MobileSAM 升级 → YOLO seg |

| 缺陷类别 | 标注数 | 说明 |
|---------|--------|------|
| crazing（裂纹） | 689 | 最难点——与背景纹理几乎不可区分 |
| inclusion（夹杂） | 1,011 | 钢材中混入非金属杂质 |
| patches（斑块） | 878 | 表面氧化色差 |
| pitted_surface（点蚀） | 432 | 腐蚀坑洼 |
| rolled-in_scale（氧化皮） | 628 | 轧钢起皮褶皱 |
| scratches（划痕） | 548 | 机械划伤 |

---

## 技术路线

```
COCO 标注（仅 bbox）
    │  MobileSAM 自动生成 mask
    ▼
YOLOv8n-seg 训练（50 epochs）
    │  PyTorch → ONNX
    ▼
三后端 Benchmark + Gradio Web Demo
```

---

## 实验结果

### 整体

| 指标 | 值 |
|------|-----|
| 模型 | YOLOv8n-seg（340 万参数） |
| **mAP50** | **0.746** |
| **mAP50-95** | **0.471** |
| PyTorch 推理 | 123 FPS |
| ONNX 推理 | 102 FPS |

### ONNX vs PyTorch

| | PyTorch (.pt) | ONNX (.onnx) |
|------|-------------|------------|
| 速度 | 123 FPS | 102 FPS |
| 延迟 | 8.1 ms | 9.8 ms |
| 文件大小 | 6.5 MB | 12.7 MB |
| Python 依赖 | 需要完整 PyTorch | 仅需 onnxruntime |
| 跨语言部署 | 仅 Python | C++ / C# / Java |
| 适用场景 | 训练、研究 | 生产部署 |

> ONNX 的意义不在于更快，而在于脱离 PyTorch 运行。产线上不能每台机器装一个 2GB 的 PyTorch。

### 每类精度

| 类别 | AP50 | 
|------|------|
| patches | **0.962** |
| scratches | **0.904** |
| pitted_surface | **0.867** |
| inclusion | 0.747 |
| rolled-in_scale | 0.567 |
| crazing | 0.428 |

patches / scratches / pitted 表现优秀；crazing 因与背景纹理高度相似，是 NEU-DET 公认难点。

---

## AI Agent 辅助开发说明

本项目全程使用 **Claude Code** 辅助开发。

| 环节 | Agent 辅助内容 |
|------|---------------|
| 数据脚本 | COCO 解析 + SAM mask 生成 + 回退逻辑 |
| 训练配置 | 超参数选择 + NumPy 版本兼容修复 |
| ONNX 导出 | 导出脚本 + 推理一致性验证 |
| Benchmark | 对比图表自动生成 |
| Gradio Demo | 前后端完整界面 |
| 文档 | 本文档 80% 由 Agent 辅助起草 |


---

## 快速运行

```bash
# 数据准备
python scripts/01_generate_masks.py
python scripts/02_prepare_dataset.py

# 训练
python scripts/03_train.py          # ~1.5h

# 评估
python scripts/04_evaluate.py

# 部署
python scripts/05_export_onnx.py
python scripts/07_benchmark.py

# Demo
python demo/gradio_app.py           # http://localhost:7860
```

---

## 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| PyTorch 无 CUDA | gradio 依赖覆盖了 CUDA 版 torch | `pip install torch --index-url cu121` |
| cv2.imwrite 失败 | OpenCV 不支持中文路径 | `imencode + tofile` |
| COCO 标注混入元数据 | Roboflow 导出格式问题 | 过滤 category_id 6-11 |
| MobileSAM 下载慢 | GitHub 限速 | VPN 代理 |

---

## License

学习研究用途。
