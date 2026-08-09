# 工业缺陷检测系统

## 这是什么
基于 YOLOv8 的工业缺陷检测系统，从数据准备、模型训练到 Web 部署的完整项目。

## 为什么做这个项目
暑假自学深度学习的实战项目，熟悉：
- 目标检测怎么训（YOLOv8 训练 + 调参）
- 数据怎么准备（YOLO 格式标注 + dataset.yaml）
- 模型怎么部署（FastAPI RESTful 接口 + Gradio 可视化界面）
- 工程化怎么做（Docker 容器化 + Git 版本控制）
- 从 0 到 1 的完整项目流程

## 运行环境
- Python 3.x
- ultralytics（YOLOv8）
- fastapi + uvicorn
- gradio
- Pillow

安装命令：
    pip install ultralytics fastapi uvicorn gradio pillow

## 文件说明

| 文件 | 说明 |
|------|------|
| `train_defect.py` | YOLOv8 训练脚本，50 轮，CPU 训练 |
| `dataset.yaml` | 数据集配置文件（手写） |
| `main.py` | FastAPI 推理接口，POST/GET |
| `app.py` | Gradio 可视化网页界面 |
| `test_model_inference.py` | 模型预测演示脚本 |
| `best.pt` | 训练好的模型权重（6.3MB） |
| `Dockerfile` | Docker 容器化配置（待构建） |
| `requirements.txt` | Python 依赖清单 |

## 运行方式

### 1. 训练模型
    python train_defect.py

### 2. 启动 FastAPI 接口
    python main.py
浏览器访问 `http://localhost:8000/docs` 测试 Swagger UI

### 3. 启动 Gradio 网页界面
    python app.py
浏览器访问 `http://localhost:7860` 上传图片检测

### 4. 快速预测测试
    python test_model_inference.py

## 训练结果

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 89.6% |
| mAP@0.5:0.95 | 74.0% |
| Precision | 76.8% |
| Recall | 91.9% |
| 训练耗时 | 46 分钟（CPU，batch=4） |
| 模型大小 | 6.3MB |
| 推理速度 | ~120-170ms/张（CPU） |

训练日志：
- Epoch 1：loss 快速下降，mAP 开始爬升
- Epoch 25：loss 趋于平稳，验证指标稳定
- Epoch 50：最终 mAP@0.5 达到 89.6%

## 推理验证

FastAPI 接口返回示例：
```json
{
  "filename": "defect_004.jpg",
  "defect_count": 4,
  "defects": [
    {
      "type": "defect",
      "confidence": 0.5359,
      "bbox": [260.63, 323.05, 307.2, 437.19]
    }
  ]
}