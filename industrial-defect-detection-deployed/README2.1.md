# 项目2.1：工业缺陷检测系统（2026.8）

> **在线体验**：https://www.modelscope.cn/studios/AAAAgi/industrial-defect-detection

## 这是什么
基于 YOLOv8 的工业缺陷检测系统，从数据准备、模型训练到 Web 部署的完整项目。

## 为什么做这个项目
暑假自学深度学习的第二个实战项目，熟悉：
- 目标检测怎么训（YOLOv8 调用预训练权重 + 微调）
- 数据怎么准备（YOLO 格式标注 + dataset.yaml）
- 模型怎么部署（FastAPI RESTful 接口 + Gradio 可视化界面）
- 云端部署怎么做（平台选择、环境兼容、故障排查）
- 从 0 到 1 的完整项目流程

## 技术栈
- **目标检测**：YOLOv8（调用官方预训练权重，在工业缺陷数据集上微调）
- **后端 API**：FastAPI
- **可视化界面**：Gradio
- **部署平台**：ModelScope 魔搭创空间（Gradio SDK）

## 文件位置
`industrial-defect-detection/` 文件夹

## 文件说明

| 文件 | 说明 |
|------|------|
| `app.py` | Gradio 可视化网页界面（部署入口） |
| `main.py` | FastAPI 推理接口（本地测试用） |
| `train_defect.py` | YOLOv8 训练脚本 |
| `dataset.yaml` | 数据集配置文件 |
| `test_model_inference.py` | 模型预测演示脚本 |
| `best.pt` | 训练好的模型权重（6.3MB） |
| `requirements.txt` | Python 依赖清单 |
| `Dockerfile` | Docker 配置（本地环境限制，待后续完善） |
| `assets/` | 训练曲线、验证效果、预测演示图 |

## 模型性能

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 89.6% |
| mAP@0.5:0.95 | 74.0% |
| Precision | 76.8% |
| Recall | 91.9% |
| 模型大小 | 6.3MB |

## 部署历程

| 尝试 | 平台 | 结果 | 原因 |
|------|------|------|------|
| 1 | Docker Desktop（本地） | ❌ 失败 | Windows 家庭版不支持 |
| 2 | Docker（VMware 虚拟机） | ❌ 失败 | 虚拟机磁盘空间不足 |
| 3 | Streamlit Cloud | ❌ 失败 | Python 3.14 太新，OpenCV 兼容性问题 |
| 4 | **ModelScope 魔搭** | ✅ **成功** | Python 3.11 + Gradio SDK，国内访问快 |

## 关键踩坑与解决

| 问题 | 现象 | 解决方案 |
|------|------|---------|
| OpenCV 报错 | `libGL.so.1: cannot open shared object file` | 改用 `opencv-python-headless` |
| Python 版本冲突 | 本地 3.14 与 OpenCV 不兼容 | 云端选 Python 3.11 镜像 |
| 文件上传丢失 | 拖拽上传只传了部分文件 | 改用"新建文件夹"后重新拖拽 |
| 模型路径 | 绝对路径在云端失效 | 改为相对路径 `best.pt` |

## 本地运行方式

### 1. 训练模型
```bash
python industrial-defect-detection/train_defect.py