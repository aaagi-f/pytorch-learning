import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 加载模型
model = YOLO('runs/detect/defect_detection/weights/best.pt')

def detect_defect(image):
    """
    接收一张图片，返回带检测框的结果图
    """
    # 预测
    results = model(image)
    
    # 获取带框的结果图
    result_img = results[0].plot()  # 直接生成带框和标签的图
    
    # 提取文字信息
    info = []
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls)]
        conf = float(box.conf)
        info.append(f"{class_name}: {conf:.2f}")
    
    # 如果没有检测到，显示"无缺陷"
    if not info:
        info = ["无缺陷"]
    
    return result_img, "\n".join(info)

# 创建Gradio界面
iface = gr.Interface(
    fn=detect_defect,                    # 处理函数
    inputs=gr.Image(type="pil"),         # 输入：图片
    outputs=[                            # 输出：图片 + 文字
        gr.Image(label="检测结果"),
        gr.Textbox(label="检测信息")
    ],
    title="工业缺陷检测系统",
    description="上传螺丝图片，自动检测缺陷",
    examples=["dataset/images/val/defect_001.jpg"]  # 示例图片
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)