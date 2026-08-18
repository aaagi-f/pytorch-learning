import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ========== 修改：模型路径改成相对路径 ==========
MODEL_PATH = 'best.pt'  # 模型文件放在app.py同目录

# 加载模型
print("正在加载模型...")
model = YOLO(MODEL_PATH)
print("模型加载完成！")

def detect_defect(image):
    """
    接收一张图片，返回带检测框的结果图
    """
    if image is None:
        return None, "请先上传图片"
    
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
demo = gr.Interface(
    fn=detect_defect,
    inputs=gr.Image(type="pil", label="上传产品图片"),
    outputs=[
        gr.Image(label="检测结果"),
        gr.Textbox(label="检测信息")
    ],
    title="工业缺陷检测系统",
    description="上传螺丝图片，自动检测缺陷（划痕、凹陷、色差等）",
    examples=[]  # 部署时去掉本地路径示例
)

if __name__ == "__main__":
    demo.launch()