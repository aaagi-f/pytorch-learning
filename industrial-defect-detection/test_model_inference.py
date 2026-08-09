from ultralytics import YOLO
from PIL import Image

# 1. 加载你自己训好的模型
model = YOLO('runs/detect/defect_detection/weights/best.pt')

# 2. 预测一张图（用验证集里的图，或者你自己找一张螺丝图）
# 建议用验证集里的 defect_001.jpg 或 normal_001.jpg
results = model('dataset/images/val/defect_001.jpg')

# 3. 保存预测结果（带框和标签的图）
results[0].save(filename='test_prediction.jpg')

# 4. 打印检测信息
for box in results[0].boxes:
    class_name = results[0].names[int(box.cls)]
    confidence = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"检测到: {class_name}, 置信度: {confidence:.4f}")
    print(f"位置: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

print("预测结果已保存到 test_prediction.jpg")