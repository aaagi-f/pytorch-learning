from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 训练
model.train(
    data='dataset.yaml',      # 配置文件路径
    epochs=50,               # 训练50轮
    imgsz=640,               # 图片尺寸
    batch=4,                 # 批次大小（显存小设4）
    name='defect_detection'  # 结果保存名
)