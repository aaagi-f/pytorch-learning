from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="工业缺陷检测API")

model = YOLO('runs/detect/defect_detection/weights/best.pt')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 读取上传的图片（bytes → PIL Image）
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # 2. 用模型预测
    results = model(image)
    
    # 3. 提取检测结果（参考你 Step 1 的循环）
    defects = []
    for box in results[0].boxes:
        defects.append({
            "type": results[0].names[int(box.cls)],
            "confidence": round(float(box.conf), 4),
            "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
        })
    
    # 4. 返回 JSON
    return JSONResponse(content={
        "filename": file.filename,
        "defect_count": len(defects),
        "defects": defects
    })

@app.get("/")
async def root():
    return {"message": "缺陷检测API正常运行", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)