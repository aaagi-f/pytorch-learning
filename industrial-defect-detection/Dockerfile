FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY runs/detect/defect_detection/weights/best.pt runs/detect/defect_detection/weights/best.pt

EXPOSE 7860

CMD ["python", "app.py"]