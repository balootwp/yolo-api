import os
import io
from typing import Any, Dict, List

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

APP_TITLE = "YOLO Inference API (CPU)"
MODEL_PATH = os.getenv("YOLO_MODEL", "/models/best.pt")
IMG_SIZE = int(os.getenv("YOLO_IMG_SIZE", "640"))
CONF = float(os.getenv("YOLO_CONF", "0.25"))

app = FastAPI(title=APP_TITLE)

model = None

def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    return model

def read_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image content")
    return img

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_path": MODEL_PATH,
        "img_size": IMG_SIZE,
        "conf": CONF,
        "loaded": model is not None,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported image type")

    image_bytes = await file.read()
    try:
        img = read_image_bytes_to_bgr(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        m = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {e}")

    # inference
    results = m.predict(img, imgsz=IMG_SIZE, conf=CONF, verbose=False)
    r0 = results[0]

    detections: List[Dict[str, Any]] = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss  = r0.boxes.cls.cpu().numpy().astype(int)

        names = r0.names if hasattr(r0, "names") else {}

        for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
            detections.append({
                "class_id": int(k),
                "class_name": names.get(int(k), str(int(k))),
                "confidence": float(c),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

    return {
        "ok": True,
        "count": len(detections),
        "detections": detections,
    }
