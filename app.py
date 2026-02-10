import os
import cv2
import numpy as np
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# 1. CORS - ตั้งค่าให้รับได้ทุกหัวลูกศร
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], # ระบุให้ชัด
    allow_headers=["*"],
)

# 2. LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

try:
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8nbg.pt")
    MODEL_REAL = YOLO(MODEL_PATH)
    print(f"✅ AI Ready with: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Model load error: {e}")
    MODEL_REAL = None

CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchangkut",
    8: "ngachang", 9: "huamao",
}

# 3. ROUTES - ดักทั้งแบบมี / และไม่มี / เพื่อกัน 405
@app.post("/detect")
@app.post("/detect/") 
async def detect(file: UploadFile = File(...)):
    if MODEL_REAL is None:
        return {"success": False, "reason": "model_not_found"}
    
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image"}

        results = MODEL_REAL.predict(source=img, conf=0.20, imgsz=640, save=False, verbose=False)[0]

        if not results.boxes:
            return {"success": False, "reason": "no_banana_detected"}

        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))
        
        return {
            "success": True,
            "banana_key": CLASS_KEYS.get(clses[best_idx], "unknown"),
            "confidence": float(confs[best_idx])
        }
    except Exception as e:
        return {"success": False, "reason": str(e)}
    finally:
        if 'img' in locals(): del img
        gc.collect()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
