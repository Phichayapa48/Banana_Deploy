import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# =========================================================
# 1. CORS - ตัวนี้จัดการ OPTIONS ให้เองอัตโนมัติ ห้ามเขียนซ้อน
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2. LOAD MODEL (Global – โหลดครั้งเดียวตอน Start)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

print("🚀 Loading Banana Expert Models...")
try:
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("YOLOv8s file not found")
    MODEL_REAL = YOLO(MODEL_PATH)
    print("✅ YOLOv8s loaded successfully")
except Exception as e:
    print(f"⚠️ Fallback to Nano: {e}")
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))

# =========================================================
# 3. CLASS MAPPING
# =========================================================
CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchangkut",
    8: "ngachang", 9: "huamao",
}

# =========================================================
# 4. ROUTES
# =========================================================

@app.get("/")
async def root():
    return {"status": "online", "message": "Banana Expert AI is ready!"}

# ดักจับทั้ง /detect และ /detect/ เพื่อป้องกัน 405 Redirect
@app.post("/detect")
@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image"}

        img = cv2.resize(img, (640, 640))

        results = MODEL_REAL.predict(
            source=img,
            conf=0.15,
            iou=0.45,
            imgsz=640,
            verbose=False
        )[0]

        if not results.boxes or len(results.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))
        banana_slug = CLASS_KEYS.get(int(clses[best_idx]), "unknown")

        return {
            "success": True,
            "banana_key": banana_slug,
            "class_name": banana_slug,
            "confidence": round(float(confs[best_idx]), 3),
            "debug": {
                "count": len(results.boxes),
                "model": "YOLOv8-optimized",
                "filename": file.filename
            }
        }
    except Exception as e:
        print(f"❌ Server error: {e}")
        return {"success": False, "reason": "server_error", "detail": str(e)}
    finally:
        await file.close()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
