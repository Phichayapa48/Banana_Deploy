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
# 1. CORS (ลำดับความสำคัญสูงสุด)
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
    # ถ้าตัว s มีปัญหา ให้ลองโหลดตัว n
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))

# =========================================================
# 3. CLASS MAPPING (ตรงกับ Supabase slug)
# =========================================================
CLASS_KEYS = {
    0: "candyapple",
    1: "namwa",
    2: "namwadam",
    3: "homthong",
    4: "nak",
    5: "thepphanom",
    6: "kai",
    7: "lepchangkut",
    8: "ngachang",
    9: "huamao",
}

# =========================================================
# 4. ROUTES & ERROR HANDLING
# =========================================================

@app.get("/")
async def root():
    return {"status": "online", "message": "Banana Expert AI is ready!"}

# ✅ ดักจับ OPTIONS Request เพื่อป้องกัน 405 (Preflight) จาก Browser
@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ✅ ป้องกัน 405 จากการเผลอเรียก GET
@app.get("/detect")
@app.get("/detect/")
@app.head("/detect")
@app.head("/detect/")
async def detect_guard():
    return {
        "status": "ok",
        "message": "Use POST /detect/ with multipart/form-data"
    }

# ✅ POST ROUTE (รับทั้งแบบมีและไม่มี Slash)
@app.post("/detect")
@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    try:
        # 1. อ่านภาพจาก Buffer
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image"}

        # 2. Resize เพื่อคุม Memory และเพิ่มความเร็ว
        img = cv2.resize(img, (640, 640))

        # 3. ประมวลผลด้วย AI
        results = MODEL_REAL.predict(
            source=img,
            conf=0.15,
            iou=0.45,
            imgsz=640,
            verbose=False
        )[0]

        # กรณีไม่เจอกล้วยในภาพ
        if not results.boxes or len(results.boxes) == 0:
            return {
                "success": False, 
                "reason": "no_banana_detected"
            }

        # 4. ดึงผลลัพธ์ที่ Confidence สูงสุด
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(np.argmax(confs))
        banana_slug = CLASS_KEYS.get(int(clses[best_idx]), "unknown")

        # 🟢 คืนค่าข้อมูลครบถ้วนสำหรับ Frontend
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
        return {
            "success": False,
            "reason": "server_error",
            "detail": str(e)
        }

    finally:
        # ปิดไฟล์เพื่อคืน Memory
        await file.close()

# =========================================================
# 5. RUN SERVER
# =========================================================
if __name__ == "__main__":
    # Render จะส่ง Port มาให้ผ่าน Environment Variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
