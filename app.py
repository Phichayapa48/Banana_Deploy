import os
import cv2
import numpy as np
import gc

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

# =========================================================
# INITIALIZE APP
# =========================================================
app = FastAPI(title="Banana Expert AI Server")

# ✅ ตั้งค่า CORS ให้รองรับการเรียกจาก Vercel และ Local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main-banana1.vercel.app",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_REAL = None

# Mapping ID จาก YOLO เป็น Key ที่จะไปหาต่อใน Supabase
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

def load_model():
    global MODEL_REAL
    if MODEL_REAL is not None:
        return

    # ค้นหาไฟล์โมเดลในโฟลเดอร์ model
    model_files = ["best_modelv8sbg.pt", "best_modelv8nbg.pt"]
    found_path = None
    
    for f in model_files:
        p = os.path.join(MODEL_DIR, f)
        if os.path.exists(p):
            found_path = p
            break

    if not found_path:
        print("❌ ไม่พบไฟล์โมเดลใน /model")
        return

    MODEL_REAL = YOLO(found_path)
    print(f"✅ โหลดโมเดลสำเร็จ: {found_path}")

# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {"status": "online", "message": "Banana Expert AI is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# 🔥 จุดแก้สำคัญ: เปลี่ยนจาก bytes เป็น UploadFile เพื่อรองรับ Multipart Form Data
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        # โหลดโมเดลแบบ Lazy Load (ช่วยประหยัด RAM ตอน Start server)
        load_model()
        if MODEL_REAL is None:
            return {"success": False, "reason": "model_not_found"}

        # 1. อ่านไฟล์รูปภาพ
        contents = await file.read()
        
        # 2. แปลงเป็น OpenCV Format
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 3. ส่งให้ YOLO Predict
        results = MODEL_REAL.predict(
            source=img,
            conf=0.25,      # ปรับค่าความเชื่อมั่นขั้นต่ำตามต้องการ
            imgsz=640,
            verbose=False
        )[0]

        # 4. ตรวจสอบผลลัพธ์
        if results.boxes is None or len(results.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        # 5. เลือกผลลัพธ์ที่มีค่าความเชื่อมั่นสูงสุด
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(np.argmax(confs))

        # 6. คืนค่าเป็น JSON ให้ Frontend
        return {
            "success": True,
            "banana_key": CLASS_KEYS.get(clses[best_idx], "unknown"),
            "confidence": float(confs[best_idx])
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"success": False, "reason": str(e)}
    
    finally:
        # สำคัญมาก: ปิดไฟล์และเคลียร์ขยะใน RAM
        await file.close()
        gc.collect()

# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    # Render จะส่งพอร์ตมาให้ผ่าน Environment Variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
