import os
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# ✅ 1. CORS Setup - เชื่อมกับ Frontend (React) ได้ทุกที่
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ✅ 2. LOAD MODELS (Memory Optimized)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# โหลดครั้งเดียวที่ Global Scope เพื่อประหยัด CPU
print("🚀 Loading Banana Expert Models...")
try:
    # พยายามโหลดตัว S (Small) ก่อน
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing: {MODEL_PATH}")
    MODEL_REAL = YOLO(MODEL_PATH)
    print(f"✅ Loaded: YOLOv8s")
except Exception as e:
    print(f"⚠️ Switching to Nano (Fallback): {e}")
    # ถ้าไม่มีตัว S หรือ RAM ไม่พอ ให้ไปใช้ตัว N (Nano) ซึ่งรันบน Render ได้ชัวร์กว่า
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))

# -------------------------
# ✅ 3. CONFIGURATION & MAPPING
# -------------------------
# ตรวจสอบชื่อ Key ให้ตรงกับ Slug ใน Database (Supabase)
CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchangkut",
    8: "ngachang", 9: "huamao",
}

# -------------------------
# ✅ 4. API ROUTES
# -------------------------

@app.get("/")
async def root():
    return {"status": "online", "message": "Banana Expert AI is ready!"}

@app.post("/detect")
@app.post("/detect/") 
async def detect(file: UploadFile = File(...)):
    try:
        # 1. อ่านไฟล์ภาพ
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 2. ปรับขนาดภาพก่อนส่งเข้า AI (ช่วยลดการใช้ RAM และประมวลผลไวขึ้น)
        # YOLOv8 มักจะใช้ 640x640 เป็นมาตรฐาน
        img_resized = cv2.resize(img, (640, 640))

        # 3. เริ่มการทำนาย (Inference)
        # ใช้เครื่องมือจัดการ Thread ของ AI เพื่อไม่ให้ Server ค้าง
        results = MODEL_REAL.predict(
            source=img_resized, 
            conf=0.15,  # ค่าความมั่นใจขั้นต่ำ
            iou=0.45,   # ป้องกันกล่องซ้อนกัน
            imgsz=640, 
            augment=False, 
            verbose=False # ปิด log เพื่อให้เร็วขึ้น
        )[0]

        # 4. ตรวจสอบว่าเจอกล้วยไหม
        if not hasattr(results, 'boxes') or len(results.boxes) == 0:
            return {
                "success": False, 
                "reason": "no_banana_detected",
                "message": "AI หาไม่เจอกรุณาถ่ายให้ชัดเจนขึ้น"
            }

        # 5. ดึงผลลัพธ์ตัวที่มั่นใจสูงสุด (Best Confidence)
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        
        best_idx = int(np.argmax(confs))
        final_conf = float(confs[best_idx])
        class_id = int(clses[best_idx])
        
        # ดึง Slug จาก Mapping
        banana_slug = CLASS_KEYS.get(class_id, "unknown")

        # 6. ส่งข้อมูลกลับไปหา React Frontend
        return {
            "success": True,
            "banana_key": banana_slug,      # ใช้สำหรับจับคู่ slug ใน DB
            "class_name": banana_slug,     # เผื่อ Frontend เรียกตัวแปรนี้
            "confidence": round(final_conf, 3),
            "debug": {
                "count": len(results.boxes),
                "model": "YOLOv8-optimized",
                "original_filename": file.filename
            }
        }

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return {
            "success": False, 
            "reason": "server_error", 
            "detail": str(e)
        }
    finally:
        # ล้าง Buffer ไฟล์ที่อ่านมาเพื่อคืน Memory
        await file.close()

# -------------------------
# ✅ 5. RUN SERVER (Production optimized)
# -------------------------
if __name__ == "__main__":
    # ดึงพอร์ตจาก Environment สำหรับรันบน Render/Heroku
    port = int(os.environ.get("PORT", 8000))
    # ไม่ใช้ reload=True บน Production เพื่อประหยัดทรัพยากร
    uvicorn.run(app, host="0.0.0.0", port=port)s
