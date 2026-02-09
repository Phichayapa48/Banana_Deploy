import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn

app = FastAPI(title="Banana Expert AI Server")

# ✅ 1. CORS Setup - ให้ Frontend (React) เรียกใช้งานได้ไม่มีปัญหา
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ✅ 2. LOAD MODELS (Optimized)
# -------------------------
print("🚀 Loading Banana Expert Models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# โหลด Model หลัก (สลับไปใช้ v8n อัตโนมัติถ้า v8s โหลดไม่สำเร็จเพื่อความเร็ว)
try:
    # พยายามโหลดตัว Small ก่อน
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8sbg.pt"))
    print("✅ MODEL_REAL: YOLOv8s loaded (Small)")
except Exception as e:
    print(f"⚠️ Switching to Fallback (Nano): {e}")
    # ถ้าเครื่องช้ามาก แนะนำให้ใช้ตัว Nano (v8n) จะซิ่งกว่าเยอะครับ
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))

# -------------------------
# ✅ 3. CONFIGURATION
# -------------------------
CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchanggud",
    8: "ngachang", 9: "huamao",
}

def preprocess_image(file: UploadFile):
    """ฟังก์ชันอ่านภาพและบีบอัดขนาดเพื่อความเร็วในการประมวลผล"""
    try:
        img_bytes = file.file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # ⚡️ หัวใจสำคัญ: Resize ให้เหลือ 640x640 ทันทีที่รับมา
            # ช่วยลดภาระ CPU ในการคำนวณลงได้มหาศาล
            img = cv2.resize(img, (640, 640))
        return img
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

# -------------------------
# ✅ 4. API ROUTES
# -------------------------

@app.get("/")
async def root():
    return {"status": "online", "message": "Banana Expert AI is ready to work!"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    try:
        # 1. รับรูปและ Resize (ใช้ฟังก์ชันที่เราสร้างไว้)
        img = preprocess_image(image)
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 2. เริ่มการทำนาย (Inference)
        # ⚡️ การตั้งค่าให้เร็ว:
        # - augment=False: ปิดการคำนวณซ้ำหลายมุม (ประหยัดเวลา 3-4 วินาที)
        # - verbose=False: ไม่ต้อง print log ยาวๆ ออกจอ
        # - conf=0.10: ตั้งค่าความเชื่อมั่นขั้นต่ำไว้ต่ำหน่อย เพื่อให้เห็นกล้วยในหลายๆ สภาพ
        results = MODEL_REAL(img, conf=0.10, iou=0.45, augment=False, verbose=False)[0]

        # 3. ตรวจสอบว่าเจออะไรไหม
        if not results.boxes or len(results.boxes) == 0:
            return {
                "success": False, 
                "reason": "no_banana_detected"
            }

        # 4. ดึงผลลัพธ์ตัวที่คะแนนสูงที่สุด (Confidence สูงสุด)
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(confs.argmax())
        
        final_conf = float(confs[best_idx])
        class_id = int(clses[best_idx])
        banana_key = CLASS_KEYS.get(class_id, "unknown")

        # 5. ส่งผลกลับไปที่หน้าจอ (Frontend)
        return {
            "success": True,
            "banana_key": banana_key,
            "confidence": round(final_conf, 3),
            "debug_info": {
                "boxes_detected": len(results.boxes),
                "model_used": "YOLOv8"
            }
        }

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return {"success": False, "reason": "server_error", "detail": str(e)}

# -------------------------
# ✅ 5. RUN SERVER
# -------------------------
if __name__ == "__main__":
    # รันพอร์ต 8000 เป็นค่ามาตรฐาน
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
