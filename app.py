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
    MODEL_PATH = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
    MODEL_REAL = YOLO(MODEL_PATH)
    print(f"✅ MODEL_REAL: YOLOv8s loaded")
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

# ปรับเป็น async เพื่อให้รองรับการอ่านไฟล์บน Cloud (Render) ได้แม่นยำขึ้น
async def preprocess_image(file: UploadFile):
    """ฟังก์ชันอ่านภาพและบีบอัดขนาดเพื่อความเร็วในการประมวลผล"""
    try:
        # ✅ 3 บรรทัดสำคัญที่ช่วยให้อ่านไฟล์จาก FormData ได้ครบถ้วน
        img_bytes = await file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            # ⚡️ หัวใจสำคัญ: Resize ให้เหลือ 640x640 ทันทีที่รับมา
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

@app.post("/detect/")
async def detect(image: UploadFile = File(...)):
    try:
        # 1. รับรูปและ Resize (ต้องใส่ await เพราะ preprocess เป็น async)
        img = await preprocess_image(image)
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # 2. เริ่มการทำนาย (Inference)
        # ⚡️ การตั้งค่าให้เร็ว: augment=False, verbose=False
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
        # ต้องใช้ float() ครอบ final_conf เพื่อให้ JSON รองรับ
        return {
            "success": True,
            "banana_key": banana_key,
            "confidence": round(float(final_conf), 3),
            "debug_info": {
                "boxes_detected": len(results.boxes),
                "model_used": "YOLOv8"
            }
        }

    except Exception as e:
        print(f"❌ Server Error: {e}")
        return {"success": False, "reason": "server_error", "detail": str(e)}

# -------------------------
# ✅ 5. RUN SERVER (3 บรรทัดสุดท้ายที่ห้ามหาย!)
# -------------------------
if __name__ == "__main__":
    # รันพอร์ตตามที่ Render กำหนดผ่าน Environment Variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
