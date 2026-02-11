import os
import cv2
import numpy as np
import gc
import torch
import sys
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="Banana Expert AI Server (3-Model Edition)")

# ตั้งค่า CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# 🚀 LOAD ALL 3 MODELS (Strict Loading)
# -------------------------
print("🚀 Loading 3 Models...")
try:
    # 1. โมเดลกรอง (Stage 1)
    MODEL_FILTER = YOLO(os.path.join(BASE_DIR, "model/best_m1_bgv8n.pt")).to("cpu")
    
    # 2. โมเดลหลัก (Stage 2: Main)
    MODEL_MAIN   = YOLO(os.path.join(BASE_DIR, "model/best_modelv8sbg.pt")).to("cpu")
    
    # 3. โมเดลสำรอง (Stage 3: Backup)
    MODEL_BACKUP = YOLO(os.path.join(BASE_DIR, "model/best_modelv8nbg.pt")).to("cpu")
    
    print("✅ 3 Models Loaded: Filter, Main, and Backup")
except Exception as e:
    # 🛑 ถ้าโมเดลพังตั้งแต่ตอนโหลด ให้สั่งปิดโปรแกรมทันที
    print(f"❌ CRITICAL ERROR: Could not load models: {e}")
    sys.exit(1)

# แผนผังสายพันธุ์กล้วย
CLASS_KEYS = {
    0: "Candyapple", 1: "Namwa", 2: "Namwadam", 3: "Homthong",
    4: "Nak", 5: "Thepphanom", 6: "Kai", 7: "Lepchanggud",
    8: "Ngachang", 9: "Huamao",
}

def read_image(file: UploadFile):
    data = np.frombuffer(file.file.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img = None
    try:
        img = read_image(image)
        if img is None: 
            return {"success": False, "reason": "invalid_image"}

        # -------------------------------------------------------
        # STAGE 1 : FILTER (กรองภาพกล้วย)
        # -------------------------------------------------------
        with torch.no_grad():
            r1 = MODEL_FILTER.predict(
                source=img, conf=0.35, imgsz=416, device="cpu", verbose=False
            )[0]
        
        if r1.boxes is None or len(r1.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        # -------------------------------------------------------
        # STAGE 2 : MAIN DETECTION (พยายามใช้ตัวหลักก่อน)
        # -------------------------------------------------------
        final_result = None
        is_backup_used = False

        try:
            with torch.no_grad():
                r_main = MODEL_MAIN.predict(
                    source=img, conf=0.25, imgsz=512, device="cpu", verbose=False
                )[0]
            
            # เช็คว่าตัวหลักเจอของไหม
            if r_main.boxes is not None and len(r_main.boxes) > 0:
                final_result = r_main
            else:
                raise ValueError("Main model found nothing")

        except Exception as e:
            # -------------------------------------------------------
            # STAGE 3 : BACKUP DETECTION (ถ้าตัวหลักพังหรือหาไม่เจอ)
            # -------------------------------------------------------
            print(f"🔄 Switching to Backup Model due to: {e}")
            is_backup_used = True
            with torch.no_grad():
                final_result = MODEL_BACKUP.predict(
                    source=img, conf=0.20, imgsz=512, device="cpu", verbose=False
                )[0]

        # ตรวจสอบผลลัพธ์สุดท้าย
        if final_result is None or final_result.boxes is None or len(final_result.boxes) == 0:
            return {"success": False, "reason": "all_models_failed"}

        # สรุปข้อมูล
        confs = final_result.boxes.conf.cpu().numpy()
        clses = final_result.boxes.cls.cpu().numpy().astype(int)
        best_idx = int(confs.argmax())
        
        return {
            "success": True,
            "banana_key": CLASS_KEYS.get(int(clses[best_idx]), "unknown"),
            "confidence": round(float(confs[best_idx]), 4),
            "used_backup": is_backup_used
        }

    except Exception as e:
        print("❌ Server Error:", e)
        return {"success": False, "reason": "server_error"}
    finally:
        # เคลียร์ Memory ทุกครั้ง
        if img is not None: 
            del img
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    # รันบนเครื่องตัวเอง (Local)
    uvicorn.run(app, host="0.0.0.0", port=10000)
