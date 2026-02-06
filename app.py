import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODELS WITH FALLBACK
# -------------------------
print("🚀 Loading Models...")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 1. Filter model (banana / non-banana)
MODEL_FILTER = YOLO(os.path.join(MODEL_DIR, "best_m1_bgv8s.pt"))

# 2. Real-image detector (Main)
try:
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8sbg.pt"))
    print("✅ MODEL_REAL: YOLOv8s loaded as Main")
except Exception as e:
    print(f"⚠️ Cannot load YOLOv8s, switching to YOLOv8n fallback... Error: {e}")
    MODEL_REAL = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))
    print("🚀 MODEL_REAL: YOLOv8n loaded as Fallback")

print("✅ All systems ready")

# -------------------------
# CLASS KEY MAP (ปรับเป็น lowercase เพื่อเชื่อม slug ใน Supabase)
# -------------------------
CLASS_KEYS = {
    0: "candyapple", 1: "namwa", 2: "namwadam", 3: "homthong",
    4: "nak", 5: "thepphanom", 6: "kai", 7: "lepchanggud",
    8: "ngachang", 9: "huamao",
}

# -------------------------
# UTILS
# -------------------------
def read_image(file: UploadFile):
    img_bytes = file.file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# -------------------------
# API
# -------------------------
@app.post("/detect")
async def detect(image: UploadFile = File(...), mode: str = Form("real")):
    try:
        img = read_image(image)
        if img is None:
            return {"success": False, "reason": "invalid_image_format"}

        # ---------- STAGE 1: FILTER ----------
        r1 = MODEL_FILTER(img, conf=0.35, verbose=False)[0]
        if r1.boxes is None or len(r1.boxes) == 0:
            return {"success": False, "reason": "no_banana_detected"}

        # ---------- STAGE 2: REAL DETECTION ----------
        used_fallback = False

        try:
            r2 = MODEL_REAL(img, conf=0.25, verbose=False)[0]
        except Exception as e:
            print("🚨 Runtime error on main model, switching to fallback (v8n)")
            fallback_model = YOLO(os.path.join(MODEL_DIR, "best_modelv8nbg.pt"))
            r2 = fallback_model(img, conf=0.25, verbose=False)[0]
            used_fallback = True

        if r2.boxes is None or len(r2.boxes) == 0:
            return {"success": False, "reason": "banana_like_object"}

        # ---------- POST PROCESS ----------
        confs = r2.boxes.conf.cpu().numpy()
        clses = r2.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(confs.argmax())
        class_id = int(clses[best_idx])
        conf = float(confs[best_idx])

        if conf < 0.40:
            return {"success": False, "reason": "low_confidence"}

        banana_key = CLASS_KEYS.get(class_id)
        if banana_key is None:
            return {"success": False, "reason": "unknown_class_id"}

        return {
            "success": True,
            "banana_key": banana_key,
            "confidence": round(conf, 3),
            "engine": "fallback" if used_fallback else "main"
        }

    except Exception as e:
        print("❌ Server Error:", e)
        return {
            "success": False,
            "reason": "server_error",
            "detail": str(e)
        }

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
