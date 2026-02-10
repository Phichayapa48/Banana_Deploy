import os
import cv2
import numpy as np
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import uvicorn
import pkg_resources

app = FastAPI(title="Banana Expert AI Server")

@app.get("/debug/deps")
def debug_deps():
    return {
        "python-multipart": pkg_resources.get_distribution("python-multipart").version
    }
    
# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://main-banana1.vercel.app",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# ROOT + HEAD (🔥 สำคัญมากสำหรับ Render)
# =========================================================
@app.get("/")
def root():
    return {"status": "ok", "service": "Banana Expert AI"}

@app.head("/")
def head_root():
    return None

@app.get("/health")
def health():
    return {"alive": True}

@app.head("/health")
def head_health():
    return None

# =========================================================
# LOAD MODEL
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_REAL = None
MODEL_PATH = None

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
    global MODEL_REAL, MODEL_PATH
    if MODEL_REAL is not None:
        return

    try:
        bg = os.path.join(MODEL_DIR, "best_modelv8sbg.pt")
        nbg = os.path.join(MODEL_DIR, "best_modelv8nbg.pt")
        MODEL_PATH = bg if os.path.exists(bg) else nbg
        MODEL_REAL = YOLO(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        MODEL_REAL = None

# =========================================================
# DETECT + HEAD
# =========================================================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    load_model()

    if MODEL_REAL is None:
        return {"success": False, "reason": "model_not_loaded"}

    try:
        img_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return {"success": False, "reason": "invalid_image"}

        results = MODEL_REAL.predict(
            source=img,
            conf=0.2,
            imgsz=640,
            verbose=False
        )[0]

        if not results.boxes or len(results.boxes) == 0:
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
        gc.collect()

@app.head("/detect")
def head_detect():
    return None

# =========================================================
# START
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
