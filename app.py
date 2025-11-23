import os
import io
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from rembg import remove
import onnxruntime as ort
import numpy as np

# ----------------------------
# ENV
# ----------------------------
MODEL_URL = os.environ.get("https://ypdmdfdwzldsifijajrm.supabase.co/storage/v1/object/sign/models/best_model.onnx?token=eyJraWQiOiJzdG9yYWdlLXVybC1zaWduaW5nLWtleV9mM2JmYjY1Yi1kMjk2LTRjMmQtODI2OS0yZGFiNjhjNzM1MGIiLCJhbGciOiJIUzI1NiJ9.eyJ1cmwiOiJtb2RlbHMvYmVzdF9tb2RlbC5vbm54IiwiaWF0IjoxNzYzNjMzMjgyLCJleHAiOjE3OTUxNjkyODJ9.WM5TnVIhiFg5ejGNEgck-DGwWtrEECg3w84IJq4p7Ik")            # URL ของ Supabase
MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "best_model.onnx")
PORT = int(os.environ.get("PORT", 8000))
MAX_UPLOAD_MB = 5
TARGET_SIZE = 640

os.environ["RMBG_MODEL"] = "u2netp"

session = None

# ----------------------------
# DOWNLOAD MODEL
# ----------------------------
def download_model_if_needed():
    if not MODEL_URL:
        raise ValueError("MODEL_URL not set")

    # ถ้ามีโมเดลแล้วให้ใช้เลย
    if os.path.exists(MODEL_LOCAL_PATH) and os.path.getsize(MODEL_LOCAL_PATH) > 5000:
        print("✅ YOLO model already exists")
        return

    print("⬇️ Downloading YOLO model from Supabase private bucket...")

    supabase_key = os.environ.get("SUPABASE_KEY")
    headers = {}
    if supabase_key:
        headers["apikey"] = supabase_key
        headers["Authorization"] = f"Bearer {supabase_key}"

    with requests.get(MODEL_URL, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(MODEL_LOCAL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print("✅ Download complete")

# ----------------------------
# LOAD MODEL
# ----------------------------
def load_yolo_model():
    global session
    if session is None:
        download_model_if_needed()
        session = ort.InferenceSession(
            MODEL_LOCAL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("🚀 YOLO ONNX model loaded")

# ----------------------------
# UTILS
# ----------------------------
def bytes_to_pil(b):
    return Image.open(io.BytesIO(b))

def resize_to_640(img):
    return img.resize((TARGET_SIZE, TARGET_SIZE))

# ----------------------------
# FASTAPI
# ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Banana Model API running", "status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "File must be an image")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(400, "Max upload size exceeded (5MB)")

    if session is None:
        load_yolo_model()

    img = bytes_to_pil(contents).convert("RGB")
    img = resize_to_640(img)

    try:
        os.environ["RMBG_SESSION_THREADS"] = "1"
        img_no_bg = remove(img)
    except Exception as e:
        print("⚠️ rembg failed:", e)
        img_no_bg = img

    arr = np.array(img_no_bg).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, :, :, :]

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})

    detections = outputs[0].tolist() if len(outputs) > 0 else []

    return {"detections": detections}

if __name__ == "__main__":
    import uvicorn
    print(f"Running on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
