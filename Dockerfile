# 1. ใช้ Slim image น่ะดีแล้ว
FROM python:3.11-slim

# 2. ติดตั้ง lib ที่จำเป็น (ใช้ CACHE ให้คุ้ม)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. จุดสำคัญ! Copy แค่ requirements แล้ว Install ก่อน
# ถ้า requirements ไม่เปลี่ยน Step นี้จะ CACHED ตลอดกาล
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. ค่อย Copy โค้ดที่เหลือตามมา
COPY . .

# 5. รัน
CMD ["python", "app.py"]
