FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_ROOT=/models

# build-essential: compiles insightface's C extension. libgomp1: OpenMP for faiss/onnxruntime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model pack into the image so containers start without a download.
RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='/models', providers=['CPUExecutionProvider'])"

COPY app ./app
COPY scripts ./scripts

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
