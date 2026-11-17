FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PAYGUARD_MODEL_PATH=/app/models/payguard-lightgbm.joblib \
    PAYGUARD_FRAUD_THRESHOLD=0.5

WORKDIR /app

# LightGBM's Linux wheel needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY infra/requirements-serving.txt /app/requirements-serving.txt
RUN pip install --no-cache-dir -r /app/requirements-serving.txt

# Copy only inference code. Model and dataset folders are intentionally supplied
# at runtime / excluded from the build context.
COPY src/__init__.py /app/src/__init__.py
COPY src/models/__init__.py /app/src/models/__init__.py
COPY src/models/model_package.py /app/src/models/model_package.py
COPY src/serving/__init__.py /app/src/serving/__init__.py
COPY src/serving/app.py /app/src/serving/app.py

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
