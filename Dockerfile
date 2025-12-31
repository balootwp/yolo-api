FROM docker.arvancloud.ir/python:3.11-slim

ARG COOLIFY_URL
ARG COOLIFY_FQDN
ARG COOLIFY_BUILD_SECRETS_HASH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app

ENV YOLO_MODEL=/models/best.pt
ENV YOLO_IMG_SIZE=640
ENV YOLO_CONF=0.25

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
