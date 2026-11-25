# FastAPI AI 추론 서버 이미지 (YOLO Pose + 6일차 Random Forest)
FROM python:3.12-slim

# OpenCV(opencv-python)가 쓰는 시스템 라이브러리
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# ① torch 는 CPU 버전 (CUDA 버전은 수 GB) — 버전은 학습 환경과 같은 2.15.0
#    torchvision 은 torch 2.15.0 과 맞는 버전을 pip 가 고름 (빌드 결과 0.29.2+cpu)
# ② 나머지는 requirements.txt 그대로 (scikit-learn 1.9.1 등 모델 로드에 중요한 버전 고정)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0" torchvision \
 && grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

# 코드 · 모델 (models/ 의 joblib·JSON·yolo11n-pose.pt 포함)
COPY . .

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
