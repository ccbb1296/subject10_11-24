# Django + Celery Worker 공용 이미지 (같은 코드, 실행 명령만 다름 → compose.yaml)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
# 기본 실행: DB 마이그레이션 → Django (Celery 는 compose.yaml 에서 command 로 바꿔 실행)
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py runserver 0.0.0.0:8000"]
