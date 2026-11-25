"""Celery 앱 — Django 설정을 읽고, 각 앱의 tasks.py 를 자동으로 찾습니다.

Worker 실행 (backend 폴더에서):
    celery -A config worker -l info
"""

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
# settings.py 의 CELERY_ 로 시작하는 값을 Celery 설정으로 사용
app.config_from_object("django.conf:settings", namespace="CELERY")
# INSTALLED_APPS 의 tasks.py (analyses/tasks.py) 자동 등록
app.autodiscover_tasks()
