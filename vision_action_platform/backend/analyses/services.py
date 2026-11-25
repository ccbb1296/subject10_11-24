"""FastAPI AI 서버 호출 — 영상을 HTTP(multipart)로 보내고 응답·타임아웃·오류를 정리합니다.

Celery Worker(tasks.py)만 사용합니다. Django 요청 함수에서 직접 부르지 않습니다.
"""

from pathlib import Path

import requests
from django.conf import settings


class AIServiceError(Exception):
    """AI 분석을 끝내지 못한 이유 (사용자에게 그대로 보여줄 문장)."""


def request_prediction(video_path: Path | str, filename: str | None = None, trim_idle: bool = False) -> dict:
    """trim_idle=True: IP Webcam 처럼 고정 길이로 녹화한 영상 → FastAPI 가 사람마다 움직인 구간만 판단."""
    url = settings.AI_SERVICE_URL.rstrip("/") + "/predict/video" + ("?trim_idle=true" if trim_idle else "")
    video_path = Path(video_path)
    filename = filename or video_path.name
    timeout = settings.AI_REQUEST_TIMEOUT

    try:
        with video_path.open("rb") as f:
            # (연결 대기 5초, 응답 대기 AI_REQUEST_TIMEOUT초)
            response = requests.post(url, files={"file": (filename, f, "video/mp4")}, timeout=(5, timeout))
    except FileNotFoundError as e:
        raise AIServiceError("분석할 영상 파일을 찾을 수 없습니다.") from e
    except requests.ConnectionError as e:
        raise AIServiceError(f"AI 서버(FastAPI)에 연결할 수 없습니다: {settings.AI_SERVICE_URL}") from e
    except requests.Timeout as e:
        raise AIServiceError(f"AI 서버 응답 시간이 초과되었습니다 ({timeout}초).") from e

    if 400 <= response.status_code < 500:
        # FastAPI가 알려준 이유 (영상 손상·너무 짧음·형식 오류 등)
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text[:200]
        raise AIServiceError(f"영상을 분석할 수 없습니다: {detail}")
    if response.status_code >= 500:
        raise AIServiceError(f"AI 서버 내부 오류가 발생했습니다 (HTTP {response.status_code}).")

    data = response.json()
    if not isinstance(data.get("predictions"), list):
        raise AIServiceError("AI 서버 응답 형식이 올바르지 않습니다.")
    return data
