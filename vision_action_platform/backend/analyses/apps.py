import logging
import re

from django.apps import AppConfig

TOKEN_IN_URL = re.compile(r"([?&]token=)[^&\s\"]+")


class RedactTokenFilter(logging.Filter):
    """요청 로그(runserver 'django.server')에서 카메라 중계 토큰 가리기.
    Worker 는 /analyses/camera/N/feed/?token=... 으로 녹화하므로, 그대로 두면 docker compose logs 에 토큰이 남음."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "token=" in message:
            record.msg = TOKEN_IN_URL.sub(r"\1***", message)
            record.args = ()
        return True


class AnalysesConfig(AppConfig):
    name = 'analyses'

    def ready(self):
        for name in ("django.server", "django.request"):
            logging.getLogger(name).addFilter(RedactTokenFilter())
