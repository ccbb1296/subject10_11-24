"""Celery 작업 — 영상 확보 → FastAPI 추론 → Django ORM으로 저장.

상태 흐름: PENDING ─(Worker 시작)→ STARTED ─→ SUCCESS
                                           └─→ FAILURE (+ error_message)
"""

import logging
import time

from datetime import timedelta

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded
from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils import timezone

from .camera_capture import CameraCaptureError, capture_clip
from .models import AnalysisJob, PersonPrediction
from .services import AIServiceError, request_prediction

logger = logging.getLogger(__name__)
Status = AnalysisJob.Status
PREPARING_MESSAGE = "카메라 준비 중"  # 화면: "준비! N초" (촬영 위치로 이동)
CAPTURING_MESSAGE = "카메라 수집 중"  # 화면이 이 메시지를 보고 녹화 카운트다운 시작


STALE_MESSAGE = "분석 Worker 가 중간에 종료되어 작업을 끝내지 못했습니다. 다시 시도하세요."


def fail_stale_jobs(queryset=None) -> int:
    """STARTED 인 채로 너무 오래된 작업 → FAILURE (Worker 강제 종료·재시작으로 _mark_failed 가 못 돈 경우).
    이렇게 해야 '처리 중'에 멈춘 작업을 삭제할 수 있고 화면 폴링도 끝남."""
    cutoff = timezone.now() - timedelta(seconds=settings.STALE_JOB_SECONDS)
    qs = AnalysisJob.objects.all() if queryset is None else queryset
    return qs.filter(status=Status.STARTED, started_at__lt=cutoff).update(
        status=Status.FAILURE, error_message=STALE_MESSAGE, result_message="", completed_at=timezone.now()
    )


def camera_source(camera_url: str) -> str:
    """Worker 가 실제로 읽을 주소 — 중계가 설정돼 있으면 Django 중계(미리보기와 공유), 아니면 카메라 직접."""
    if not settings.CAMERA_RELAY_BASE:
        return camera_url
    index = settings.ALLOWED_CAMERA_URLS.index(camera_url) + 1
    return f"{settings.CAMERA_RELAY_BASE}/analyses/camera/{index}/feed/?token={settings.CAMERA_RELAY_TOKEN}"


def enqueue_analysis(job: AnalysisJob) -> None:
    """DB 저장(commit)이 끝난 뒤 호출 → Redis 대기열에 작업 등록."""
    try:
        result = run_analysis.delay(job.id)
    except Exception as e:  # Redis가 꺼져 있는 등
        logger.exception("작업 등록 실패 job=%s", job.id)
        job.status = Status.FAILURE
        job.error_message = f"작업 대기열(Redis)에 등록하지 못했습니다: {e.__class__.__name__}"
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "completed_at"])
        return
    job.celery_task_id = result.id
    AnalysisJob.objects.filter(id=job.id).update(celery_task_id=result.id)


@shared_task(bind=True)
def run_analysis(self, job_id: int) -> str:
    # ① PENDING 인 작업만 STARTED 로 (같은 작업이 두 번 실행되지 않게 조건부 UPDATE)
    started = AnalysisJob.objects.filter(id=job_id, status=Status.PENDING).update(
        status=Status.STARTED, started_at=timezone.now(), celery_task_id=self.request.id or ""
    )
    if not started:
        logger.warning("job=%s 은 PENDING 이 아니라서 건너뜀", job_id)
        return "skipped"

    job = AnalysisJob.objects.get(id=job_id)
    t0 = time.perf_counter()
    logger.info("job=%s 분석 시작 (%s)", job_id, job.input_type)
    try:
        # ② 영상 확보 → ③ FastAPI 추론
        if job.input_type == AnalysisJob.InputType.UPLOAD:
            result = request_prediction(job.video_file.path, filename=job.video_file.name.rsplit("/", 1)[-1])
        else:
            # IP Webcam: 준비 시간 → 정해진 시간 수집 → media 에 영상으로 남김 → 업로드와 같은 경로로 추론
            # 준비 시간: 버튼을 누르고 촬영 위치로 걸어가는 모습이 녹화에 섞이지 않게 (team17 데모도 3초 준비)
            if settings.CAMERA_PREPARE_SECONDS > 0:
                AnalysisJob.objects.filter(id=job.id).update(result_message=PREPARING_MESSAGE)
                time.sleep(settings.CAMERA_PREPARE_SECONDS)

            def mark_capturing():  # 첫 화면이 들어온 순간 → 화면 카운트다운 시작 신호
                AnalysisJob.objects.filter(id=job.id).update(result_message=CAPTURING_MESSAGE)

            clip = capture_clip(
                camera_source(job.camera_url), settings.CAMERA_CAPTURE_SECONDS, on_start=mark_capturing
            )
            try:
                with clip.open("rb") as f:
                    job.video_file.save(f"camera_job{job.id}.mp4", File(f), save=False)
                job.save(update_fields=["video_file"])
            finally:
                clip.unlink(missing_ok=True)
            logger.info("job=%s 카메라 %s초 수집 완료 → %s", job_id, settings.CAMERA_CAPTURE_SECONDS, job.video_file.name)
            # 녹화는 항상 8초 → 앞뒤 가만히 있던 시간을 잘라 동작 구간만 판단 (학습 영상은 동작만 잘라 둔 것)
            result = request_prediction(
                job.video_file.path, filename=job.video_file.name.rsplit("/", 1)[-1], trim_idle=True
            )
    except (AIServiceError, CameraCaptureError) as e:
        _mark_failed(job, str(e), t0)
        return "failure"
    except SoftTimeLimitExceeded:
        _mark_failed(job, f"분석 제한 시간({settings.CELERY_TASK_SOFT_TIME_LIMIT}초)을 넘겨 중단했습니다.", t0)
        return "failure"
    except Exception as e:  # 예상하지 못한 오류도 FAILURE 로 남김
        logger.exception("job=%s 예상하지 못한 오류", job_id)
        _mark_failed(job, f"예상하지 못한 오류: {e.__class__.__name__}: {e}", t0)
        return "failure"

    # ④ Django ORM으로 결과 저장
    _save_result(job, result, t0)
    logger.info("job=%s 완료: 사람 %d명", job_id, len(result["predictions"]))
    return "success"


def _save_result(job: AnalysisJob, result: dict, t0: float) -> None:
    with transaction.atomic():  # 사람별 결과와 상태를 한 번에 (중간에 실패하면 모두 취소)
        job.predictions.all().delete()
        PersonPrediction.objects.bulk_create(
            [
                PersonPrediction(
                    job=job,
                    track_id=p["track_id"],
                    action_label=p["action_label"],
                    confidence=p["confidence"],
                    is_uncertain=p.get("is_uncertain", False),
                    valid_frames=p.get("valid_frames", 0),
                    window_count=p.get("window_count", 1),
                    top3=p.get("top3", []),
                    rule_note=(p.get("rule") or {}).get("reason", "")[:200],
                )
                for p in result["predictions"]
            ]
        )
        job.status = Status.SUCCESS
        job.error_message = ""
        # 사람 0명이어도 SUCCESS — 이유는 result_message 에 ("영상에서 사람을 찾지 못했습니다.")
        job.result_message = "" if result.get("message") == "ok" else (result.get("message") or "")[:200]
        job.processing_time_sec = round(time.perf_counter() - t0, 2)
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "error_message", "result_message", "processing_time_sec", "completed_at"])


def _mark_failed(job: AnalysisJob, message: str, t0: float) -> None:
    logger.warning("job=%s 실패: %s", job.id, message)
    job.status = Status.FAILURE
    job.error_message = message
    job.result_message = ""  # "카메라 수집 중" 표시 지우기
    job.processing_time_sec = round(time.perf_counter() - t0, 2)
    job.completed_at = timezone.now()
    job.save(update_fields=["status", "error_message", "result_message", "processing_time_sec", "completed_at"])
