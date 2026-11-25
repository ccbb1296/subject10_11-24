"""HTML 화면 — 사용자 분석 화면 · 작업 상세 · 관리자 대시보드.

데이터 변경(생성·수정·삭제)은 모두 DRF API(/api/analyses/)로 하고, 여기서는 화면만 그립니다.
대시보드 숫자는 Django ORM 집계로 만듭니다 (FastAPI는 DB를 모름).
"""

import hmac

from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db.models import Avg, Count, Q
from django.http import Http404, HttpResponseForbidden, JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_POST
from django.shortcuts import get_object_or_404, render

from .camera_relay import BOUNDARY, get_hub, mjpeg_stream
from .models import ACTION_CHOICES, AnalysisJob, PersonPrediction
from .tasks import fail_stale_jobs

ACTION_LABELS = dict(ACTION_CHOICES)


def camera_choices():
    """허용 카메라 목록 → 화면에는 주소 대신 '카메라 1' 처럼 표시 (접속 정보 노출 줄이기)."""
    return [{"url": url, "name": f"카메라 {i}", "index": i} for i, url in enumerate(settings.ALLOWED_CAMERA_URLS, 1)]


# ── IP Webcam 미리보기 / Worker 녹화용 중계 (camera_relay.py) ──────────
def _camera_url(index: int) -> str:
    urls = settings.ALLOWED_CAMERA_URLS
    if not 1 <= index <= len(urls):
        raise Http404("등록되지 않은 카메라입니다.")
    return urls[index - 1]


def _hub(index: int):
    return get_hub(_camera_url(index), settings.CAMERA_ROTATION)


def _mjpeg_response(index: int) -> StreamingHttpResponse:
    response = StreamingHttpResponse(
        mjpeg_stream(_hub(index)), content_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}"
    )
    response["Cache-Control"] = "no-store"
    return response


@login_required
def camera_stream(request, index):
    """브라우저 미리보기 (<img src=…>) — 로그인한 사용자만."""
    return _mjpeg_response(index)


@login_required
def camera_status(request, index):
    """미리보기 아래 '연결됨 / 연결 안 됨' 표시용."""
    return JsonResponse(_hub(index).status())


@login_required
@require_POST
def camera_rotate(request, index):
    """'↻ 화면 돌리기' — 시계 방향 90도씩. 미리보기와 다음 녹화 모두 적용 (서버 재시작 시 .env 값으로)."""
    # 녹화 중에 돌리면 앞뒤 프레임 크기가 달라져 영상이 찌그러짐 → 이 카메라로 분석 중이면 막음 (모든 사용자 공통 설정)
    recording = AnalysisJob.objects.filter(
        input_type=AnalysisJob.InputType.CAMERA, camera_url=_camera_url(index), status=AnalysisJob.Status.STARTED
    ).exists()
    if recording:
        return JsonResponse({"detail": "이 카메라로 녹화·분석 중이라 화면을 돌릴 수 없습니다. 끝난 뒤 다시 누르세요."}, status=409)
    hub = _hub(index)
    hub.set_rotation((hub.rotation + 90) % 360)
    return JsonResponse(hub.status())


def camera_feed(request, index):
    """Celery Worker 녹화용 — 로그인 대신 .env 의 CAMERA_RELAY_TOKEN 으로 확인 (컨테이너 내부 호출)."""
    token = request.GET.get("token", "")
    if not settings.CAMERA_RELAY_TOKEN or not hmac.compare_digest(token, settings.CAMERA_RELAY_TOKEN):
        return HttpResponseForbidden("invalid token")
    return _mjpeg_response(index)


@login_required
def index(request):
    jobs = request.user.analysis_jobs.prefetch_related("predictions")[:10]
    return render(
        request,
        "analyses/index.html",
        {
            "jobs": jobs,
            "actions": ACTION_CHOICES,
            "cameras": camera_choices(),
            "capture_seconds": settings.CAMERA_CAPTURE_SECONDS,
            "prepare_seconds": settings.CAMERA_PREPARE_SECONDS,
            "video_extensions": ", ".join(settings.ALLOWED_VIDEO_EXTENSIONS),
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "page_config": {
                "actionLabels": ACTION_LABELS,
                "uncertainThreshold": settings.UNCERTAIN_CONFIDENCE,
                "captureSeconds": settings.CAMERA_CAPTURE_SECONDS,
                "prepareSeconds": settings.CAMERA_PREPARE_SECONDS,
            },
        },
    )


@login_required
def detail(request, pk):
    # 본인 작업만 (남의 작업 → 404, API와 같은 규칙)
    fail_stale_jobs(AnalysisJob.objects.filter(user=request.user, pk=pk))
    job = get_object_or_404(AnalysisJob.objects.prefetch_related("predictions"), pk=pk, user=request.user)
    return render(request, "analyses/detail.html", {"job": job, "action_labels": ACTION_LABELS})


@staff_member_required
def dashboard(request):
    # ① 6개 동작별 예측 건수 (확실 / 불확실 나눠서) — 0건인 동작도 표시
    by_action = {
        row["action_label"]: row
        for row in PersonPrediction.objects.values("action_label").annotate(
            total=Count("id"), uncertain=Count("id", filter=Q(is_uncertain=True))
        )
    }
    action_chart = {
        "labels": [f"{ACTION_LABELS[a]}\n({a})" for a, _ in ACTION_CHOICES],
        "certain": [by_action.get(a, {}).get("total", 0) - by_action.get(a, {}).get("uncertain", 0) for a, _ in ACTION_CHOICES],
        "uncertain": [by_action.get(a, {}).get("uncertain", 0) for a, _ in ACTION_CHOICES],
    }

    # ② 작업 상태별 건수
    by_status = dict(AnalysisJob.objects.values_list("status").annotate(n=Count("id")))
    status_chart = {
        "labels": [label for _, label in AnalysisJob.Status.choices],
        "keys": [key for key, _ in AnalysisJob.Status.choices],
        "counts": [by_status.get(key, 0) for key, _ in AnalysisJob.Status.choices],
    }

    # (선택) 입력 방식별 건수·평균 처리 시간
    by_input = {
        row["input_type"]: row
        for row in AnalysisJob.objects.values("input_type").annotate(
            n=Count("id"), avg_sec=Avg("processing_time_sec", filter=Q(status=AnalysisJob.Status.SUCCESS))
        )
    }
    input_rows = [
        {"label": label, "count": by_input.get(key, {}).get("n", 0), "avg_sec": by_input.get(key, {}).get("avg_sec")}
        for key, label in AnalysisJob.InputType.choices
    ]

    total_jobs = sum(status_chart["counts"])
    finished = by_status.get("SUCCESS", 0) + by_status.get("FAILURE", 0)
    context = {
        "stats": {
            "users": get_user_model().objects.count(),
            "jobs": total_jobs,
            "predictions": PersonPrediction.objects.count(),
            "success_rate": round(by_status.get("SUCCESS", 0) / finished * 100) if finished else None,
        },
        "input_rows": input_rows,
        "recent_jobs": AnalysisJob.objects.select_related("user").annotate(people=Count("predictions"))[:10],
        "chart_data": {"action": action_chart, "status": status_chart},
    }
    return render(request, "analyses/dashboard.html", context)
