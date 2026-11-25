"""분석 업무 테이블 2개.

auth_user (Django 기본)  1 ── N  AnalysisJob  1 ── N  PersonPrediction
"""

from django.conf import settings
from django.db import models

# 팀(team17) 모델이 아는 동작 17개 — ai_service/models/team17/action_classes.csv 와 같은 값·순서(A001~A017)
ACTION_CHOICES = [
    ("kiosk_tap", "키오스크 두드리기"),
    ("throw_one_hand", "한손 물건 던지기"),
    ("front_kick", "앞차기"),
    ("grab_shake", "잡고 흔들기"),
    ("push", "밀치기"),
    ("swing", "휘두르기"),
    ("sit_floor", "바닥에 앉기"),
    ("sit_on_object", "위에 앉기"),
    ("lie_forward", "앞으로 눕기"),
    ("lie_backward", "뒤로 눕기"),
    ("body_slam", "몸통 박치기"),
    ("crouch_take_item", "쪼그려 물건 빼내기"),
    ("shotgun", "샷건"),
    ("vomit", "구토"),
    ("fall_sideways", "옆으로 쓰러지기"),
    ("fall_backward", "뒤로 넘어지기"),
    ("fall_forward", "앞으로 넘어지기"),
]


class AnalysisJob(models.Model):
    """사용자가 요청한 분석 작업 1건 (영상 업로드 또는 IP Webcam 수집)."""

    class InputType(models.TextChoices):
        UPLOAD = "upload", "영상 업로드"
        CAMERA = "camera", "IP Webcam"

    # 상태 이름은 Celery 표기로 통일 (2단계 설계)
    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        STARTED = "STARTED", "처리 중"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    # ── 누가 · 무엇을 ────────────────────────────
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,  # 회원 탈퇴 시 그 회원의 작업도 삭제
        related_name="analysis_jobs",
    )
    title = models.CharField("제목", max_length=100, blank=True)
    memo = models.TextField("메모", blank=True)

    # ── 입력 ─────────────────────────────────────
    input_type = models.CharField("입력 방식", max_length=10, choices=InputType.choices)
    # 영상 파일 자체는 media 폴더에, DB에는 경로만 저장
    video_file = models.FileField("영상 파일", upload_to="uploads/%Y/%m/%d/", blank=True)
    camera_url = models.CharField("카메라 주소", max_length=200, blank=True)
    # 사용자가 "이 동작을 했다"고 고른 값 (정답 검증된 Label 아님)
    expected_action = models.CharField(
        "수행하려는 동작", max_length=30, choices=ACTION_CHOICES, blank=True
    )

    # ── 처리 상태 (서버만 변경) ──────────────────
    status = models.CharField(
        "상태", max_length=10, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    celery_task_id = models.CharField("Celery 작업 ID", max_length=50, blank=True)
    error_message = models.TextField("오류 메시지", blank=True)
    result_message = models.CharField("결과 메시지", max_length=200, blank=True)  # 예: 사람 없음
    processing_time_sec = models.FloatField("처리 시간(초)", null=True, blank=True)

    # ── 시간 ─────────────────────────────────────
    created_at = models.DateTimeField("생성 시각", auto_now_add=True)
    started_at = models.DateTimeField("시작 시각", null=True, blank=True)
    completed_at = models.DateTimeField("완료 시각", null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "분석 작업"
        verbose_name_plural = "분석 작업"

    def __str__(self) -> str:
        return f"#{self.pk} {self.get_input_type_display()} {self.status} ({self.user})"

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.SUCCESS, self.Status.FAILURE)


class PersonPrediction(models.Model):
    """작업 1건 안에서 추적한 사람 1명의 예측 결과."""

    job = models.ForeignKey(
        AnalysisJob,
        on_delete=models.CASCADE,  # 작업 삭제 시 결과도 같이 삭제
        related_name="predictions",
    )
    # 한 작업 안에서만 의미 있는 추적 번호 (영구 사람 ID 아님)
    track_id = models.PositiveIntegerField("Track ID")
    action_label = models.CharField("예측 동작", max_length=30, choices=ACTION_CHOICES)
    confidence = models.FloatField("Confidence")
    is_uncertain = models.BooleanField("판단 불확실", default=False)
    valid_frames = models.PositiveIntegerField("유효 프레임 수", default=0)
    window_count = models.PositiveIntegerField("처리한 Window 수", default=1)
    # 확률 상위 3개 [{"action_label", "action_name_ko", "confidence"}] — 17개 중 헷갈린 후보 확인용
    top3 = models.JSONField("상위 3개 후보", default=list, blank=True)
    # 모델 결과 뒤 보정 규칙이 쓰였으면 그 이유 (예: 밀치기↔잡고 흔들기를 팔꿈치 접은 횟수로 결정)
    rule_note = models.CharField("보정 규칙", max_length=200, blank=True)
    predicted_at = models.DateTimeField("예측 시각", auto_now_add=True)

    class Meta:
        ordering = ["job", "track_id"]
        verbose_name = "사람별 예측"
        verbose_name_plural = "사람별 예측"
        constraints = [
            # 같은 작업 안에서 같은 Track ID 결과가 두 번 저장되지 않게
            models.UniqueConstraint(fields=["job", "track_id"], name="uniq_job_track"),
        ]

    def __str__(self) -> str:
        return f"Job#{self.job_id} track {self.track_id}: {self.action_label} ({self.confidence:.2f})"
