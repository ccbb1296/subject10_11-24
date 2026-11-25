"""입력 검증과 JSON 표현.

- AnalysisJobSerializer       : 생성 + 목록·상세 (AI 결과·상태는 읽기 전용)
- AnalysisJobUpdateSerializer : PATCH — 제목·메모만 수정
- AnalysisJobStatusSerializer : 상태 조회 (브라우저 폴링용, 가벼움)
"""

from pathlib import Path

from django.conf import settings
from rest_framework import serializers

from .models import AnalysisJob, PersonPrediction


class PersonPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PersonPrediction
        fields = ["track_id", "action_label", "confidence", "is_uncertain", "top3", "rule_note", "valid_frames", "window_count", "predicted_at"]
        read_only_fields = fields  # AI 결과는 누구도 API로 수정 불가


class AnalysisJobSerializer(serializers.ModelSerializer):
    user = serializers.CharField(source="user.username", read_only=True)
    predictions = PersonPredictionSerializer(many=True, read_only=True)
    is_finished = serializers.BooleanField(read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            "id", "user", "title", "memo",
            "input_type", "video_file", "camera_url", "expected_action",
            "status", "is_finished", "celery_task_id", "error_message", "result_message",
            "processing_time_sec", "created_at", "started_at", "completed_at",
            "predictions",
        ]
        # 서버(Worker)만 정하는 값 — 사용자가 보내도 무시됨
        read_only_fields = [
            "status", "celery_task_id", "error_message", "result_message",
            "processing_time_sec", "created_at", "started_at", "completed_at",
        ]

    def validate_video_file(self, file):
        if not file:
            return file
        suffix = Path(file.name).suffix.lower()
        if suffix not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise serializers.ValidationError(
                f"영상 파일만 올릴 수 있습니다 ({', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}). 받은 형식: '{suffix or '없음'}'"
            )
        if file.size == 0:
            raise serializers.ValidationError("빈 파일입니다.")
        if file.size > settings.MAX_UPLOAD_MB * 1024 * 1024:
            raise serializers.ValidationError(f"파일이 너무 큽니다 (최대 {settings.MAX_UPLOAD_MB}MB).")
        return file

    def validate_camera_url(self, url):
        # 임의 주소 호출 금지 → .env 허용 목록에 있는 카메라만
        if url and url not in settings.ALLOWED_CAMERA_URLS:
            raise serializers.ValidationError("허용된 카메라가 아닙니다. 목록에서 선택하세요.")
        return url

    def validate(self, attrs):
        input_type = attrs.get("input_type")
        video, camera = attrs.get("video_file"), attrs.get("camera_url")
        if input_type == AnalysisJob.InputType.UPLOAD:
            if not video:
                raise serializers.ValidationError({"video_file": "영상 업로드 방식은 영상 파일이 필요합니다."})
            if camera:
                raise serializers.ValidationError({"camera_url": "영상 업로드 방식에는 카메라를 지정하지 않습니다."})
        elif input_type == AnalysisJob.InputType.CAMERA:
            if not camera:
                raise serializers.ValidationError({"camera_url": "IP Webcam 방식은 카메라 선택이 필요합니다."})
            if video:
                raise serializers.ValidationError({"video_file": "IP Webcam 방식에는 영상 파일을 올리지 않습니다."})
        return attrs


class AnalysisJobUpdateSerializer(serializers.ModelSerializer):
    """PATCH: 제목·메모만. 그 밖의 필드를 보내면 400으로 알려줌 (조용히 무시하지 않음)."""

    EDITABLE = {"title", "memo"}

    class Meta:
        model = AnalysisJob
        fields = ["title", "memo"]

    def validate(self, attrs):
        extra = set(self.initial_data.keys()) - self.EDITABLE
        if extra:
            raise serializers.ValidationError(
                {"detail": f"제목(title)과 메모(memo)만 수정할 수 있습니다. 수정 불가: {sorted(extra)}"}
            )
        return attrs


class AnalysisJobStatusSerializer(serializers.ModelSerializer):
    is_finished = serializers.BooleanField(read_only=True)
    prediction_count = serializers.IntegerField(source="predictions.count", read_only=True)

    class Meta:
        model = AnalysisJob
        fields = ["id", "status", "is_finished", "error_message", "result_message", "prediction_count", "completed_at"]
        read_only_fields = fields
