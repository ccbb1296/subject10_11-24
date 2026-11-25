"""DRF 분석 작업 API

POST   /api/analyses/              작업 생성 (PENDING) → Celery 등록 → 작업 ID 즉시 반환
GET    /api/analyses/              내 작업 목록
GET    /api/analyses/{id}/         상세 + 사람별 결과
PATCH  /api/analyses/{id}/         제목·메모만 수정
DELETE /api/analyses/{id}/         삭제 (처리 중이면 409)
GET    /api/analyses/{id}/status/  상태만 (폴링용)
"""

from django.db import transaction
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import AnalysisJob
from .serializers import (
    AnalysisJobSerializer,
    AnalysisJobStatusSerializer,
    AnalysisJobUpdateSerializer,
)
from .tasks import enqueue_analysis, fail_stale_jobs


class AnalysisJobViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated]
    http_method_names = ["get", "post", "patch", "delete", "head", "options"]  # PUT(전체 덮어쓰기) 막음

    def get_queryset(self):
        # 본인 작업만 → 남의 작업 ID로 접근하면 404 (존재 여부도 알려주지 않음)
        mine = AnalysisJob.objects.filter(user=self.request.user)
        fail_stale_jobs(mine)  # Worker 가 죽어 STARTED 로 멈춘 작업 정리 (폴링·삭제가 끝나도록)
        return (
            mine
            .select_related("user")
            .prefetch_related("predictions")
        )

    def get_serializer_class(self):
        if self.action == "partial_update":
            return AnalysisJobUpdateSerializer
        if self.action == "status":
            return AnalysisJobStatusSerializer
        return AnalysisJobSerializer

    def perform_create(self, serializer):
        # 요청 사용자를 서버가 지정 (다른 사람 이름으로 작업 생성 불가)
        job = serializer.save(user=self.request.user)
        # DB 저장이 확정(commit)된 뒤에 Celery 등록 → Worker가 아직 없는 작업을 찾는 일 방지
        # 분석은 기다리지 않고 바로 응답(작업 ID) 반환
        transaction.on_commit(lambda: enqueue_analysis(job))

    def partial_update(self, request, *args, **kwargs):
        super().partial_update(request, *args, **kwargs)
        # 수정 후에는 전체 작업 정보를 돌려줌
        return Response(AnalysisJobSerializer(self.get_object(), context={"request": request}).data)

    def destroy(self, request, *args, **kwargs):
        job = self.get_object()
        if job.status in (AnalysisJob.Status.PENDING, AnalysisJob.Status.STARTED):
            return Response(
                {"detail": f"분석이 진행 중({job.get_status_display()})이라 삭제할 수 없습니다. 끝난 뒤 삭제하세요."},
                status=status.HTTP_409_CONFLICT,
            )
        if job.video_file:
            job.video_file.delete(save=False)  # media 폴더의 영상도 삭제
        job.delete()  # PersonPrediction 은 CASCADE 로 함께 삭제
        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        return Response(self.get_serializer(self.get_object()).data)
