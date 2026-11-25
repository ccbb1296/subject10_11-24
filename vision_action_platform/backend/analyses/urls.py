"""화면 주소 (HTML) — config/urls.py 에서 path("", include("analyses.urls"))
DRF API(JSON) 주소는 api_urls.py 에 따로 있음."""

from django.urls import path

from . import page_views

app_name = "analyses"

urlpatterns = [
    path("analyses/", page_views.index, name="index"),  # 사용자 분석 화면
    path("analyses/<int:pk>/", page_views.detail, name="detail"),  # 작업 상세 (영상·결과·메모)
    path("dashboard/", page_views.dashboard, name="dashboard"),  # 관리자 전용 대시보드
    # IP Webcam 중계: 미리보기(브라우저) · 상태 · Worker 녹화용
    path("analyses/camera/<int:index>/stream/", page_views.camera_stream, name="camera_stream"),
    path("analyses/camera/<int:index>/status/", page_views.camera_status, name="camera_status"),
    path("analyses/camera/<int:index>/feed/", page_views.camera_feed, name="camera_feed"),
    path("analyses/camera/<int:index>/rotate/", page_views.camera_rotate, name="camera_rotate"),
]
