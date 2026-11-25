"""DRF API 주소 (JSON) — config/urls.py 에서 path("api/", include("analyses.api_urls"))

router 가 ViewSet 함수에 맞춰 주소를 자동으로 만듦:
POST   /api/analyses/              create          분석 요청 (작업 ID 바로 반환)
GET    /api/analyses/              list            내 작업 목록
GET    /api/analyses/{id}/         retrieve        상세 + 사람별 결과
PATCH  /api/analyses/{id}/         partial_update  제목·메모만 수정
DELETE /api/analyses/{id}/         destroy         삭제
GET    /api/analyses/{id}/status/  status          상태만 (JS 폴링용)
"""

from rest_framework.routers import DefaultRouter

from .views import AnalysisJobViewSet

app_name = "analyses_api"

router = DefaultRouter()
router.register("analyses", AnalysisJobViewSet, basename="analysis")

urlpatterns = router.urls
