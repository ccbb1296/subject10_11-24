"""URL 지도

/admin/                Django Admin (관리자)
/accounts/...          회원가입 · 로그인 · 로그아웃
/api/analyses/...      DRF 분석 작업 API
/analyses/             사용자 분석 화면
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

urlpatterns = [
    path("", RedirectView.as_view(url="/analyses/", permanent=False)),
    path("admin/", admin.site.urls),
    path("accounts/", include("accounts.urls")),
    path("api/", include("analyses.api_urls")),  # DRF API (JSON) — analyses/api_urls.py
    path("", include("analyses.urls")),  # 화면 (HTML) — analyses/urls.py
]

# 개발 중에만 업로드 영상(media) 파일 제공
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
