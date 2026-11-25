from django.contrib.auth import views as auth_views
from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    # 로그인·로그아웃은 Django 기본 뷰 사용 (직접 만들지 않음)
    path("login/", auth_views.LoginView.as_view(), name="login"),  # templates/registration/login.html
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),  # POST 요청만 허용
]
