from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render


def signup(request):
    """회원가입 — Django 기본 User + UserCreationForm (아이디 중복·비밀번호 규칙 자동 검사)."""
    if request.user.is_authenticated:
        return redirect("/analyses/")

    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # 가입 후 바로 로그인
            return redirect("/analyses/")
    else:
        form = UserCreationForm()
    return render(request, "accounts/signup.html", {"form": form})
