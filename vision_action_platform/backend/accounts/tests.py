from django.contrib.auth import get_user_model
from django.test import TestCase

User = get_user_model()
PW = "Vision-test-2026!"


class SignupLoginTests(TestCase):
    def test_signup_creates_user_and_logs_in(self):
        r = self.client.post("/accounts/signup/", {"username": "alice", "password1": PW, "password2": PW})
        self.assertRedirects(r, "/analyses/", fetch_redirect_response=False)
        self.assertTrue(User.objects.filter(username="alice").exists())
        self.assertEqual(self.client.get("/analyses/").status_code, 200)  # 바로 로그인 상태

    def test_duplicate_username_rejected(self):
        User.objects.create_user("alice", password=PW)
        r = self.client.post("/accounts/signup/", {"username": "alice", "password1": PW, "password2": PW})
        self.assertEqual(r.status_code, 200)  # 폼 다시 보여줌
        self.assertEqual(User.objects.filter(username="alice").count(), 1)

    def test_weak_or_mismatched_password_rejected(self):
        self.client.post("/accounts/signup/", {"username": "bob", "password1": "1234", "password2": "1234"})
        self.client.post("/accounts/signup/", {"username": "carol", "password1": PW, "password2": PW + "x"})
        self.assertFalse(User.objects.filter(username__in=["bob", "carol"]).exists())

    def test_login_and_logout(self):
        User.objects.create_user("alice", password=PW)
        r = self.client.post("/accounts/login/", {"username": "alice", "password": PW})
        self.assertRedirects(r, "/analyses/", fetch_redirect_response=False)
        self.client.post("/accounts/logout/")
        r = self.client.get("/analyses/")
        self.assertEqual(r.status_code, 302)
        self.assertIn("/accounts/login/", r["Location"])

    def test_anonymous_redirected_to_login(self):
        r = self.client.get("/analyses/")
        self.assertRedirects(r, "/accounts/login/?next=/analyses/", fetch_redirect_response=False)
