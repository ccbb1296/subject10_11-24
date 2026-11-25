"""분석 API 권한·CRUD 테스트 (과제 T05·T06·T07 일부)."""

import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, override_settings
from rest_framework.test import APITestCase

from .camera_capture import CameraCaptureError, capture_clip
from .models import ACTION_CHOICES, AnalysisJob, PersonPrediction
from .services import AIServiceError
from .tasks import run_analysis

User = get_user_model()
TMP_MEDIA = tempfile.mkdtemp()
CAMERA = "http://192.168.0.10:4747/video"


def video(name="clip.mp4", content=b"fake-video-bytes"):
    return SimpleUploadedFile(name, content, content_type="video/mp4")


@override_settings(MEDIA_ROOT=TMP_MEDIA, ALLOWED_CAMERA_URLS=[CAMERA])
class AnalysisApiTests(APITestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def setUp(self):
        self.a = User.objects.create_user("user_a", password="x")
        self.b = User.objects.create_user("user_b", password="x")
        self.b_job = AnalysisJob.objects.create(user=self.b, input_type="upload", status="SUCCESS", title="B의 작업")
        PersonPrediction.objects.create(job=self.b_job, track_id=1, action_label="swing", confidence=0.7)

    # ── 로그인 ──────────────────────────────────
    def test_anonymous_blocked(self):
        self.assertEqual(self.client.get("/api/analyses/").status_code, 403)
        self.assertEqual(self.client.post("/api/analyses/", {"input_type": "upload"}).status_code, 403)

    # ── Create ──────────────────────────────────
    def test_create_upload_job(self):
        self.client.force_authenticate(self.a)
        r = self.client.post(
            "/api/analyses/",
            {"input_type": "upload", "video_file": video(), "title": "손 흔들기", "expected_action": "swing",
             "status": "SUCCESS", "user": self.b.id},  # status·user 는 보내도 무시되어야 함
            format="multipart",
        )
        self.assertEqual(r.status_code, 201, r.data)
        job = AnalysisJob.objects.get(id=r.data["id"])
        self.assertEqual(job.user, self.a)  # 서버가 요청자로 지정
        self.assertEqual(job.status, "PENDING")  # 상태는 서버가 정함
        self.assertTrue(job.video_file.name.startswith("uploads/"))  # DB에는 경로만

    def test_create_camera_job(self):
        self.client.force_authenticate(self.a)
        r = self.client.post("/api/analyses/", {"input_type": "camera", "camera_url": CAMERA}, format="json")
        self.assertEqual(r.status_code, 201, r.data)

    def test_create_invalid_inputs(self):
        self.client.force_authenticate(self.a)
        cases = {
            "업로드인데 파일 없음": {"input_type": "upload"},
            "카메라인데 주소 없음": {"input_type": "camera"},
            "허용 안 된 카메라": {"input_type": "camera", "camera_url": "http://evil.example.com/video"},
            "영상이 아닌 파일": {"input_type": "upload", "video_file": video("photo.jpg")},
            "업로드+카메라 동시": {"input_type": "upload", "video_file": video(), "camera_url": CAMERA},
            "없는 동작": {"input_type": "upload", "video_file": video(), "expected_action": "dance"},
        }
        for name, data in cases.items():
            with self.subTest(name):
                r = self.client.post("/api/analyses/", data, format="multipart")
                self.assertEqual(r.status_code, 400, f"{name}: {r.data}")
        self.assertEqual(AnalysisJob.objects.filter(user=self.a).count(), 0)

    # ── Read ────────────────────────────────────
    def test_list_shows_only_my_jobs(self):
        AnalysisJob.objects.create(user=self.a, input_type="upload", title="A의 작업")
        self.client.force_authenticate(self.a)
        r = self.client.get("/api/analyses/")
        self.assertEqual([j["title"] for j in r.data], ["A의 작업"])

    def test_detail_and_status_include_results(self):
        self.client.force_authenticate(self.b)
        r = self.client.get(f"/api/analyses/{self.b_job.id}/")
        self.assertEqual(r.data["predictions"][0]["action_label"], "swing")
        r = self.client.get(f"/api/analyses/{self.b_job.id}/status/")
        self.assertEqual(r.data, {**r.data, "status": "SUCCESS", "is_finished": True, "prediction_count": 1})

    # ── T07: A는 B의 작업을 조회·수정·삭제 못 함 ──
    def test_other_users_job_is_404(self):
        self.client.force_authenticate(self.a)
        url = f"/api/analyses/{self.b_job.id}/"
        self.assertEqual(self.client.get(url).status_code, 404)
        self.assertEqual(self.client.get(url + "status/").status_code, 404)
        self.assertEqual(self.client.patch(url, {"title": "해킹"}, format="json").status_code, 404)
        self.assertEqual(self.client.delete(url).status_code, 404)
        self.b_job.refresh_from_db()
        self.assertEqual(self.b_job.title, "B의 작업")

    # ── Update ──────────────────────────────────
    def test_patch_title_memo_only(self):
        self.client.force_authenticate(self.b)
        url = f"/api/analyses/{self.b_job.id}/"
        r = self.client.patch(url, {"title": "새 제목", "memo": "메모"}, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertEqual((r.data["title"], r.data["memo"]), ("새 제목", "메모"))
        for field, value in {"status": "FAILURE", "input_type": "camera", "expected_action": "sit_floor"}.items():
            with self.subTest(field):
                self.assertEqual(self.client.patch(url, {field: value}, format="json").status_code, 400)
        self.b_job.refresh_from_db()
        self.assertEqual(self.b_job.status, "SUCCESS")

    def test_put_not_allowed(self):
        self.client.force_authenticate(self.b)
        r = self.client.put(f"/api/analyses/{self.b_job.id}/", {"title": "x"}, format="json")
        self.assertEqual(r.status_code, 405)

    # ── Delete ──────────────────────────────────
    def test_delete_finished_job_cascades(self):
        self.client.force_authenticate(self.b)
        self.assertEqual(self.client.delete(f"/api/analyses/{self.b_job.id}/").status_code, 204)
        self.assertFalse(PersonPrediction.objects.filter(job_id=self.b_job.id).exists())

    def test_delete_running_job_conflict(self):
        job = AnalysisJob.objects.create(user=self.a, input_type="upload", status="STARTED")
        self.client.force_authenticate(self.a)
        self.assertEqual(self.client.delete(f"/api/analyses/{job.id}/").status_code, 409)
        self.assertTrue(AnalysisJob.objects.filter(id=job.id).exists())


# ══════════════════════════════════════════════════════════════
# 7단계: Celery 등록 · 작업(tasks.run_analysis) — FastAPI·Redis 는 가짜(mock)로 대체
# ══════════════════════════════════════════════════════════════
FAKE_RESULT = {
    "predictions": [
        {"track_id": 1, "action_label": "swing", "confidence": 0.64, "is_uncertain": False, "valid_frames": 126, "window_count": 1},
        {"track_id": 2, "action_label": "sit_floor", "confidence": 0.23, "is_uncertain": True, "valid_frames": 80, "window_count": 1},
    ],
    "message": "ok",
}


@override_settings(MEDIA_ROOT=TMP_MEDIA, ALLOWED_CAMERA_URLS=[CAMERA], CAMERA_RELAY_BASE="", CAMERA_PREPARE_SECONDS=0)
class CeleryFlowTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("worker_test", password="x")

    def _upload_job(self):
        return AnalysisJob.objects.create(user=self.user, input_type="upload", video_file=video())

    def test_create_enqueues_after_commit_and_returns_immediately(self):
        """T05: 저장 commit 후 작업 등록, 응답은 PENDING + task ID (분석을 기다리지 않음)."""
        self.client.force_authenticate(self.user)
        fake = mock.Mock(id="task-123")
        with mock.patch("analyses.tasks.run_analysis.delay", return_value=fake) as delay:
            with self.captureOnCommitCallbacks(execute=True) as callbacks:
                r = self.client.post("/api/analyses/", {"input_type": "upload", "video_file": video()}, format="multipart")
        self.assertEqual(r.status_code, 201)
        self.assertEqual(len(callbacks), 1)  # commit 뒤에 실행되도록 등록됨
        delay.assert_called_once_with(r.data["id"])
        self.assertEqual(r.data["status"], "PENDING")  # 분석 결과를 기다리지 않고 응답
        # (테스트에서는 commit 콜백이 응답 뒤에 실행되므로 task ID는 DB에서 확인)
        self.assertEqual(AnalysisJob.objects.get(id=r.data["id"]).celery_task_id, "task-123")

    def test_redis_down_marks_failure(self):
        self.client.force_authenticate(self.user)
        with mock.patch("analyses.tasks.run_analysis.delay", side_effect=ConnectionError("redis down")):
            with self.captureOnCommitCallbacks(execute=True):
                r = self.client.post("/api/analyses/", {"input_type": "upload", "video_file": video()}, format="multipart")
        job = AnalysisJob.objects.get(id=r.data["id"])
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("Redis", job.error_message)

    def test_task_success_saves_predictions(self):
        job = self._upload_job()
        with mock.patch("analyses.tasks.request_prediction", return_value=FAKE_RESULT):
            self.assertEqual(run_analysis(job.id), "success")
        job.refresh_from_db()
        self.assertEqual(job.status, "SUCCESS")
        self.assertIsNotNone(job.started_at)
        self.assertIsNotNone(job.completed_at)
        self.assertEqual(
            list(job.predictions.values_list("track_id", "action_label", "is_uncertain")),
            [(1, "swing", False), (2, "sit_floor", True)],
        )

    def test_task_no_person_is_success_with_message(self):
        job = self._upload_job()
        result = {"predictions": [], "message": "영상에서 사람을 찾지 못했습니다."}
        with mock.patch("analyses.tasks.request_prediction", return_value=result):
            run_analysis(job.id)
        job.refresh_from_db()
        self.assertEqual((job.status, job.result_message), ("SUCCESS", "영상에서 사람을 찾지 못했습니다."))
        self.assertEqual(job.predictions.count(), 0)

    def test_task_fastapi_error_marks_failure(self):
        job = self._upload_job()
        err = AIServiceError("AI 서버(FastAPI)에 연결할 수 없습니다: http://localhost:8001")
        with mock.patch("analyses.tasks.request_prediction", side_effect=err):
            self.assertEqual(run_analysis(job.id), "failure")
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("연결할 수 없습니다", job.error_message)
        self.assertEqual(job.predictions.count(), 0)

    def test_task_unexpected_error_marks_failure(self):
        job = self._upload_job()
        with mock.patch("analyses.tasks.request_prediction", side_effect=KeyError("boom")):
            run_analysis(job.id)
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("예상하지 못한 오류", job.error_message)

    def test_task_runs_only_once(self):
        job = self._upload_job()
        with mock.patch("analyses.tasks.request_prediction", return_value=FAKE_RESULT) as req:
            run_analysis(job.id)
            self.assertEqual(run_analysis(job.id), "skipped")  # 이미 SUCCESS → 다시 실행 안 함
        self.assertEqual(req.call_count, 1)

    # ── 8단계: IP Webcam ─────────────────────────
    def test_camera_job_saves_clip_and_predicts(self):
        job = AnalysisJob.objects.create(user=self.user, input_type="camera", camera_url=CAMERA)
        clip = Path(tempfile.mkstemp(suffix=".mp4")[1])
        clip.write_bytes(b"fake-mp4")
        with mock.patch("analyses.tasks.capture_clip", return_value=clip) as cap, \
             mock.patch("analyses.tasks.request_prediction", return_value=FAKE_RESULT) as req:
            self.assertEqual(run_analysis(job.id), "success")
        cap.assert_called_once_with(CAMERA, 8, on_start=mock.ANY)  # 중계 없으면 카메라 직접
        job.refresh_from_db()
        self.assertTrue(job.video_file.name.startswith("uploads/") and "camera_job" in job.video_file.name)
        self.assertEqual(req.call_args.args[0], job.video_file.path)  # 수집 영상으로 같은 추론 경로
        self.assertFalse(clip.exists())  # 임시 파일 정리
        self.assertEqual(job.predictions.count(), 2)

    def test_camera_error_marks_failure(self):
        job = AnalysisJob.objects.create(user=self.user, input_type="camera", camera_url=CAMERA)
        err = CameraCaptureError("카메라에 연결할 수 없습니다 (…)")
        with mock.patch("analyses.tasks.capture_clip", side_effect=err), \
             mock.patch("analyses.tasks.request_prediction") as req:
            self.assertEqual(run_analysis(job.id), "failure")
        req.assert_not_called()
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("카메라에 연결할 수 없습니다", job.error_message)


class CameraCaptureTests(SimpleTestCase):
    """camera_capture.capture_clip 실제 동작 (영상 파일·닫힌 포트로 시험)."""

    def setUp(self):
        self.src = Path(tempfile.mkstemp(suffix=".avi")[1])
        w = cv2.VideoWriter(str(self.src), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for i in range(40):
            w.write(np.full((48, 64, 3), i * 5, np.uint8))
        w.release()

    def tearDown(self):
        self.src.unlink(missing_ok=True)

    def test_capture_writes_mp4(self):
        out = capture_clip(str(self.src), seconds=5, stall_seconds=0.5)
        try:
            cap = cv2.VideoCapture(str(out))
            self.assertGreaterEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            cap.release()
        finally:
            out.unlink(missing_ok=True)

    def test_unreachable_camera_fails_fast(self):
        t0 = time.monotonic()
        with self.assertRaises(CameraCaptureError) as ctx:
            capture_clip("http://127.0.0.1:9/video", seconds=8, open_timeout=2)
        self.assertIn("연결할 수 없습니다", str(ctx.exception))
        self.assertLess(time.monotonic() - t0, 10)


# ══════════════════════════════════════════════════════════════
# 9단계: 사용자 화면 · 상세 · 관리자 대시보드
# ══════════════════════════════════════════════════════════════
@override_settings(ALLOWED_CAMERA_URLS=[CAMERA])
class PageTests(APITestCase):
    def setUp(self):
        self.a = User.objects.create_user("page_a", password="x")
        self.b = User.objects.create_user("page_b", password="x")
        self.admin = User.objects.create_user("page_admin", password="x", is_staff=True)
        self.job_a = AnalysisJob.objects.create(user=self.a, input_type="upload", status="SUCCESS", title="A 작업")
        PersonPrediction.objects.create(job=self.job_a, track_id=1, action_label="swing", confidence=0.7)
        PersonPrediction.objects.create(job=self.job_a, track_id=2, action_label="swing", confidence=0.3, is_uncertain=True)
        job_b = AnalysisJob.objects.create(user=self.b, input_type="camera", camera_url=CAMERA, status="FAILURE")
        self.job_b = job_b

    def test_index_shows_form_and_my_history_only(self):
        self.client.force_login(self.a)
        r = self.client.get("/analyses/")
        html = r.content.decode()
        self.assertContains(r, "AI 분석 시작")
        self.assertIn("카메라 1", html)  # 카메라 주소 대신 이름 표시
        self.assertIn("A 작업", html)
        self.assertNotIn(f'/analyses/{self.job_b.id}/"', html)  # 남의 작업 없음

    def test_detail_own_only(self):
        self.client.force_login(self.a)
        self.assertEqual(self.client.get(f"/analyses/{self.job_a.id}/").status_code, 200)
        self.assertEqual(self.client.get(f"/analyses/{self.job_b.id}/").status_code, 404)

    def test_dashboard_admin_only(self):
        r = self.client.get("/dashboard/")
        self.assertEqual(r.status_code, 302)  # 비로그인 → 로그인
        self.client.force_login(self.a)
        r = self.client.get("/dashboard/")
        self.assertEqual(r.status_code, 302)  # 일반 회원 → 차단
        self.assertNotIn("/dashboard/", r["Location"].split("?")[0])

    def test_dashboard_numbers_come_from_db(self):
        self.client.force_login(self.admin)
        r = self.client.get("/dashboard/")
        self.assertEqual(r.status_code, 200)
        charts = r.context["chart_data"]
        idx = [a for a, _ in ACTION_CHOICES].index("swing")
        self.assertEqual((charts["action"]["certain"][idx], charts["action"]["uncertain"][idx]), (1, 1))
        self.assertEqual(sum(charts["action"]["certain"]) + sum(charts["action"]["uncertain"]), PersonPrediction.objects.count())
        status = dict(zip(charts["status"]["keys"], charts["status"]["counts"]))
        self.assertEqual((status["SUCCESS"], status["FAILURE"], status["PENDING"]), (1, 1, 0))
        # 새 분석이 생기면 숫자가 바뀌는지
        AnalysisJob.objects.create(user=self.a, input_type="upload", status="PENDING")
        status = dict(zip(*[self.client.get("/dashboard/").context["chart_data"]["status"][k] for k in ("keys", "counts")]))
        self.assertEqual(status["PENDING"], 1)


# ══════════════════════════════════════════════════════════════
# IP Webcam 실시간 미리보기 · 카메라 중계 (camera_relay.py)
# ══════════════════════════════════════════════════════════════
@override_settings(ALLOWED_CAMERA_URLS=[CAMERA], CAMERA_RELAY_BASE="http://django:8000", CAMERA_RELAY_TOKEN="tok123")
class CameraRelayTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("relay_user", password="x")

    def test_worker_uses_relay_when_configured(self):
        from .tasks import camera_source

        self.assertEqual(camera_source(CAMERA), "http://django:8000/analyses/camera/1/feed/?token=tok123")
        with override_settings(CAMERA_RELAY_BASE=""):
            self.assertEqual(camera_source(CAMERA), CAMERA)

    def test_preview_requires_login_and_known_camera(self):
        self.assertEqual(self.client.get("/analyses/camera/1/status/").status_code, 302)  # 비로그인 → 로그인
        self.client.force_login(self.user)
        self.assertEqual(self.client.get("/analyses/camera/9/status/").status_code, 404)  # 허용 목록 밖

    def test_feed_requires_token(self):
        self.assertEqual(self.client.get("/analyses/camera/1/feed/").status_code, 403)
        self.assertEqual(self.client.get("/analyses/camera/1/feed/?token=wrong").status_code, 403)

    def test_hub_shares_frames_and_streams_mjpeg(self):
        from . import camera_relay

        src = Path(tempfile.mkstemp(suffix=".avi")[1])
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
        for i in range(60):
            w.write(np.full((48, 64, 3), i * 4, np.uint8))
        w.release()
        try:
            hub = camera_relay.CameraHub(str(src))
            fid, jpg = hub.wait_frame(-1, timeout=5)
            self.assertTrue(jpg and jpg[:2] == b"\xff\xd8")  # JPEG
            chunk = next(camera_relay.mjpeg_stream(hub))  # 두 번째 소비자도 같은 허브에서 받음
            self.assertTrue(chunk.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        finally:
            hub.last_access = 0  # 아무도 안 봄 → 스레드 종료
            src.unlink(missing_ok=True)


class CameraRotationTests(APITestCase):
    def test_rotate_button_cycles_and_frames_turn_upright(self):
        from . import camera_relay

        src = Path(tempfile.mkstemp(suffix=".avi")[1])
        w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))  # 가로 64 × 세로 48
        for _ in range(40):
            w.write(np.zeros((48, 64, 3), np.uint8))
        w.release()
        try:
            hub = camera_relay.CameraHub(str(src), rotation=90)
            hub.wait_frame(-1, timeout=5)
            self.assertEqual(hub.size, (48, 64))  # 90도 돌리면 세로 화면
        finally:
            hub.last_access = 0
            src.unlink(missing_ok=True)

    @override_settings(ALLOWED_CAMERA_URLS=[CAMERA], CAMERA_ROTATION=0)
    def test_rotate_endpoint(self):
        from . import camera_relay

        camera_relay._hubs.pop(CAMERA, None)
        user = User.objects.create_user("rot_user", password="x")
        self.assertEqual(self.client.post("/analyses/camera/1/rotate/").status_code, 302)  # 비로그인
        self.client.force_login(user)
        self.assertEqual(self.client.get("/analyses/camera/1/rotate/").status_code, 405)  # POST 만
        self.assertEqual(self.client.post("/analyses/camera/1/rotate/").json()["rotation"], 90)
        self.assertEqual(self.client.post("/analyses/camera/1/rotate/").json()["rotation"], 180)
        camera_relay._hubs[CAMERA].last_access = 0


@override_settings(MEDIA_ROOT=TMP_MEDIA, ALLOWED_CAMERA_URLS=[CAMERA], CAMERA_RELAY_BASE="", CAMERA_PREPARE_SECONDS=5)
class CameraPrepareTests(APITestCase):
    def test_prepare_then_capture_and_trim(self):
        user = User.objects.create_user("prep_user", password="x")
        job = AnalysisJob.objects.create(user=user, input_type="camera", camera_url=CAMERA)
        clip = Path(tempfile.mkstemp(suffix=".mp4")[1])
        clip.write_bytes(b"fake-mp4")
        seen = []

        def fake_sleep(sec):  # 준비 시간 동안 화면에 '준비 중' 이 보이는지
            seen.append((sec, AnalysisJob.objects.get(id=job.id).result_message))

        with mock.patch("analyses.tasks.time.sleep", side_effect=fake_sleep), \
             mock.patch("analyses.tasks.capture_clip", return_value=clip), \
             mock.patch("analyses.tasks.request_prediction", return_value=FAKE_RESULT) as req:
            run_analysis(job.id)
        self.assertEqual(seen, [(5, "카메라 준비 중")])
        self.assertTrue(req.call_args.kwargs["trim_idle"])  # 카메라는 움직인 구간만 판단
        job.refresh_from_db()
        self.assertEqual((job.status, job.result_message), ("SUCCESS", ""))


class RuleNoteTests(APITestCase):
    @override_settings(MEDIA_ROOT=TMP_MEDIA)
    def test_rule_note_saved_and_shown(self):
        user = User.objects.create_user("rule_user", password="x")
        job = AnalysisJob.objects.create(user=user, input_type="upload", video_file=video())
        result = {"predictions": [{"track_id": 1, "action_label": "grab_shake", "confidence": 0.3, "is_uncertain": True,
                                   "valid_frames": 90, "rule": {"reason": "팔꿈치를 4회 접었다 폄 → 여러 번 = 잡고 흔들기"}}],
                  "message": "ok"}
        with mock.patch("analyses.tasks.request_prediction", return_value=result):
            run_analysis(job.id)
        self.client.force_authenticate(user)
        p = self.client.get(f"/api/analyses/{job.id}/").data["predictions"][0]
        self.assertIn("팔꿈치를 4회", p["rule_note"])


# ══════════════════════════════════════════════════════════════
# 코드리뷰 수정: 멈춘 작업 정리 · 녹화 중 회전 막기 · 토큰 로그 가리기
# ══════════════════════════════════════════════════════════════
@override_settings(MEDIA_ROOT=TMP_MEDIA, ALLOWED_CAMERA_URLS=[CAMERA], STALE_JOB_SECONDS=300)
class ReviewFixTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user("fix_user", password="x")
        self.client.force_authenticate(self.user)

    def test_stale_started_job_becomes_failure_and_deletable(self):
        from datetime import timedelta

        from django.utils import timezone

        old = timezone.now() - timedelta(seconds=600)
        stale = AnalysisJob.objects.create(user=self.user, input_type="upload", status="STARTED", started_at=old)
        fresh = AnalysisJob.objects.create(user=self.user, input_type="upload", status="STARTED", started_at=timezone.now())
        st = self.client.get(f"/api/analyses/{stale.id}/status/").json()
        self.assertEqual((st["status"], st["is_finished"]), ("FAILURE", True))
        self.assertEqual(self.client.get(f"/api/analyses/{fresh.id}/status/").json()["status"], "STARTED")
        self.assertEqual(self.client.delete(f"/api/analyses/{stale.id}/").status_code, 204)

    def test_soft_time_limit_marks_failure(self):
        from celery.exceptions import SoftTimeLimitExceeded

        job = AnalysisJob.objects.create(user=self.user, input_type="upload", video_file=video())
        with mock.patch("analyses.tasks.request_prediction", side_effect=SoftTimeLimitExceeded()):
            self.assertEqual(run_analysis(job.id), "failure")
        job.refresh_from_db()
        self.assertEqual(job.status, "FAILURE")
        self.assertIn("제한 시간", job.error_message)

    def test_rotate_blocked_while_recording(self):
        from . import camera_relay

        camera_relay._hubs.pop(CAMERA, None)
        self.client.force_login(self.user)
        job = AnalysisJob.objects.create(user=self.user, input_type="camera", camera_url=CAMERA, status="STARTED")
        r = self.client.post("/analyses/camera/1/rotate/")
        self.assertEqual(r.status_code, 409)
        self.assertNotIn(CAMERA, camera_relay._hubs)  # 회전값을 건드리지 않음
        job.status = "SUCCESS"
        job.save()
        self.assertEqual(self.client.post("/analyses/camera/1/rotate/").status_code, 200)
        camera_relay._hubs[CAMERA].last_access = 0


class CaptureSizeChangeTests(SimpleTestCase):
    def test_frame_size_change_mid_capture_fails(self):
        frames = [np.zeros((48, 64, 3), np.uint8)] * 5 + [np.zeros((64, 48, 3), np.uint8)] * 5
        fake = mock.MagicMock()
        fake.isOpened.return_value = True
        fake.read.side_effect = [(True, f) for f in frames]
        with mock.patch("analyses.camera_capture.cv2.VideoCapture", return_value=fake):
            with self.assertRaises(CameraCaptureError) as ctx:
                capture_clip("http://x/video", seconds=5)
        self.assertIn("회전", str(ctx.exception))


class TokenLogRedactionTests(SimpleTestCase):
    def test_token_hidden_in_server_log(self):
        import logging

        from .apps import RedactTokenFilter

        record = logging.LogRecord("django.server", logging.INFO, "", 0, '"%s" %s %s',
                                   ("GET /analyses/camera/1/feed/?token=secret123&x=1 HTTP/1.1", "200", "0"), None)
        RedactTokenFilter().filter(record)
        self.assertNotIn("secret123", record.getMessage())
        self.assertIn("token=***&x=1", record.getMessage())
