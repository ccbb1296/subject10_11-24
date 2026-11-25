"""IP Webcam 중계 — 카메라에 '한 번만' 연결해서 여러 곳(브라우저 미리보기 · Worker 녹화)에 같은 화면을 나눠 줍니다.

왜 필요한가?
  DroidCam 은 동시에 한 클라이언트만 연결됩니다. 브라우저가 휴대폰에 직접 붙으면 Worker 가 녹화를 못 하고
  ("Busy"), Worker 가 녹화 중이면 미리보기가 안 됩니다. → Django 안의 CameraHub 가 카메라 연결을 독점하고
  최신 프레임(JPEG)을 보관 → 미리보기(MJPEG)와 Worker 녹화(MJPEG)가 둘 다 여기서 받아 감.

동작
  - 누군가 프레임을 요청하면 그때 카메라에 연결 (백그라운드 스레드)
  - 아무도 안 보면 IDLE_SECONDS 뒤 연결 해제 → 다른 프로그램이 카메라를 쓸 수 있음
"""

from __future__ import annotations

import threading
import time

import cv2

IDLE_SECONDS = 5.0  # 이 시간 동안 아무도 안 보면 카메라 연결 해제
# 휴대폰을 세워 찍어도 DroidCam 은 가로 화면으로 보냄 → 시계 방향 회전 (미리보기·녹화 모두 적용)
# 누운 사람으로 보이면 '넘어지기·눕기'로 잘못 판단하므로 사람이 똑바로 서 보이게 맞춰야 함
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
OPEN_TIMEOUT_MS = 5000
RETRY_SECONDS = 2.0  # 연결 실패 시 다시 시도 간격
BOUNDARY = "frame"


class CameraHub:
    def __init__(self, url: str, rotation: int = 0):
        self.url = url
        self.rotation = rotation if rotation in ROTATIONS else 0
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.frame_id = 0
        self.size: tuple[int, int] | None = None
        self.error = ""
        self.connected = False
        self.last_access = 0.0
        self._thread: threading.Thread | None = None

    # ── 외부에서 쓰는 함수 ──────────────────────
    def touch(self) -> None:
        """'지금 보고 있다' 표시 + 연결 스레드가 없으면 시작."""
        self.last_access = time.monotonic()
        with self.cond:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name=f"camera-hub:{self.url}", daemon=True)
                self._thread.start()

    def wait_frame(self, last_id: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        """last_id 보다 새 프레임이 올 때까지 기다렸다가 (frame_id, jpeg) 반환."""
        self.touch()
        with self.cond:
            # 새 프레임이 '실제로 있을 때'까지 대기 (첫 화면 전에 빈 값으로 바로 돌아가 헛도는 것 방지)
            self.cond.wait_for(lambda: self.frame_id != last_id and self.jpeg is not None, timeout=timeout)
            return self.frame_id, self.jpeg

    def status(self) -> dict:
        self.touch()
        return {"connected": self.connected, "error": self.error, "size": self.size, "rotation": self.rotation}

    def set_rotation(self, degrees: int) -> None:
        if degrees not in ROTATIONS:
            raise ValueError("rotation 은 0, 90, 180, 270 중 하나")
        self.rotation = degrees

    # ── 카메라 읽기 스레드 ──────────────────────
    def _idle(self) -> bool:
        return time.monotonic() - self.last_access > IDLE_SECONDS

    def _run(self) -> None:
        while not self._idle():
            cap = cv2.VideoCapture(
                self.url, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS, cv2.CAP_PROP_READ_TIMEOUT_MSEC, OPEN_TIMEOUT_MS],
            )
            if not cap.isOpened():
                cap.release()
                self.connected = False
                self.error = "카메라에 연결할 수 없습니다. 휴대폰 DroidCam 앱이 켜져 있는지, 같은 와이파이인지, 다른 앱이 연결돼 있지 않은지 확인하세요."
                time.sleep(RETRY_SECONDS)
                continue
            self.connected, self.error = True, ""
            fails = 0
            try:
                while not self._idle():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        fails += 1
                        if fails > 30:  # 약 1~2초 동안 화면이 안 옴 → 다시 연결
                            self.error = "카메라 영상이 끊겼습니다. 다시 연결 중…"
                            break
                        time.sleep(0.05)
                        continue
                    fails = 0
                    if ROTATIONS[self.rotation] is not None:
                        frame = cv2.rotate(frame, ROTATIONS[self.rotation])
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        continue
                    with self.cond:
                        self.jpeg = buf.tobytes()
                        self.size = (frame.shape[1], frame.shape[0])
                        self.frame_id += 1
                        self.cond.notify_all()
            finally:
                cap.release()
                self.connected = False
        with self.cond:  # 아무도 안 봐서 종료
            self.jpeg = None
            self.cond.notify_all()


_hubs: dict[str, CameraHub] = {}
_hubs_lock = threading.Lock()


def get_hub(url: str, rotation: int = 0) -> CameraHub:
    """카메라 주소별 허브 1개. rotation 은 처음 만들 때의 기본값 (.env CAMERA_ROTATION), 이후 화면 버튼으로 변경."""
    with _hubs_lock:
        if url not in _hubs:
            _hubs[url] = CameraHub(url, rotation)
        return _hubs[url]


def mjpeg_stream(hub: CameraHub, max_wait: float = 15.0):
    """MJPEG(multipart/x-mixed-replace) 생성기 — 브라우저 <img> 와 OpenCV 둘 다 읽을 수 있는 형식.
    새 프레임이 max_wait 초 동안 없으면 종료 (카메라 꺼짐)."""
    last = -1
    waited = 0.0
    while True:
        t0 = time.monotonic()
        fid, jpg = hub.wait_frame(last, timeout=2.0)
        if jpg is None or fid == last:
            waited += time.monotonic() - t0
            if waited > max_wait:
                return
            continue
        waited, last = 0.0, fid
        yield (
            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
            + jpg + b"\r\n"
        )
