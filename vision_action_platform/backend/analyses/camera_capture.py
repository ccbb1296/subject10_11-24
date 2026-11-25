"""허용된 IP Webcam에서 정해진 시간(기본 8초)만 프레임을 읽어 MP4 한 개로 저장합니다.

- 누가 읽나?   Celery Worker (웹 요청 안에서 카메라를 붙잡지 않음)
- 어디서 읽나? Django 카메라 중계(camera_relay.py) — 브라우저 미리보기와 같은 화면 (DroidCam 동시 접속 1개라서)
- 언제 끝나나? 첫 화면이 들어온 뒤 `seconds` 초가 지나면 종료
               연결 실패 / 프레임이 `stall_seconds` 이상 안 들어옴 / 전체 제한 시간 초과 → 오류
- 만든 MP4는 업로드 영상과 똑같이 FastAPI `/predict/video` 로 보냄 (추론 경로 재사용)
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

MIN_FRAMES = 10  # FastAPI(학습 정보 min_valid_frames)와 같은 최소 프레임 수


class CameraCaptureError(Exception):
    """카메라 연결 실패·끊김·시간 초과 (사용자에게 보여줄 문장)."""


def _open(camera_url: str, open_timeout: float) -> cv2.VideoCapture:
    ms = int(open_timeout * 1000)
    cap = cv2.VideoCapture(
        camera_url,
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms],
    )
    if not cap.isOpened():
        cap.release()
        # 주소는 오류 문구에 넣지 않음 (중계 토큰·카메라 IP 가 화면·DB 에 남지 않게)
        raise CameraCaptureError(
            "카메라에 연결할 수 없습니다. 휴대폰 DroidCam 앱이 켜져 있는지(화면이 꺼지면 멈춤), 같은 와이파이인지, "
            "다른 프로그램이 카메라를 쓰고 있지 않은지(DroidCam은 한 번에 1곳만 연결) 확인하세요."
        )
    return cap


def capture_clip(
    camera_url: str,
    seconds: float,
    *,
    open_timeout: float = 5.0,
    stall_seconds: float = 3.0,
    on_start: Callable[[], None] | None = None,
) -> Path:
    """카메라에서 `seconds` 초 동안 프레임을 모아 임시 MP4 경로를 돌려줍니다 (호출한 쪽이 삭제).

    `seconds` 는 **첫 화면이 들어온 순간부터** 셉니다 (연결 시간 제외). 그 순간 on_start() 를 불러
    화면의 카운트다운을 실제 녹화와 맞춥니다.
    """
    cap = _open(camera_url, open_timeout)
    # 실제 FPS는 카메라·와이파이마다 달라서, 메모리에 JPEG로 모은 뒤 실제 시간으로 FPS를 계산해 저장
    frames: list[bytes] = []
    size: tuple[int, int] | None = None
    opened_at = time.monotonic()
    started: float | None = None  # 첫 프레임 시각
    last_frame_at = opened_at
    hard_limit = seconds + open_timeout + stall_seconds + 5  # 전체 제한 시간
    try:
        while True:
            now = time.monotonic()
            if started is not None and now - started >= seconds:
                break
            if now - opened_at > hard_limit:
                raise CameraCaptureError(f"카메라 수집 시간이 초과되었습니다 ({hard_limit:.0f}초).")
            ok, frame = cap.read()
            if not ok or frame is None:
                wait_limit = stall_seconds if started is not None else open_timeout + stall_seconds
                if time.monotonic() - last_frame_at > wait_limit:
                    if len(frames) >= MIN_FRAMES and camera_url_is_file(camera_url):
                        break  # (시험용 파일 입력이 끝난 경우)
                    raise CameraCaptureError(
                        f"카메라 영상이 끊겼습니다 ({stall_seconds:.0f}초 동안 화면이 들어오지 않음, "
                        f"받은 프레임 {len(frames)}개)."
                    )
                time.sleep(0.05)
                continue
            last_frame_at = time.monotonic()
            if started is None:  # 첫 화면 → 여기서부터 녹화 시간 시작
                started = last_frame_at
                if on_start:
                    on_start()
            if size is None:
                size = (frame.shape[1], frame.shape[0])
            elif (frame.shape[1], frame.shape[0]) != size:
                # 녹화 중 화면 회전·해상도 변경 → 늘려 맞추면 사람이 찌그러져 잘못 판단하므로 실패로 알림
                raise CameraCaptureError("녹화 중에 카메라 화면 크기(회전)가 바뀌었습니다. 회전을 맞춘 뒤 다시 시도하세요.")
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                frames.append(jpg.tobytes())
    finally:
        cap.release()

    elapsed = max(time.monotonic() - (started or opened_at), 1e-6)
    if len(frames) < MIN_FRAMES or size is None:
        raise CameraCaptureError(
            f"카메라에서 받은 화면이 너무 적습니다 ({len(frames)}프레임 / {elapsed:.1f}초, 최소 {MIN_FRAMES}프레임)."
        )

    # 실제로 받은 속도로 FPS 계산 (비정상 값 방지: 1~60 범위)
    fps = min(max(len(frames) / elapsed, 1.0), 60.0)
    out = Path(tempfile.mkstemp(prefix="camera_", suffix=".mp4")[1])
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    try:
        for data in frames:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img.shape[1::-1] != size:
                img = cv2.resize(img, size)
            writer.write(img)
    finally:
        writer.release()
    return out


def camera_url_is_file(camera_url: str) -> bool:
    return "://" not in camera_url
