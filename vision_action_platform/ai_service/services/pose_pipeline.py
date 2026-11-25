"""영상 → YOLO Pose → 사람(track_id)별 (시각, 관절 17개 좌표) 모음.

team17 웹 데모(team17/server.py run_upload)와 같은 순서로 처리합니다 (학습·검증 때와 같은 입력):
- 프레임 긴 변을 960px 로 줄임 (shrink)
- 40fps 넘는 영상은 2프레임마다 1장 (학습 데이터 추출과 같은 약 30fps)
- YOLO Pose predict → Tracks.add (직전 위치와 가까운 사람에게 같은 ID)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from services.model_loader import ModelBundle
from src.team17_tracking import Tracks

WORK_LONG_SIDE = 960  # team17/server.py 와 같은 값
# 이 값보다 낮은 사람 검출은 버림 (YOLO 자체 기준 person_confidence 0.25 는 그대로, 추적 단계에서 한 번 더)
MIN_PERSON_CONF = float(os.getenv("MIN_PERSON_CONF", "0.5"))


class VideoReadError(Exception):
    """영상 파일을 열 수 없거나 프레임이 없음."""


@dataclass
class VideoInfo:
    frames: int
    fps: float
    duration_sec: float
    width: int
    height: int


def shrink(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = WORK_LONG_SIDE / max(h, w)
    return cv2.resize(frame, (int(w * s), int(h * s))) if s < 1 else frame


def read_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise VideoReadError("영상을 열 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식입니다.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, _ = cap.read()
        if not ok or width <= 0 or height <= 0:
            raise VideoReadError("영상에서 프레임을 읽을 수 없습니다.")
    finally:
        cap.release()
    if fps <= 0:
        fps = 30.0  # FPS 정보가 없는 영상은 30으로 가정
    return VideoInfo(frames, fps, frames / fps if frames else 0.0, width, height)


def _drop_weak_people(result):
    """신뢰도가 낮은 '사람' 검출 제거 — 옷걸이에 걸린 겉옷·마네킹 등이 0.25~0.4 로 잡혀
    가장 오래 보인 사람으로 뽑히는 문제 방지 (실제 사람은 보통 0.8 이상)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return result
    keep = (boxes.conf >= MIN_PERSON_CONF).nonzero().flatten().tolist()
    return result if len(keep) == len(boxes) else result[keep]


def extract_tracks(video_path: Path, bundle: ModelBundle) -> tuple[VideoInfo, Tracks]:
    """영상 전체를 읽어 사람별 키포인트를 모은 Tracks 를 돌려줍니다."""
    info = read_video_info(video_path)
    tracks = Tracks()  # 요청마다 새로 → 이전 영상의 ID 가 섞이지 않음
    stride = 2 if info.fps > 40 else 1
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                with bundle.pose_lock:
                    result = bundle.pose_model().predict(
                        shrink(frame),
                        imgsz=bundle.image_size,
                        conf=bundle.person_confidence,
                        device=bundle.device,
                        verbose=False,
                    )[0]
                tracks.add(idx / info.fps, _drop_weak_people(result))
            idx += 1
    finally:
        cap.release()
    if idx == 0:
        raise VideoReadError("영상에서 프레임을 읽을 수 없습니다.")
    if info.frames <= 0:
        info.frames = idx
        info.duration_sec = idx / info.fps
    return info, tracks
