"""POST /predict/video — Celery Worker가 보낸 영상 1건을 받아 사람별 예측 JSON을 돌려줍니다.

상황별 응답 (2단계 API 계약)
- 200 + predictions 1개 이상 : 정상
- 200 + predictions []       : 영상은 정상인데 사람이 없음
- 400                        : 파일 형식 오류 · 빈 파일 · 영상 손상
- 413                        : 파일이 너무 큼
- 422                        : 너무 짧음 / 유효 프레임 부족
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from services.action_inference import predict_people
from services.pose_pipeline import VideoReadError, extract_tracks, read_video_info

router = APIRouter(tags=["predictions"])

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))


def _uncertain_threshold() -> float:
    return float(os.getenv("UNCERTAIN_CONFIDENCE", "0.40"))


@router.post("/predict/video")
def predict_video(
    request: Request,
    file: UploadFile = File(..., description="분석할 영상 (mp4 등)"),
    trim_idle: bool = Query(False, description="True: 사람마다 움직인 구간만 판단 (IP Webcam 처럼 앞뒤에 가만히 있는 시간이 긴 영상)"),
):
    """영상 1건 → YOLO Pose·사람별 추적 → team17 Feature(363) → ExtraTrees (17개 동작)."""
    started = time.perf_counter()
    bundle = request.app.state.bundle

    # ① 입력 검사: 확장자
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"지원하지 않는 파일 형식입니다: '{suffix or '없음'}' (허용: {sorted(ALLOWED_EXTENSIONS)})")

    # ② 임시 파일로 저장 (OpenCV·YOLO는 파일 경로로 읽음)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        size_mb = tmp_path.stat().st_size / 1024 / 1024
        if size_mb == 0:
            raise HTTPException(400, "빈 파일입니다.")
        if size_mb > MAX_UPLOAD_MB:
            raise HTTPException(413, f"파일이 너무 큽니다: {size_mb:.1f}MB (최대 {MAX_UPLOAD_MB}MB)")

        # ③ 영상이 읽히는지 · 너무 짧지 않은지
        try:
            info = read_video_info(tmp_path)
        except VideoReadError as e:
            raise HTTPException(400, str(e)) from e
        if 0 < info.frames < bundle.min_valid_frames:
            raise HTTPException(
                422, f"영상이 너무 짧습니다: {info.frames}프레임 (최소 {bundle.min_valid_frames}프레임 필요)"
            )

        # ④ 추론
        try:
            info, tracks = extract_tracks(tmp_path, bundle)
        except VideoReadError as e:
            raise HTTPException(400, str(e)) from e
        predictions, skipped = predict_people(tracks, bundle, _uncertain_threshold(), trim_idle=trim_idle)
    finally:
        tmp_path.unlink(missing_ok=True)

    # ⑤ 결과 정리
    if not tracks.data:
        message = "영상에서 사람을 찾지 못했습니다."
    elif not predictions:
        raise HTTPException(
            422,
            f"사람은 검출됐지만 유효 프레임이 부족합니다 (사람별 최소 {bundle.min_valid_frames}프레임): {skipped}",
        )
    else:
        message = "ok"

    return {
        "video": {
            "filename": file.filename,
            "frames": info.frames,
            "fps": round(info.fps, 2),
            "duration_sec": round(info.duration_sec, 2),
        },
        "predictions": predictions,
        "skipped_tracks": skipped,  # 프레임이 부족하거나 잠깐 스친 사람 (예측 안 함)
        "uncertain_threshold": _uncertain_threshold(),
        "message": message,
        "processing_time_sec": round(time.perf_counter() - started, 2),
    }
