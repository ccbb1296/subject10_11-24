"""team17 동작 인식 모델(17개 동작)과 YOLO Pose를 서버 시작 때 한 번만 불러와 재사용합니다.

models/team17/
- clip_model.joblib   : ExtraTrees (영상 전체 통계 Feature 363개 → 17개 동작) — 팀 5명 데이터로 학습한 최종 모델
- hybrid_info.json    : classes, clip_feature_names(순서), window_weight, 교차검증 결과
- action_classes.csv  : action_label ↔ 한글 이름
models/yolo11n-pose.pt + project.json : YOLO Pose 설정
"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from ultralytics import YOLO

from src.common import load_config, resolve_device
from src.team17_features import keypoints_to_features
from src.team17_tracking import MIN_FRAMES

AI_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = AI_ROOT / "models"
TEAM17_DIR = MODELS_DIR / "team17"


@dataclass
class ModelBundle:
    clip_model: object
    window_model: object | None
    window_weight: float
    classes: list[str]
    names_ko: dict[str, str]
    feature_names: list[str]
    min_valid_frames: int
    pose_model_path: Path
    image_size: int
    person_confidence: float
    device: object
    cv_summary: dict
    # YOLO는 여러 요청이 동시에 쓰면 안전하지 않아서 한 번에 하나씩 사용
    pose_lock: threading.Lock = field(default_factory=threading.Lock)
    _pose_model: YOLO | None = None

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def pose_model(self) -> YOLO:
        if self._pose_model is None:
            self._pose_model = YOLO(str(self.pose_model_path))
        return self._pose_model


def _dummy_feature_count() -> int:
    """가짜 키포인트로 Feature 함수를 한 번 돌려 실제로 만들어지는 길이를 확인."""
    rng = np.random.default_rng(0)
    xy = (rng.normal(size=(30, 17, 2)) * 50 + 300).astype(np.float32)
    clip_vec, _ = keypoints_to_features(np.arange(30), xy, 30.0)
    return len(clip_vec)


def load_models() -> ModelBundle:
    """모델을 읽고 학습 조건(Class·Feature 수·순서)이 맞는지 검사합니다. 하나라도 다르면 서버가 뜨지 않음."""
    info = json.loads((TEAM17_DIR / "hybrid_info.json").read_text(encoding="utf-8"))
    if info.get("clip_feature_set") != "clip_stats":
        raise RuntimeError(f"지원하지 않는 Feature 방식: {info.get('clip_feature_set')}")
    w = float(info["window_weight"])
    clip_model = joblib.load(TEAM17_DIR / "clip_model.joblib") if w < 1 else None  # 직접 학습한 신뢰 가능한 파일만
    window_model = joblib.load(TEAM17_DIR / "window_model.joblib") if w > 0 else None
    with (TEAM17_DIR / "action_classes.csv").open(encoding="utf-8-sig") as f:
        names_ko = {row["action_label"]: row["action_name_ko"] for row in csv.DictReader(f)}
    cfg = load_config()

    model = clip_model or window_model
    bundle = ModelBundle(
        clip_model=clip_model,
        window_model=window_model,
        window_weight=w,
        classes=[str(c) for c in model.classes_],
        names_ko=names_ko,
        feature_names=list(info["clip_feature_names"]),
        min_valid_frames=MIN_FRAMES,
        pose_model_path=AI_ROOT / cfg["pose_model"],
        image_size=int(cfg["image_size"]),
        person_confidence=float(cfg["person_confidence"]),
        device=resolve_device(cfg.get("device", "auto")),
        cv_summary={c["protocol"]: {"accuracy": c["accuracy"], "macro_f1": c["macro_f1"]} for c in info.get("cross_validation", [])},
    )

    # ── 학습 조건 검사 ──
    n_in = getattr(model, "n_features_in_", None)
    made = _dummy_feature_count()
    if not (n_in == bundle.feature_count == made):
        raise RuntimeError(f"Feature 수 불일치: 모델 {n_in}, JSON {bundle.feature_count}, Feature 함수 {made}")
    if bundle.classes != [str(c) for c in info["classes"]]:
        raise RuntimeError(f"Class 불일치: 모델 {bundle.classes} / JSON {info['classes']}")
    if set(bundle.classes) - set(names_ko):
        raise RuntimeError(f"한글 이름 없는 동작: {set(bundle.classes) - set(names_ko)}")
    if not bundle.pose_model_path.exists():
        raise RuntimeError(f"YOLO Pose 가중치 없음: {bundle.pose_model_path}")

    bundle.pose_model()  # 첫 요청이 느리지 않게 미리 로드
    return bundle
