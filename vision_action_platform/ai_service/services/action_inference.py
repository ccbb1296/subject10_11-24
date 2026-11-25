"""사람(track_id)별 키포인트 → team17 Feature(363) → ExtraTrees → 동작·Confidence·Top-3.

team17_ip_webcam_test.py ActionModel.predict / to_record, team17/server.py finish 와 같은 규칙.
"""

from __future__ import annotations

import numpy as np

from services.model_loader import ModelBundle
from services.rules import push_vs_shake
from src.team17_features import keypoints_to_features, smooth, torso_scale
from src.team17_tracking import Tracks

MIN_SHARE_OF_TOP = 0.15  # 가장 오래 보인 사람의 15% 미만 프레임만 잡힌 사람은 잠깐 스친 것으로 보고 제외
MIN_TRIM_FRAMES = 20  # 잘라 낸 구간이 이보다 짧으면 자르지 않음


def predict_proba(times: np.ndarray, xy: np.ndarray, bundle: ModelBundle) -> np.ndarray:
    # 학습 Feature 는 '초' 단위 시간만 사용 → 밀리초를 프레임 번호처럼 넘기고 fps=1000 (원본과 동일)
    ids = np.round((times - times[0]) * 1000).astype(np.int64)
    clip_vec, wins = keypoints_to_features(ids, xy.astype(np.float32), 1000.0)
    proba = np.zeros(len(bundle.classes))
    if bundle.window_model is not None:
        proba += bundle.window_weight * bundle.window_model.predict_proba(wins).mean(axis=0)
    if bundle.clip_model is not None:
        proba += (1 - bundle.window_weight) * bundle.clip_model.predict_proba(clip_vec[None])[0]
    return proba


def to_record(track_id: int, proba: np.ndarray, times: np.ndarray, bundle: ModelBundle, uncertain_threshold: float) -> dict:
    order = np.argsort(-proba)
    top3 = [
        {"action_label": bundle.classes[i], "action_name_ko": bundle.names_ko.get(bundle.classes[i], ""),
         "confidence": round(float(proba[i]), 4)}
        for i in order[:3]
    ]
    best = top3[0]
    return {
        "track_id": int(track_id),
        "action_label": best["action_label"],  # model.classes_ 순서 = 확률 배열 순서
        "action_name_ko": best["action_name_ko"],
        "confidence": best["confidence"],
        "is_uncertain": best["confidence"] < uncertain_threshold,
        "valid_frames": int(len(times)),
        "duration_sec": round(float(times[-1] - times[0]), 2),
        "window_count": 1,  # 영상(수집 구간) 전체를 한 번에 판단
        "top3": top3,
    }


def motion_segment(ts: np.ndarray, xy: np.ndarray, pad: float = 0.5, rel: float = 0.25, min_speed: float = 0.15):
    """움직인 구간 [a, z) — 앞뒤로 가만히 서 있던 시간을 잘라 냄.

    학습 영상은 동작 구간만 잘라 둔 것이라 '영상 길이'도 Feature 로 쓰임 (앞차기 평균 3.2초, 위에 앉기 7초).
    IP Webcam 은 항상 8초를 녹화하므로, 그대로 넣으면 짧은 동작이 긴 동작(앉기·넘어지기)으로 기움.
    관절 평균 속도(몸통 길이/초)가 최대값의 25% 또는 0.15 를 넘는 첫~마지막 구간 + 앞뒤 0.5초.
    """
    s = smooth(xy, 5)
    scale = torso_scale(xy)
    speed = np.linalg.norm(np.diff(s, axis=0), axis=2).mean(axis=1) / scale / np.maximum(np.diff(ts), 1e-3)
    speed = np.convolve(speed, np.ones(5) / 5, mode="same")
    moving = np.where(speed > max(rel * speed.max(), min_speed))[0]
    if len(moving) == 0:
        return 0, len(ts)
    keep = np.where((ts >= ts[moving[0]] - pad) & (ts <= ts[moving[-1] + 1] + pad))[0]
    a, z = int(keep[0]), int(keep[-1]) + 1
    return (a, z) if z - a >= MIN_TRIM_FRAMES else (0, len(ts))


def predict_people(
    tracks: Tracks, bundle: ModelBundle, uncertain_threshold: float, trim_idle: bool = False
) -> tuple[list[dict], list[dict]]:
    """(예측한 사람 목록, 프레임이 부족하거나 잠깐 스쳐서 제외한 사람 목록).
    trim_idle=True (IP Webcam) 이면 사람마다 움직인 구간만 판단."""
    predictions = []
    for tid, ts, xy in tracks.ready(bundle.min_valid_frames):
        a, z = motion_segment(ts, xy) if trim_idle else (0, len(ts))
        seg_t, seg_xy = ts[a:z] - ts[a], xy[a:z]
        proba = predict_proba(seg_t, seg_xy, bundle)
        proba, rule_note = push_vs_shake(proba, bundle.classes, seg_xy)  # 밀치기↔잡고 흔들기 보정 (rules.py)
        record = to_record(tid, proba, seg_t, bundle, uncertain_threshold)
        record["valid_frames"] = int(len(ts))  # 잡힌 프레임 수 (사람 선택 기준)
        if rule_note:
            record["rule"] = rule_note
        if trim_idle:
            record["analyzed_range_sec"] = [round(float(ts[a] - ts[0]), 2), round(float(ts[z - 1] - ts[0]), 2)]
        predictions.append(record)
    skipped = [
        {"track_id": int(tid), "valid_frames": len(d["t"])}
        for tid, d in tracks.data.items()
        if len(d["t"]) < bundle.min_valid_frames
    ]
    if predictions:
        top = max(p["valid_frames"] for p in predictions)
        kept = [p for p in predictions if p["valid_frames"] >= MIN_SHARE_OF_TOP * top]
        skipped += [{"track_id": p["track_id"], "valid_frames": p["valid_frames"]} for p in predictions if p not in kept]
        predictions = kept
    predictions.sort(key=lambda p: p["track_id"])
    return predictions, sorted(skipped, key=lambda s: s["track_id"])
