"""team17 (17개 동작) 추론용 Feature — 원본 코드를 그대로 복사한 모듈 (수정 금지).

원본 (subject10_pose_action/scripts)
- team17_day06_train.py   : torso_scale, motion_frame_features, dynamic_frame_features, clip_summary_features …
- team17_window_model.py  : segment_features, window_slices, keypoints_to_features
학습 때와 **같은 함수**를 써야 같은 숫자가 나오므로 내용을 바꾸지 않습니다.
(scripts/build_team17_features.py 로 원본에서 다시 생성 가능)
"""

from __future__ import annotations

import numpy as np

from src.feature_utils import selected_joint_angles

# ── team17_day06_train.py ──

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10

L_HIP, R_HIP, L_AN, R_AN = 11, 12, 15, 16

MIN_VALID_FRAMES = 10

def torso_scale(xy: np.ndarray) -> float:
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    shoulder = (xy[:, L_SH] + xy[:, R_SH]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    return float(np.median(torso[torso > 1])) if (torso > 1).any() else 1.0

def motion_frame_features(xy: np.ndarray) -> np.ndarray:
    """(T,17,2) → (T,42). 영상 단위 몸통 길이로 정규화해서 누워도 값이 폭주하지 않게 함."""
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    shoulder = (xy[:, L_SH] + xy[:, R_SH]) / 2
    scale = torso_scale(xy)

    norm = (xy - hip[:, None, :]) / scale
    angles = np.array([list(selected_joint_angles(k).values()) for k in xy], dtype=np.float32)
    angles = np.nan_to_num(angles, nan=180.0)
    hip_move = (hip - hip[0]) / scale  # 첫 프레임 대비 골반 이동
    vec = shoulder - hip
    tilt = np.degrees(np.arctan2(np.abs(vec[:, 0]), -vec[:, 1]))  # 0=똑바로 섬, 90=누움
    extent = xy.max(axis=1) - xy.min(axis=1)
    aspect = extent[:, 1] / np.maximum(extent[:, 0], 1.0)  # 세로/가로: 서면 크고 누우면 작음

    feats = np.concatenate(
        [norm.reshape(len(xy), -1), angles, hip_move, tilt[:, None], aspect[:, None]], axis=1
    ).astype(np.float32)
    return feats

def smooth(a: np.ndarray, k: int = 5) -> np.ndarray:
    """시간축 이동평균 (관절 떨림 때문에 속도가 튀는 것을 줄임)."""
    if len(a) < k:
        return a
    pad = k // 2
    padded = np.concatenate([np.repeat(a[:1], pad, axis=0), a, np.repeat(a[-1:], pad, axis=0)], axis=0)
    kernel = np.ones(k) / k
    flat = padded.reshape(len(padded), -1)
    out = np.stack([np.convolve(flat[:, i], kernel, mode="valid") for i in range(flat.shape[1])], axis=1)
    return out.reshape(a.shape)

def direction_changes(v: np.ndarray, thresh: float) -> int:
    """속도 부호가 바뀐 횟수(일정 속도 이상만) — 앞뒤로 흔들면 커지고 한 번 밀면 작음."""
    sign = np.sign(v) * (np.abs(v) > thresh)
    sign = sign[sign != 0]
    return int((np.diff(sign) != 0).sum()) if len(sign) > 1 else 0

def dynamics(xy: np.ndarray, t: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """부드럽게 한 좌표와 속도(몸통 길이/초). 이미지 y는 아래로 갈수록 커짐."""
    s = smooth(xy)
    v = np.gradient(s, t, axis=0) / scale
    return s, v

def dynamic_frame_features(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(T,17,2) → (T,6): 양 손목 속도 x,y, 두 손 거리, 코 위아래 속도."""
    scale = torso_scale(xy)
    s, v = dynamics(xy, t, scale)
    hand_dist = np.linalg.norm(s[:, L_WR] - s[:, R_WR], axis=1) / scale
    return np.concatenate([v[:, L_WR], v[:, R_WR], hand_dist[:, None], v[:, NOSE, 1:2]], axis=1).astype(np.float32)

def clip_summary_features(xy: np.ndarray, t: np.ndarray, motion: np.ndarray) -> dict[str, float]:
    """영상 전체를 요약한 값. 샷건/밀치기/흔들기, 눕기/앉기/넘어지기를 가르는 용도."""
    scale = torso_scale(xy)
    s, v = dynamics(xy, t, scale)
    dt = np.gradient(t)
    shoulder = (s[:, L_SH] + s[:, R_SH]) / 2
    hip = (s[:, L_HIP] + s[:, R_HIP]) / 2
    n = len(xy)
    head, tail = slice(0, max(1, n // 5)), slice(n - max(1, n // 5), n)
    out: dict[str, float] = {}

    # 손목: 샷건(아래로 세게) / 밀치기(한 번 뻗음) / 잡고 흔들기(여러 번 왕복) / 휘두르기(궤적 김)
    wrist = {}
    for side, wr, sh in (("l", L_WR, L_SH), ("r", R_WR, R_SH)):
        vel = v[:, wr]
        speed = np.linalg.norm(vel, axis=1)
        wrist[side] = {
            "max_speed": speed.max(),
            "max_down_speed": vel[:, 1].max(),
            "max_up_speed": (-vel[:, 1]).max(),
            "max_side_speed": np.abs(vel[:, 0]).max(),
            "path_len": float((speed * dt).sum()),
            "dir_changes_y": direction_changes(vel[:, 1], 1.0),
            "dir_changes_x": direction_changes(vel[:, 0], 1.0),
            "height_max": ((shoulder[:, 1] - s[:, wr, 1]) / scale).max(),  # +면 어깨보다 위
            "height_min": ((shoulder[:, 1] - s[:, wr, 1]) / scale).min(),
            "reach_max": (np.linalg.norm(s[:, wr] - s[:, sh], axis=1) / scale).max(),
        }
    for k in wrist["l"]:
        a, b = wrist["l"][k], wrist["r"][k]
        out[f"wrist_l_{k}"], out[f"wrist_r_{k}"] = a, b
        out[f"wrist_hi_{k}"], out[f"wrist_lo_{k}"] = max(a, b), min(a, b)

    hand_dist = np.linalg.norm(s[:, L_WR] - s[:, R_WR], axis=1) / scale
    out.update(hand_dist_mean=hand_dist.mean(), hand_dist_min=hand_dist.min(),
               hand_dist_max=hand_dist.max(), hand_dist_std=hand_dist.std())

    # 발목: 앞차기
    ank = {}
    for side, an in (("l", L_AN), ("r", R_AN)):
        ank[side] = (np.linalg.norm(v[:, an], axis=1).max(), ((hip[:, 1] - s[:, an, 1]) / scale).max())
    for i, k in enumerate(("max_speed", "height_max")):
        a, b = ank["l"][i], ank["r"][i]
        out.update({f"ankle_l_{k}": a, f"ankle_r_{k}": b, f"ankle_hi_{k}": max(a, b), f"ankle_lo_{k}": min(a, b)})

    # 몸 전체: 넘어지기(빠르게 내려감) vs 눕기(천천히) vs 앉기(상체가 세워진 채 끝남)
    hip_v = (v[:, L_HIP] + v[:, R_HIP]) / 2
    tilt, aspect = motion[:, 40], motion[:, 41]
    tilt_rate = np.abs(np.gradient(tilt, t))
    out.update(
        hip_max_down_speed=hip_v[:, 1].max(), hip_max_up_speed=(-hip_v[:, 1]).max(),
        hip_max_side_speed=np.abs(hip_v[:, 0]).max(),
        hip_total_dy=(hip[tail, 1].mean() - hip[head, 1].mean()) / scale,
        hip_total_dx=abs(hip[tail, 0].mean() - hip[head, 0].mean()) / scale,
        nose_max_down_speed=v[:, NOSE, 1].max(),
        nose_total_dy=(s[tail, NOSE, 1].mean() - s[head, NOSE, 1].mean()) / scale,
        tilt_start=tilt[head].mean(), tilt_end=tilt[tail].mean(), tilt_max=tilt.max(),
        tilt_max_rate=tilt_rate.max(),
        aspect_start=aspect[head].mean(), aspect_end=aspect[tail].mean(), aspect_min=aspect.min(),
    )

    # 관절각 변화 범위 (팔꿈치: 뻗기/접기 반복, 무릎: 앉기·차기)
    angles = motion[:, 34:38]
    for i, name in enumerate(("l_elbow", "r_elbow", "l_knee", "r_knee")):
        out[f"{name}_angle_range"] = float(np.ptp(angles[:, i]))
        out[f"{name}_angle_min"] = float(angles[:, i].min())
    return {k: float(val) for k, val in out.items()}



class _T:  # team17_window_model.py 가 T.xxx 로 부르던 이름을 그대로 쓰기 위한 연결
    motion_frame_features = staticmethod(motion_frame_features)
    dynamic_frame_features = staticmethod(dynamic_frame_features)
    clip_summary_features = staticmethod(clip_summary_features)


T = _T

# ── team17_window_model.py ──

WINDOW_SEC = 2.0

STRIDE_SEC = 0.25

MIN_WINDOW_FRAMES = 10

def segment_features(xy: np.ndarray, t: np.ndarray, motion: np.ndarray, dyn: np.ndarray) -> np.ndarray:
    m = motion.copy()
    m[:, 38:40] -= m[0, 38:40]  # 골반 이동량은 구간 시작 기준으로 다시
    frame = np.concatenate([m, dyn], axis=1)
    stats = [frame.mean(0), frame.std(0), frame.min(0), frame.max(0),
             np.percentile(frame, 10, axis=0), np.percentile(frame, 90, axis=0)]
    summary = list(T.clip_summary_features(xy, t, motion).values())
    vec = np.concatenate(stats + [summary, [t[-1] - t[0]]]).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

def window_slices(t: np.ndarray) -> list[np.ndarray]:
    if t[-1] - t[0] <= WINDOW_SEC:
        return [np.ones(len(t), dtype=bool)]
    out, s = [], t[0]
    while s + WINDOW_SEC <= t[-1] + 1e-6:
        idx = (t >= s) & (t < s + WINDOW_SEC)
        if idx.sum() >= MIN_WINDOW_FRAMES:
            out.append(idx)
        s += STRIDE_SEC
    return out or [np.ones(len(t), dtype=bool)]

def keypoints_to_features(ids: np.ndarray, xy: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """키포인트 → (영상 전체 Feature, 구간 Feature 배열). 웹·실시간에서도 이 함수를 그대로 씀."""
    t = ids / fps
    motion = T.motion_frame_features(xy)
    dyn = T.dynamic_frame_features(xy, t)
    clip_vec = segment_features(xy, t, motion, dyn)
    wins = [segment_features(xy[i], t[i], motion[i], dyn[i]) for i in window_slices(t)]
    return clip_vec, np.stack(wins)
