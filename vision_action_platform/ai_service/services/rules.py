"""모델 결과 뒤에 붙이는 보정 규칙 (재학습 없음).

밀치기(push) ↔ 잡고 흔들기(grab_shake) 는 팀 교차검증에서도 서로 가장 많이 헷갈린 짝
(잡고 흔들기 정답률 53%, 밀치기 80%). 둘 다 '팔을 앞으로 뻗는' 동작이라 영상 통계로는 비슷하지만,
  - 밀치기     : 팔꿈치를 한 번 폈다가 돌아옴 (뻗기+되돌리기 → 세면 2~2.5회)
  - 잡고 흔들기 : 팔꿈치를 여러 번 접었다 폈다 함 (정면·옆 3.5~6회)
→ 모델의 1·2위가 정확히 이 두 동작일 때만, 팔꿈치 각도가 크게 바뀐 횟수로 하나를 고름.

검증 (팀 교차검증 255개, 학습에 안 쓴 사람, 기준 3.5회): 밀치기 80% → 87%, 잡고 흔들기 53% → 60%,
전체 217 → 219, 다른 동작 영향 없음 (표본이 작아 참고 수준)
"""

from __future__ import annotations

import numpy as np

from src.feature_utils import selected_joint_angles
from src.team17_features import smooth

PAIR = ("push", "grab_shake")
ELBOW_SWING_DEG = 25.0  # 이만큼 이상 각도가 바뀌어야 '접기/펴기' 한 번으로 셈
# 접었다 편 횟수가 이 이상이면 잡고 흔들기.
# 3.0 → 3.5: 밀치기 뒤 돌아서 걸어 나갈 때 팔 흔들림(30~50°)이 한 번 더 세져 3회가 된 실제 녹화가 잡고 흔들기로 바뀜.
#            교차검증 전체 정답 수는 219 로 같음 (밀치기 12→13, 잡고 흔들기 10→9)
SHAKE_MIN_CYCLES = 3.5


def swing_cycles(signal: np.ndarray, min_change: float) -> float:
    """신호가 min_change 이상 오르내린 횟수 / 2 = 접었다 편 횟수 (작은 떨림은 무시)."""
    s = np.asarray(signal, dtype=float)
    if len(s) < 3:
        return 0.0
    turns = [0] + [k for k in range(1, len(s) - 1) if (s[k] - s[k - 1]) * (s[k + 1] - s[k]) < 0] + [len(s) - 1]
    swings, last = 0, s[turns[0]]
    for k in turns[1:]:
        if abs(s[k] - last) >= min_change:
            swings += 1
            last = s[k]
    return swings / 2


def fill_missing_keypoints(xy: np.ndarray) -> np.ndarray:
    """YOLO 가 못 찾은 관절은 (0, 0) 으로 옴 → 그대로 각도를 재면 수십~백 도씩 튀어 '접었다 펴기'로 세짐.
    관절마다 (0,0)·NaN 프레임을 앞뒤 보인 프레임 사이 직선 보간으로 채움 (끝은 가장 가까운 값).
    한 번도 안 보인 관절은 그대로 둠 (각도가 일정 → 횟수 0)."""
    out = np.array(xy, dtype=float, copy=True)
    missing = ~np.isfinite(out).all(axis=2) | (out == 0).all(axis=2)  # (T, 17)
    frames = np.arange(len(out))
    for j in range(out.shape[1]):
        miss = missing[:, j]
        if miss.any() and not miss.all():
            for c in range(2):
                out[miss, j, c] = np.interp(frames[miss], frames[~miss], out[~miss, j, c])
    return out


def elbow_cycles(xy: np.ndarray) -> float:
    """양팔 중 팔꿈치를 더 많이 접었다 편 쪽의 횟수."""
    s = smooth(fill_missing_keypoints(xy), 5)  # 빠진 관절 채우기 → 관절 떨림 줄이기 (팀 Feature 와 같은 이동평균)
    angles = np.array(
        [[a["left_elbow_angle"], a["right_elbow_angle"]] for a in (selected_joint_angles(k) for k in s)], dtype=float
    )
    angles = np.nan_to_num(angles, nan=180.0)
    return max(swing_cycles(angles[:, 0], ELBOW_SWING_DEG), swing_cycles(angles[:, 1], ELBOW_SWING_DEG))


def push_vs_shake(proba: np.ndarray, classes: list[str], xy: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """1·2위가 밀치기·잡고 흔들기면 팔꿈치 횟수로 순서를 정함. (바뀐 확률, 보정 설명 또는 None)."""
    order = np.argsort(-proba)
    top2 = {classes[order[0]], classes[order[1]]}
    if top2 != set(PAIR):
        return proba, None
    cycles = elbow_cycles(xy)
    chosen = "grab_shake" if cycles >= SHAKE_MIN_CYCLES else "push"
    note = {
        "rule": "elbow_cycles",
        "elbow_cycles": cycles,
        "model_choice": classes[order[0]],
        "chosen": chosen,
        "reason": (
            f"팔꿈치 접었다 펴기 {cycles:g}회 (기준 {SHAKE_MIN_CYCLES:g}회) → "
            + ("여러 번 반복 = 잡고 흔들기" if chosen == "grab_shake" else "한 번 뻗고 돌아옴 = 밀치기")
        ),
    }
    if chosen == classes[order[0]]:
        return proba, note
    # 1·2위 확률을 맞바꿔서 선택한 동작이 1위가 되게 (확률 합은 그대로)
    adjusted = proba.copy()
    i, j = order[0], order[1]
    adjusted[i], adjusted[j] = proba[j], proba[i]
    return adjusted, note
