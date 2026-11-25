"""team17 원본 스크립트에서 추론에 필요한 함수·상수를 그대로(verbatim) 뽑아 src/team17_features.py 생성."""
import ast, sys
from pathlib import Path
AI_ROOT = Path(__file__).resolve().parents[1]               # vision_action_platform/ai_service
SRC = AI_ROOT.parents[1] / "scripts"                      # subject10_pose_action/scripts (팀 원본)
OUT = AI_ROOT / "src" / "team17_features.py"

def pick(path, names):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef,)) and node.name in names:
            found[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    found[t.id] = ast.get_source_segment(text, node)
                if isinstance(t, ast.Tuple) and any(isinstance(e, ast.Name) and e.id in names for e in t.elts):
                    found[",".join(e.id for e in t.elts)] = ast.get_source_segment(text, node)
    return found

train = pick(SRC / "team17_day06_train.py", {
    "NOSE", "L_HIP", "torso_scale", "motion_frame_features", "smooth", "direction_changes",
    "dynamics", "dynamic_frame_features", "clip_summary_features", "MIN_VALID_FRAMES"})
window = pick(SRC / "team17_window_model.py", {
    "WINDOW_SEC", "STRIDE_SEC", "MIN_WINDOW_FRAMES", "segment_features", "window_slices", "keypoints_to_features"})
webcam = pick(SRC / "team17_ip_webcam_test.py", {"UNCERTAIN", "MIN_FRAMES"})

order_train = ["NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR".replace(" ", ""), "L_HIP,R_HIP,L_AN,R_AN", "MIN_VALID_FRAMES",
               "torso_scale", "motion_frame_features", "smooth", "direction_changes", "dynamics",
               "dynamic_frame_features", "clip_summary_features"]
order_window = ["WINDOW_SEC", "STRIDE_SEC", "MIN_WINDOW_FRAMES", "segment_features", "window_slices", "keypoints_to_features"]

parts = ['''"""team17 (17개 동작) 추론용 Feature — 원본 코드를 그대로 복사한 모듈 (수정 금지).

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
''']
for k in order_train:
    parts.append(train[k] + "\n")
parts.append('''

class _T:  # team17_window_model.py 가 T.xxx 로 부르던 이름을 그대로 쓰기 위한 연결
    motion_frame_features = staticmethod(motion_frame_features)
    dynamic_frame_features = staticmethod(dynamic_frame_features)
    clip_summary_features = staticmethod(clip_summary_features)


T = _T

# ── team17_window_model.py ──
''')
for k in order_window:
    parts.append(window[k] + "\n")
OUT.write_text("\n".join(parts), encoding="utf-8")
missing = [k for k in order_train if k not in train] + [k for k in order_window if k not in window]
print("생성:", OUT, "/ 빠진 것:", missing or "없음", "/ webcam 상수:", webcam)

# ── Tracks (사람별 키포인트 모으기 + 간단 추적) : team17_ip_webcam_test.py ──
text = (SRC / "team17_ip_webcam_test.py").read_text(encoding="utf-8")
tree = ast.parse(text)
tracks = next(ast.get_source_segment(text, n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tracks")
OUT2 = OUT.parent / "team17_tracking.py"
OUT2.write_text('''"""team17 사람별 키포인트 수집·추적 — team17_ip_webcam_test.py 의 Tracks 클래스를 그대로 복사 (수정 금지).

YOLO Pose 결과를 프레임마다 add() 하면, 직전 위치와 가까운 사람에게 같은 track_id 를 붙여
사람별 (시각, 관절 17개 좌표) 를 모읍니다. 1초 넘게 안 보이면 새 ID.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

MIN_FRAMES = 10  # 사람별로 이 프레임 수 이상 잡혀야 예측 (원본 상수)


''' + tracks + "\n", encoding="utf-8")
print("생성:", OUT2)
