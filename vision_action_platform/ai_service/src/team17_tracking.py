"""team17 사람별 키포인트 수집·추적 — team17_ip_webcam_test.py 의 Tracks 클래스를 그대로 복사 (수정 금지).

YOLO Pose 결과를 프레임마다 add() 하면, 직전 위치와 가까운 사람에게 같은 track_id 를 붙여
사람별 (시각, 관절 17개 좌표) 를 모읍니다. 1초 넘게 안 보이면 새 ID.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

MIN_FRAMES = 10  # 사람별로 이 프레임 수 이상 잡혀야 예측 (원본 상수)


class Tracks:
    def __init__(self, keep_sec: float | None = None):
        self.data = defaultdict(lambda: {"t": deque(), "xy": deque()})
        self.keep_sec = keep_sec
        self.last_center: dict[int, tuple[np.ndarray, float]] = {}
        self.next_id = 1

    def _assign_ids(self, centers: list[np.ndarray], t: float, max_dist: float) -> list[int]:
        """간단한 추적: 직전 위치와 가장 가까운 사람에게 같은 ID (1초 넘게 안 보이면 새 ID)."""
        alive = {tid: c for tid, (c, lt) in self.last_center.items() if t - lt <= 1.0}
        pairs = sorted((np.linalg.norm(c - pc), i, tid) for i, c in enumerate(centers) for tid, pc in alive.items())
        ids, used = [None] * len(centers), set()
        for dist, i, tid in pairs:
            if dist <= max_dist and ids[i] is None and tid not in used:
                ids[i] = tid
                used.add(tid)
        for i in range(len(ids)):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
        for i, tid in enumerate(ids):
            self.last_center[tid] = (centers[i], t)
        return ids

    def add(self, t: float, result) -> list[tuple[int, np.ndarray, np.ndarray]]:
        drawn = []
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return drawn
        xy_all = result.keypoints.xy.cpu().numpy()
        conf_all = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else np.ones(xy_all.shape[:2])
        box_conf = result.boxes.conf.cpu().numpy()
        keep = [i for i, xy in enumerate(xy_all) if np.isfinite(xy).all() and not (xy == 0).all()]
        keep.sort(key=lambda i: -box_conf[i])
        xy_all, conf_all = xy_all[keep], conf_all[keep]
        centers = [result.boxes.xywh[i, :2].cpu().numpy() for i in keep]
        max_dist = 0.25 * float(np.hypot(*result.orig_shape))
        ids = self._assign_ids(centers, t, max_dist)
        for tid, xy, conf in zip(ids, xy_all, conf_all):
            d = self.data[int(tid)]
            d["t"].append(t)
            d["xy"].append(xy)
            drawn.append((int(tid), xy, conf))
        if self.keep_sec:
            for d in self.data.values():
                while d["t"] and d["t"][0] < t - self.keep_sec:
                    d["t"].popleft()
                    d["xy"].popleft()
        return drawn

    def ready(self, min_frames: int = MIN_FRAMES):
        for tid, d in self.data.items():
            if len(d["t"]) >= min_frames:
                yield tid, np.array(d["t"]), np.stack(d["xy"])
