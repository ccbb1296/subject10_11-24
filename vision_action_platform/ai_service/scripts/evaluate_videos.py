"""FastAPI와 같은 추론 코드(services/)로 team17 영상 여러 개를 예측해 정답률을 계산합니다.

파일 이름 규칙: S01_A013_T01.mp4 → 사람 S01, 동작 A013(shotgun), 촬영 T01
⚠ 최종 모델은 팀 영상 전체로 학습했으므로, 이 결과는 "학습 데이터를 다시 맞히는지" 확인용입니다.
  실제 성능은 hybrid_info.json 의 교차검증(사람 단위) 값을 보세요.

실행 (ai_service 폴더에서):
    python scripts/evaluate_videos.py ~/ai_vision/subject10_pose_action/data/team17/videos --takes T01
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(AI_ROOT.parent / ".env")

from services.action_inference import predict_people  # noqa: E402
from services.model_loader import TEAM17_DIR, load_models  # noqa: E402
from services.pose_pipeline import extract_tracks  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--takes", default="T01,T02,T03,T04",
                    help="쉼표로 구분. T01 정면, T02 옆, T03 뒤, T04 = T02(옆) 좌우반전 증강 (모델은 4가지 모두로 학습)")
parser.add_argument("--out", default=str(AI_ROOT / "reports" / "evaluation_team17.csv"))
args = parser.parse_args()

with (TEAM17_DIR / "action_classes.csv").open(encoding="utf-8-sig") as f:
    ACTION_IDS = {row["action_id"]: row["action_label"] for row in csv.DictReader(f)}
takes = set(args.takes.split(","))
threshold = float(os.getenv("UNCERTAIN_CONFIDENCE", "0.40"))
bundle = load_models()
rows = []
for video in sorted(Path(args.root).expanduser().rglob("*.mp4")):
    m = re.match(r"(S\d+)_(A\d+)_(T\d+)", video.name)
    if not m or m.group(2) not in ACTION_IDS or m.group(3) not in takes:
        continue
    subject, truth, take = m.group(1), ACTION_IDS[m.group(2)], m.group(3)
    t0 = time.perf_counter()
    _, tracks = extract_tracks(video, bundle)
    preds, _ = predict_people(tracks, bundle, threshold)
    best = max(preds, key=lambda p: p["valid_frames"]) if preds else None  # 가장 오래 보인 사람 = 주인공
    rows.append({
        "video": video.name, "subject": subject, "take": take, "truth": truth,
        "pred": best["action_label"] if best else "-",
        "confidence": best["confidence"] if best else 0.0,
        "uncertain": best["is_uncertain"] if best else True,
        "top3": "|".join(t["action_label"] for t in best["top3"]) if best else "",
        "correct": bool(best and best["action_label"] == truth),
        "sec": round(time.perf_counter() - t0, 2),
    })
    r = rows[-1]
    print(f"{'✅' if r['correct'] else '❌'} {r['video']:<20} 정답 {truth:<17} 예측 {r['pred']:<17} {r['confidence']:.2f}"
          f"{' (불확실)' if r['uncertain'] else ''}")


def acc(items):
    return sum(r["correct"] for r in items) / len(items) if items else float("nan")


TAKE_NAMES = {"T01": "정면", "T02": "옆", "T03": "뒤", "T04": "옆 좌우반전"}
print(f"\n=== 정답률 (촬영 {sorted(takes)}) ===")
for key, title in (("subject", "사람별"), ("take", "촬영별")):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    print(f"[{title}]")
    for g, items in sorted(groups.items()):
        name = f" ({TAKE_NAMES[g]})" if key == "take" and g in TAKE_NAMES else ""
        print(f"  {g}{name}  {sum(r['correct'] for r in items)}/{len(items)} = {acc(items):.0%}")
by_action = defaultdict(list)
for r in rows:
    by_action[r["truth"]].append(r)
print("[동작별 — 틀린 것만]")
for a, items in sorted(by_action.items()):
    wrong = [f"{r['subject']}_{r['take']}→{r['pred']}" for r in items if not r["correct"]]
    if wrong:
        print(f"  {a:<17} {len(items) - len(wrong)}/{len(items)}  {', '.join(wrong)}")
print(f"전체 {sum(r['correct'] for r in rows)}/{len(rows)} = {acc(rows):.0%}")
top3 = sum(r["truth"] in r["top3"].split("|") for r in rows)
print(f"Top-3 안에 정답 {top3}/{len(rows)} = {top3 / len(rows):.0%}")
sure = [r for r in rows if not r["uncertain"]]
print(f"확실(≥{threshold}) {sum(r['correct'] for r in sure)}/{len(sure)} = {acc(sure):.0%} / 불확실 {len(rows) - len(sure)}개")
print("교차검증(사람 단위, 실제 성능):", bundle.cv_summary)

out = Path(args.out)
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"저장: {out}")
