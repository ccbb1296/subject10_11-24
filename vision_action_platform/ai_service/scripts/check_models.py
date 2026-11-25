"""모델 준비 검증: team17 모델 파일이 제대로 복사되고 학습 조건(17개 동작·Feature 363개·순서)이 맞는지 확인.

실행 (ai_service 폴더에서):
    python scripts/check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from services.model_loader import TEAM17_DIR, load_models  # noqa: E402

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'OK' if ok else 'FAIL'}] {name}  {detail}")


for fname in ["clip_model.joblib", "hybrid_info.json", "action_classes.csv"]:
    p = TEAM17_DIR / fname
    check(f"파일 존재: team17/{fname}", p.exists(), f"{p.stat().st_size:,} bytes" if p.exists() else "")
p = AI_ROOT / "models" / "yolo11n-pose.pt"
check("파일 존재: yolo11n-pose.pt", p.exists())

try:
    b = load_models()  # 내부에서 Feature 수·Class·순서 검사 (다르면 예외)
    check("모델 로드 + 학습 조건 검사", True, type(b.clip_model).__name__)
    check("동작 17개", len(b.classes) == 17, ", ".join(b.classes))
    check("한글 이름", all(c in b.names_ko for c in b.classes), ", ".join(b.names_ko[c] for c in b.classes[:5]) + " …")
    check("Feature 수 (모델 = JSON = Feature 함수)", True, str(b.feature_count))
    check("교차검증 결과", bool(b.cv_summary), str(b.cv_summary))
    check("YOLO Pose 로드", True, f"device={b.device}")
except Exception as e:  # noqa: BLE001
    check("모델 로드 + 학습 조건 검사", False, repr(e))

failed = [r for r in results if not r[1]]
print()
print("모델 준비 검증 통과 ✅" if not failed else f"실패 {len(failed)}개 ❌")
sys.exit(1 if failed else 0)
