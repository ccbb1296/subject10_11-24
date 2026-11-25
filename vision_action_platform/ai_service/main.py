"""FastAPI AI 추론 서버 (DB 없음 — 추론 결과 JSON만 반환).

실행 (ai_service 폴더에서):
    uvicorn main:app --host 0.0.0.0 --port 8001
    → http://localhost:8001/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# 최상위 vision_action_platform/.env (UNCERTAIN_CONFIDENCE 등)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from routers import predictions  # noqa: E402
from services.model_loader import load_models  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 때 모델을 한 번만 로드 → 모든 요청에서 재사용
    app.state.bundle = load_models()
    yield


app = FastAPI(
    title="Vision Action AI Service",
    description="팀 17개 동작 모델(ExtraTrees) + YOLO Pose로 영상 속 사람별 동작을 예측합니다.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(predictions.router)


@app.get("/health", tags=["health"])
def health():
    bundle = getattr(app.state, "bundle", None)
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model": "team17 ExtraTrees (clip_stats)",
        "classes": bundle.classes if bundle else [],
        "feature_count": bundle.feature_count if bundle else None,
        "cross_validation": bundle.cv_summary if bundle else None,
        # resolve_device 는 GPU면 0, 아니면 "cpu" 를 돌려줌
        "device": (f"cuda:{bundle.device}" if isinstance(bundle.device, int) else str(bundle.device)) if bundle else None,
    }
