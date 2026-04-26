"""Endpoints de respuesta y estadisticas.

POST /api/answer  -> body {id: int, klass_answered: "A"|"I"|"E"|"V"}
                     resp {correct: bool, klass_asked: str, current_streak: int,
                           best_streak: int, accuracy: float}
GET  /api/stats   -> resp con totales y desglose por clase
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.data.vucetich import LABEL_TO_SYMBOL, VucetichClass

router = APIRouter()

VALID_SYMBOLS = {LABEL_TO_SYMBOL[c] for c in VucetichClass}


class AnswerIn(BaseModel):
    id: int = Field(ge=1)
    klass_answered: str = Field(min_length=1, max_length=1)


@router.post("/answer")
def post_answer(payload: AnswerIn, request: Request) -> dict:
    if payload.klass_answered not in VALID_SYMBOLS:
        raise HTTPException(
            status_code=400,
            detail=f"klass_answered debe ser uno de {sorted(VALID_SYMBOLS)}")
    result = request.app.state.db_record_answer(
        payload.id, payload.klass_answered)
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"intento {payload.id} no existe")
    stats = request.app.state.db_stats_overall()
    return {
        "correct": bool(result["correct"]),
        "klass_asked": result["klass_asked"],
        "klass_answered": result["klass_answered"],
        "source": result["source"],
        "current_streak": stats["current_streak"],
        "best_streak": stats["best_streak"],
        "accuracy": stats["accuracy"],
        "total": stats["total"],
    }


@router.get("/stats")
def get_stats(request: Request) -> dict:
    return request.app.state.db_stats_overall()
