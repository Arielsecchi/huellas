"""FastAPI app: monta los routers + sirve el frontend estatico.

Lifespan:
  - on startup: carga el Generator (~57 MB), el RealPool (~90 MB) y abre la
    SQLite. Una sola vez.
  - on shutdown: cierra la SQLite.

Variables de entorno (todas opcionales):
  HUELLAS_WEIGHTS   path al generator.pt (default: huellas_out_final v1)
  HUELLAS_DB        path a la sqlite (default: app/backend/data/stats.db)
  HUELLAS_DEVICE    "cpu" | "cuda" (default: auto)

Uso dev:
  uvicorn app.backend.main:app --reload --port 8000

Uso prod local:
  uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .db import (DEFAULT_DB_PATH, init_db, insert_pending,
                 purge_stale_pending, record_answer, stats_overall)
from .inference import DEFAULT_WEIGHTS, GANSampler
from .real_samples import RealPool
from .routes.samples import router as samples_router
from .routes.stats import router as stats_router

# PROJECT_ROOT = huellas-gan/. El frontend vive en app/frontend/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"

logger = logging.getLogger("huellas")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    weights_path = Path(os.environ.get("HUELLAS_WEIGHTS", DEFAULT_WEIGHTS))
    db_path = Path(os.environ.get("HUELLAS_DB", DEFAULT_DB_PATH))
    device_pref = os.environ.get("HUELLAS_DEVICE")
    device = (torch.device(device_pref) if device_pref
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("device      = %s", device)
    logger.info("weights     = %s", weights_path)
    logger.info("db          = %s", db_path)

    # GAN
    app.state.gan_sampler = GANSampler.load(weights_path, device=device)
    logger.info("GAN cargado (z_dim=%d)", app.state.gan_sampler.z_dim)

    # Pool de reales (para Arcos, ver real_samples.py)
    from src.data.vucetich import LABEL_TO_SYMBOL
    app.state.real_pool = RealPool.load()
    counts = {LABEL_TO_SYMBOL[k]: len(idxs)
              for k, idxs in app.state.real_pool.indices_by_class.items()}
    logger.info("RealPool counts = %s", counts)

    # SQLite
    conn = init_db(db_path)
    purged = purge_stale_pending(conn)
    if purged:
        logger.info("limpie %d intentos pendientes viejos", purged)
    app.state.db_conn = conn
    # Bind helpers para que los routers no importen db.py directo:
    app.state.db_insert_pending = lambda klass_sym, source: insert_pending(
        conn, klass_sym, source)
    app.state.db_record_answer = lambda att_id, sym: record_answer(
        conn, att_id, sym)
    app.state.db_stats_overall = lambda: stats_overall(conn)

    try:
        yield
    finally:
        conn.close()
        logger.info("db cerrada, bye")


app = FastAPI(title="Huellas Vucetich — practica",
              version="0.1.0",
              lifespan=lifespan)

# CORS abierto: la app es local, no hay riesgos. Si en algun momento se sube
# a algun host publico, restringir.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(samples_router, prefix="/api", tags=["samples"])
app.include_router(stats_router, prefix="/api", tags=["stats"])

# Frontend estatico montado AL FINAL: en Starlette un mount en "/" cachea
# todo lo que no matcheo antes. Por eso health, samples y stats van arriba.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True),
              name="frontend")
else:
    logger.warning("FRONTEND_DIR no existe: %s", FRONTEND_DIR)
