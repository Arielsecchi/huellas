"""Endpoint de muestreo: GET /api/sample.

Estrategia de fuentes (decision Fase 6):
  - Arco         -> RealPool (el GAN v1 colapso esa clase).
  - I, E, V      -> GANSampler.

Devuelve JSON `{id, png_b64, klass_asked_preview, source}`.

`klass_asked_preview` es solo para debug en consola (ver Network tab); el
juego REAL no lo usa para validar — eso lo hace el backend en /api/answer.
Si querés esconderlo del todo, sacalo del response model.
"""

from __future__ import annotations

import base64
import random

from fastapi import APIRouter, Request

from src.data.vucetich import LABEL_TO_SYMBOL, VucetichClass

router = APIRouter()

# Distribucion del muestreo: uniforme entre las 4 clases. La app es para
# practicar, asi que queremos ver las 4 con la misma frecuencia (la
# distribucion natural 8.3/33.1/37.1/21.5 sesgaria la practica al inicio).
_RNG = random.Random()


@router.get("/sample")
def get_sample(request: Request) -> dict:
    """Genera una huella, la persiste como pendiente y la devuelve al cliente."""
    klass = _RNG.choice(list(VucetichClass))
    sym = LABEL_TO_SYMBOL[klass]

    if klass == VucetichClass.ARCO:
        png = request.app.state.real_pool.sample_png(klass)
        source = "real"
    else:
        png = request.app.state.gan_sampler.sample_png(klass)
        source = "gan"

    attempt_id = request.app.state.db_insert_pending(sym, source)
    return {
        "id": attempt_id,
        "png_b64": base64.b64encode(png).decode("ascii"),
        "source": source,
        # NO incluimos klass_asked: el cliente NO debe saber la respuesta hasta
        # que mande su intento a /api/answer.
    }
