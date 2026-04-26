"""Pool de huellas REALES de SOCOFing por clase Vucetich.

Razon de existir: el modelo v1 colapso sobre la clase Arco (8.3% del dataset
es muy poco para que la cDCGAN aprenda esa cola). Para ofrecer las 4 clases
con calidad uniforme en la app de practica, servimos:

  - Arcos        -> imagenes REALES de SOCOFing
  - I, E, V      -> imagenes generadas por el GAN (`inference.py`)

Asi el usuario ve siempre material correcto para entrenar el ojo, y la
limitacion del GAN queda contenida en una sola clase.

El npz entero (~90 MB) cabe en RAM facilmente. Lo cargamos una sola vez
al startup y guardamos solo los indices de cada clase para muestrear barato.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.vucetich import LABEL_TO_SYMBOL, VucetichClass

# PROJECT_ROOT = huellas-gan/. data/processed/ vive ahi.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "processed" / "images.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"


class RealPool:
    """Banco en memoria de imagenes reales indexadas por clase."""

    def __init__(self, images: np.ndarray,
                 indices_by_class: dict[VucetichClass, list[int]],
                 rng: random.Random) -> None:
        self.images = images   # (N, 128, 128) uint8
        self.indices_by_class = indices_by_class
        self.rng = rng

    @classmethod
    def load(cls,
             images_path: Path = DEFAULT_IMAGES,
             metadata_path: Path = DEFAULT_METADATA,
             seed: int | None = None) -> "RealPool":
        if not images_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Faltan datos procesados:\n  {images_path}\n  {metadata_path}\n"
                "Corre antes: python -m src.data.preprocess && "
                "python -m src.data.label_vucetich --full")
        images = np.load(images_path)["images"]
        with open(metadata_path, encoding="utf-8") as f:
            meta = list(csv.DictReader(f))
        indices: dict[VucetichClass, list[int]] = {c: [] for c in VucetichClass}
        for i, row in enumerate(meta):
            sym = row["vucetich_symbol"]
            for klass, klass_sym in LABEL_TO_SYMBOL.items():
                if sym == klass_sym:
                    indices[klass].append(i)
                    break
        for klass, idxs in indices.items():
            if not idxs:
                raise RuntimeError(
                    f"No hay imagenes reales para la clase {LABEL_TO_SYMBOL[klass]}")
        rng = random.Random(seed)
        return cls(images, indices, rng)

    def sample_uint8(self, klass: VucetichClass) -> np.ndarray:
        """Devuelve UNA imagen real uint8 (128, 128) de la clase pedida."""
        idx = self.rng.choice(self.indices_by_class[klass])
        return self.images[idx]

    def sample_png(self, klass: VucetichClass) -> bytes:
        """Wrapper a PNG bytes."""
        from .inference import _uint8_to_png_bytes
        return _uint8_to_png_bytes(self.sample_uint8(klass))
