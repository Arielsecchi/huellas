"""Pre-proceso de SOCOFing para el entrenamiento de la GAN.

Por cada imagen:
  - abrimos en grayscale (los 3 canales RGB del BMP son identicos, los descartamos)
  - resize a IMG_SIZE x IMG_SIZE con cv2.INTER_AREA
  - guardamos como uint8 en un array (N, IMG_SIZE, IMG_SIZE)

Salida:
  - data/processed/images.npz       array (N, IMG_SIZE, IMG_SIZE) uint8
  - data/processed/metadata.csv     metadata parseada por imagen, misma N y orden

La normalizacion a [-1, 1] se aplica al vuelo dentro del Dataset PyTorch
para evitar duplicar el dataset en float32.

Uso: python -m src.data.preprocess [--size 128]
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import socofing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = PROJECT_ROOT / "data" / "raw" / "socofing" / "Real"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_PATH = PROCESSED_DIR / "images.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.csv"

DEFAULT_SIZE = 128


def preprocess_one(path: Path, size: int) -> np.ndarray:
    """Lee una BMP, devuelve un array (size, size) uint8 grayscale."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def build_dataset(size: int = DEFAULT_SIZE) -> None:
    paths = sorted(REAL_DIR.glob(f"*{socofing.IMAGE_EXT}"))
    if not paths:
        raise SystemExit(f"No hay imagenes en {REAL_DIR}. Correr download primero.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    n = len(paths)
    images = np.empty((n, size, size), dtype=np.uint8)
    rows = []
    for i, p in enumerate(tqdm(paths, desc="preprocesando")):
        images[i] = preprocess_one(p, size)
        m = socofing.parse_filename(p)
        rows.append({
            "index": i,
            "filename": p.name,
            "subject_id": m.subject_id,
            "gender": m.gender,
            "hand": m.hand,
            "finger": m.finger,
        })

    np.savez_compressed(IMAGES_PATH, images=images)
    print(f"[ok] imagenes: {IMAGES_PATH} shape={images.shape} dtype={images.dtype} "
          f"tam={IMAGES_PATH.stat().st_size / 1e6:.1f} MB")

    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] metadata: {METADATA_PATH} ({len(rows)} filas)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"resolucion de salida (default: {DEFAULT_SIZE})")
    args = parser.parse_args()
    build_dataset(args.size)


if __name__ == "__main__":
    main()
