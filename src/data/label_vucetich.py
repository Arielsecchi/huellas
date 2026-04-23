"""Etiquetado Vucetich por deteccion de puntos singulares (Poincare index).

Pipeline clasico (Hong et al. 1998, Kawagoe & Tojo 1984):

  1. Campo de orientacion por bloques: en cada bloque calculamos la
     orientacion dominante de las crestas con el metodo del gradiente
     al cuadrado.
  2. Mascara: descartamos bloques de fondo (intensidad alta uniforme,
     donde no hay crestas).
  3. Poincare index: en cada bloque del campo sumamos las diferencias
     angulares sobre un loop 3x3 alrededor. Valores esperados:
        +0.5  ->  nucleo (core)
        -0.5  ->  delta
         0    ->  zona normal
  4. Clasificacion por conteo de cores detectados:
        0 cores              -> Arco (sin singularidades)
        1 core               -> Presilla (I o E segun mano y posicion)
        2+ cores             -> Verticilo (dos loops concentricos)

Por que conteo por cores y no por deltas: en huellas de baja resolucion
(128x128) los deltas suelen caer cerca del borde del area impresa, donde
el campo de orientacion es ruidoso. Los cores viven en el centro y son
mucho mas estables. Perdemos precision en los bordes, pero la heuristica
resulta mas robusta en la practica.

Para distinguir Presilla Interna de Externa usamos la posicion del core
relativa al centro horizontal y la mano. En una presilla, el core esta
del lado OPUESTO al delta, y el delta del lado del pulgar = Interna:

  - mano DERECHA  + core a la DERECHA del centro  -> Interna
  - mano DERECHA  + core a la IZQUIERDA del centro -> Externa
  - mano IZQUIERDA + core a la IZQUIERDA del centro -> Interna
  - mano IZQUIERDA + core a la DERECHA del centro  -> Externa
  (regla: core del lado del meñique -> Interna)

La heuristica tiene ~70-80% de accuracy tipica contra ground truth real.

Uso:
  python -m src.data.label_vucetich              # 20 samples + visualizacion
  python -m src.data.label_vucetich --full       # aplica a las 6000 + CSV
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from . import socofing
from .vucetich import LABEL_TO_SYMBOL, VucetichClass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "images.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
SAMPLE_VIZ_PATH = OUTPUT_DIR / "vucetich_labels_sample.png"

BLOCK_SIZE = 16
POINCARE_THRESHOLD = 0.18  # |indice| > esto para considerar singular
MASK_INTENSITY_THRESHOLD = 240  # bloques con media > esto son fondo
MASK_STD_THRESHOLD = 10  # bloques con std < esto son fondo plano
MASK_EROSION_ITER = 0  # con 0 no erosionamos: deltas viven en el borde
SAMPLE_VIZ_COUNT = 20
SAMPLE_VIZ_COLS = 5
SAMPLE_SEED = 42


@dataclass
class Classification:
    klass: VucetichClass
    cores: list[tuple[int, int]]  # (row, col) en coordenadas de bloque
    deltas: list[tuple[int, int]]


def _compute_orientation_field(img: np.ndarray, block_size: int) -> np.ndarray:
    """Devuelve un campo de orientaciones (rad, 0..pi) por bloques."""
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    gxy = 2 * gx * gy
    gxx_gyy = gx ** 2 - gy ** 2

    h, w = img.shape
    nr, nc = h // block_size, w // block_size
    field = np.empty((nr, nc), dtype=np.float32)
    for i in range(nr):
        for j in range(nc):
            r0, c0 = i * block_size, j * block_size
            vx = gxy[r0:r0 + block_size, c0:c0 + block_size].sum()
            vy = gxx_gyy[r0:r0 + block_size, c0:c0 + block_size].sum()
            field[i, j] = 0.5 * np.arctan2(vx, vy)

    # suavizado suave del campo (promedio en sin/cos para evitar wrap-around)
    sin2 = np.sin(2 * field)
    cos2 = np.cos(2 * field)
    sin2 = cv2.GaussianBlur(sin2, (3, 3), 0.5)
    cos2 = cv2.GaussianBlur(cos2, (3, 3), 0.5)
    field = 0.5 * np.arctan2(sin2, cos2)
    field = np.mod(field, np.pi)
    return field


def _compute_mask(img: np.ndarray, block_size: int) -> np.ndarray:
    """True en bloques que contienen huella (no fondo)."""
    h, w = img.shape
    nr, nc = h // block_size, w // block_size
    mask = np.zeros((nr, nc), dtype=bool)
    for i in range(nr):
        for j in range(nc):
            r0, c0 = i * block_size, j * block_size
            block = img[r0:r0 + block_size, c0:c0 + block_size]
            if block.mean() < MASK_INTENSITY_THRESHOLD and block.std() > MASK_STD_THRESHOLD:
                mask[i, j] = True
    if MASK_EROSION_ITER > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel,
                         iterations=MASK_EROSION_ITER).astype(bool)
    return mask


def _poincare_at(field: np.ndarray, i: int, j: int) -> float:
    """Indice de Poincare en el loop 3x3 alrededor de (i, j)."""
    # recorrido antihorario de los 8 vecinos
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1),  (1, 1),
               (1, 0),  (1, -1), (0, -1),
               (-1, -1)]
    angles = [field[i + di, j + dj] for di, dj in offsets]
    total = 0.0
    for k in range(8):
        d = angles[k + 1] - angles[k]
        # wrap al rango (-pi/2, pi/2] para orientaciones (no vectores)
        if d > np.pi / 2:
            d -= np.pi
        elif d <= -np.pi / 2:
            d += np.pi
        total += d
    return total / (2 * np.pi)


def _cluster_points(points: list[tuple[int, int]],
                    distance: int = 2) -> list[tuple[int, int]]:
    """Colapsa puntos singulares vecinos en uno solo (su centroide)."""
    if not points:
        return []
    remaining = set(points)
    clusters: list[list[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        # BFS por vecindad Chebyshev
        frontier = [seed]
        while frontier:
            next_frontier = []
            for p in frontier:
                for q in list(remaining):
                    if abs(q[0] - p[0]) <= distance and abs(q[1] - p[1]) <= distance:
                        cluster.append(q)
                        remaining.remove(q)
                        next_frontier.append(q)
            frontier = next_frontier
        clusters.append(cluster)
    return [(int(np.mean([p[0] for p in c])), int(np.mean([p[1] for p in c])))
            for c in clusters]


def classify(img: np.ndarray, hand: str) -> Classification:
    """Clasifica una huella pre-procesada (128x128 uint8) en clase Vucetich."""
    field = _compute_orientation_field(img, BLOCK_SIZE)
    mask = _compute_mask(img, BLOCK_SIZE)

    nr, nc = field.shape
    raw_cores: list[tuple[int, int]] = []
    raw_deltas: list[tuple[int, int]] = []
    for i in range(1, nr - 1):
        for j in range(1, nc - 1):
            if not mask[i, j]:
                continue
            idx = _poincare_at(field, i, j)
            if idx > POINCARE_THRESHOLD:
                raw_cores.append((i, j))
            elif idx < -POINCARE_THRESHOLD:
                raw_deltas.append((i, j))

    cores = _cluster_points(raw_cores)
    deltas = _cluster_points(raw_deltas)

    n_cores = len(cores)
    if n_cores == 0:
        klass = VucetichClass.ARCO
    elif n_cores >= 2:
        klass = VucetichClass.VERTICILO
    else:
        # presilla: distinguir I/E por posicion del core respecto al
        # centro horizontal y la mano (core del lado del meñique = Interna)
        _, nc = field.shape
        core_col = cores[0][1]
        core_on_right_half = core_col >= nc / 2
        if hand == "Right":
            klass = (VucetichClass.PRESILLA_INTERNA if core_on_right_half
                     else VucetichClass.PRESILLA_EXTERNA)
        else:  # Left
            klass = (VucetichClass.PRESILLA_INTERNA if not core_on_right_half
                     else VucetichClass.PRESILLA_EXTERNA)

    return Classification(klass=klass, cores=cores, deltas=deltas)


def _load_dataset() -> tuple[np.ndarray, list[dict]]:
    if not IMAGES_PATH.exists() or not METADATA_PATH.exists():
        raise SystemExit("Falta correr preprocess antes: python -m src.data.preprocess")
    images = np.load(IMAGES_PATH)["images"]
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return images, meta


def _visualize_sample(images: np.ndarray, meta: list[dict]) -> None:
    rng = random.Random(SAMPLE_SEED)
    idxs = rng.sample(range(len(images)), SAMPLE_VIZ_COUNT)
    n_rows = (SAMPLE_VIZ_COUNT + SAMPLE_VIZ_COLS - 1) // SAMPLE_VIZ_COLS
    fig, axes = plt.subplots(n_rows, SAMPLE_VIZ_COLS, figsize=(14, 3 * n_rows))
    for ax, idx in zip(axes.flat, idxs):
        img = images[idx]
        hand = meta[idx]["hand"]
        finger = meta[idx]["finger"]
        result = classify(img, hand)
        symbol = LABEL_TO_SYMBOL[result.klass]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        # marcamos cores y deltas escalados al pixel
        for i, j in result.cores:
            ax.scatter(j * BLOCK_SIZE + BLOCK_SIZE // 2,
                       i * BLOCK_SIZE + BLOCK_SIZE // 2,
                       c="red", marker="o", s=80, edgecolors="white", linewidths=1.5)
        for i, j in result.deltas:
            ax.scatter(j * BLOCK_SIZE + BLOCK_SIZE // 2,
                       i * BLOCK_SIZE + BLOCK_SIZE // 2,
                       c="cyan", marker="^", s=80, edgecolors="black", linewidths=1.5)
        ax.set_title(f"#{idx} {hand[0]} {finger} -> {symbol} "
                     f"(c={len(result.cores)}, d={len(result.deltas)})",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    # tapar axes sobrantes
    for ax in axes.flat[SAMPLE_VIZ_COUNT:]:
        ax.axis("off")
    fig.suptitle("Etiquetado Vucetich (Poincare) — rojo=core, cyan=delta",
                 fontsize=12)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAMPLE_VIZ_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {SAMPLE_VIZ_PATH}")


def _label_all(images: np.ndarray, meta: list[dict]) -> None:
    """Clasifica las 6000 y reescribe metadata.csv con columna vucetich."""
    rows = []
    counts = {c: 0 for c in VucetichClass}
    for i, row in enumerate(tqdm(meta, desc="etiquetando")):
        result = classify(images[i], row["hand"])
        counts[result.klass] += 1
        rows.append({
            **row,
            "vucetich": int(result.klass),
            "vucetich_symbol": LABEL_TO_SYMBOL[result.klass],
            "n_cores": len(result.cores),
            "n_deltas": len(result.deltas),
        })
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"[ok] metadata reescrito con labels en {METADATA_PATH}")
    print("[distribucion]")
    for c, n in counts.items():
        print(f"  {LABEL_TO_SYMBOL[c]} {c.name:<20s} {n:>4d}  ({100*n/total:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="aplica a las 6000 y reescribe metadata.csv")
    args = parser.parse_args()

    images, meta = _load_dataset()
    if args.full:
        _label_all(images, meta)
    else:
        _visualize_sample(images, meta)


if __name__ == "__main__":
    main()
