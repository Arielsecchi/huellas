"""Exploracion rapida del dataset SOCOFing.

Imprime stats basicas (shape, rango de valores, distribuciones) y guarda
una grilla de 20 muestras aleatorias en outputs/evaluation/socofing_sample_grid.png
para inspeccion visual.

Uso: python -m src.data.explore
"""

import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import socofing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = PROJECT_ROOT / "data" / "raw" / "socofing" / "Real"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
GRID_PATH = OUTPUT_DIR / "socofing_sample_grid.png"

GRID_ROWS = 4
GRID_COLS = 5
SAMPLE_SEED = 42


def _iter_real_paths() -> list[Path]:
    return sorted(REAL_DIR.glob(f"*{socofing.IMAGE_EXT}"))


def _print_shape_stats(paths: list[Path]) -> None:
    shapes = Counter()
    dtypes = Counter()
    mins, maxs = [], []
    for p in paths[:200]:  # muestra: con 200 alcanza para estimar
        arr = np.array(Image.open(p))
        shapes[arr.shape] += 1
        dtypes[str(arr.dtype)] += 1
        mins.append(int(arr.min()))
        maxs.append(int(arr.max()))
    print("[shapes]  (muestra de 200)")
    for shape, n in shapes.most_common():
        print(f"  {shape}: {n}")
    print(f"[dtypes]  {dict(dtypes)}")
    print(f"[rango]   min {min(mins)}-{max(mins)}  max {min(maxs)}-{max(maxs)}")


def _print_metadata_distribution(paths: list[Path]) -> None:
    genders: Counter[str] = Counter()
    hands: Counter[str] = Counter()
    fingers: Counter[str] = Counter()
    subjects: set[int] = set()
    for p in paths:
        m = socofing.parse_filename(p)
        genders[m.gender] += 1
        hands[m.hand] += 1
        fingers[m.finger] += 1
        subjects.add(m.subject_id)
    print(f"[total]   {len(paths)} imagenes")
    print(f"[sujetos] {len(subjects)} unicos (esperado ~600)")
    print(f"[genero]  {dict(genders)}")
    print(f"[mano]    {dict(hands)}")
    print(f"[dedo]    {dict(fingers)}")


def _save_sample_grid(paths: list[Path]) -> None:
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(paths, GRID_ROWS * GRID_COLS)
    fig, axes = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(12, 10))
    for ax, p in zip(axes.flat, sample):
        img = np.array(Image.open(p))
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        m = socofing.parse_filename(p)
        ax.set_title(f"#{m.subject_id} {m.gender} {m.hand[0]} {m.finger}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("SOCOFing — 20 muestras aleatorias (seed=42)", fontsize=12)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRID_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[grilla]  {GRID_PATH}")


def main() -> None:
    paths = _iter_real_paths()
    if not paths:
        raise SystemExit(f"No hay imagenes en {REAL_DIR}. "
                         "Corre antes: python -m src.data.download_socofing")
    _print_shape_stats(paths)
    print()
    _print_metadata_distribution(paths)
    print()
    _save_sample_grid(paths)


if __name__ == "__main__":
    main()
