"""Grilla visual: 5 muestras por cada clase Vucetich asignada.

Uso: python -m src.data.viz_labels
"""

import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .vucetich import LABEL_TO_NAME, LABEL_TO_SYMBOL, VucetichClass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "images.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "vucetich_labels_by_class.png"

COLS = 6
SEED = 7


def main() -> None:
    images = np.load(IMAGES_PATH)["images"]
    with open(METADATA_PATH, encoding="utf-8") as f:
        meta = list(csv.DictReader(f))

    by_class: dict[VucetichClass, list[int]] = {c: [] for c in VucetichClass}
    for row in meta:
        c = VucetichClass(int(row["vucetich"]))
        by_class[c].append(int(row["index"]))

    rng = random.Random(SEED)
    fig, axes = plt.subplots(len(VucetichClass), COLS,
                             figsize=(2 * COLS, 2 * len(VucetichClass)))
    for r, c in enumerate(VucetichClass):
        idxs = by_class[c]
        sample = rng.sample(idxs, min(COLS, len(idxs)))
        # label a la izquierda de la fila
        axes[r, 0].set_ylabel(
            f"{LABEL_TO_SYMBOL[c]} — {LABEL_TO_NAME[c]}\n(n={len(idxs)})",
            fontsize=10, rotation=0, labelpad=60, va="center",
        )
        for ax, idx in zip(axes[r], sample):
            img = images[idx]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            m = meta[idx]
            ax.set_title(f"#{idx} {m['hand'][0]} {m['finger']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Muestras por clase segun heuristica Poincare", fontsize=12)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
