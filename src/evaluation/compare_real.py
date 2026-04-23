"""Comparativa real vs sintetica + baseline del classifier Poincare.

Sirve para responder dos preguntas de la Fase 5:

  1. **Visual**: ¿las muestras del Generator se parecen a las reales del dataset
     dentro de cada clase Vucetich? Generamos un grid 4 filas (clases) x
     2 grupos (REAL | SYNTH) de 8 muestras cada uno = 16 cols por fila.
  2. **Baseline del classifier**: ¿la heuristica Poincare clasifica bien las
     huellas REALES, o tiene un sesgo conocido? Corremos el mismo sanity
     check de `sample.py` sobre N reales por clase y reportamos hit-rate.
     Si el baseline real para una clase es bajo, el bajo hit-rate del GAN
     en esa clase NO es necesariamente culpa del GAN: puede ser ruido del
     classifier.

Lectura conjunta con `sample.py`:
  - GAN A->A 0% pero baseline real A->A tambien bajo  -> classifier sesgado.
  - GAN A->A 0% y baseline real A->A alto             -> mode collapse o
                                                          falla real del G
                                                          en la clase A.

Uso:
  python -m src.evaluation.compare_real
  python -m src.evaluation.compare_real --n-per-class 32 --seed 7
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.label_vucetich import classify
from ..data.vucetich import LABEL_TO_SYMBOL, NUM_CLASSES, SYMBOL_TO_LABEL, VucetichClass
from ..models.gan import Generator
from ..training.config import PROJECT_ROOT
from .sample import (CONFUSION_COLS, DEFAULT_WEIGHTS, _detected_bucket,
                     _load_generator, _resolve_device, generate_per_class)

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "processed" / "images.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
DEFAULT_N_PER_CLASS = 8
DEFAULT_SEED = 1234


def _load_real_dataset(images_path: Path, metadata_path: Path
                       ) -> tuple[np.ndarray, list[dict]]:
    if not images_path.exists() or not metadata_path.exists():
        raise SystemExit(
            f"Faltan datos procesados:\n  {images_path}\n  {metadata_path}\n"
            "Corre antes: python -m src.data.preprocess && "
            "python -m src.data.label_vucetich --full")
    images = np.load(images_path)["images"]
    with open(metadata_path, encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    return images, meta


def _sample_real_per_class(images: np.ndarray,
                           meta: list[dict],
                           n_per_class: int,
                           seed: int
                           ) -> dict[VucetichClass, tuple[np.ndarray, list[str]]]:
    """Devuelve {clase: (imgs uint8 [N,128,128], hands [N])}.

    Mantenemos `hand` porque el classifier Poincare lo necesita para
    distinguir I de E (al revés que con muestras del GAN, en reales la
    mano viene del metadata y vale la pena pasarla bien).
    """
    rng = np.random.default_rng(seed)
    out: dict[VucetichClass, tuple[np.ndarray, list[str]]] = {}
    for klass in VucetichClass:
        sym = LABEL_TO_SYMBOL[klass]
        idxs = [i for i, row in enumerate(meta) if row["vucetich_symbol"] == sym]
        if not idxs:
            raise SystemExit(f"No hay muestras reales para clase {sym}")
        chosen = rng.choice(idxs, size=min(n_per_class, len(idxs)),
                            replace=False)
        imgs = images[chosen]
        hands = [meta[i]["hand"] for i in chosen]
        out[klass] = (imgs, hands)
    return out


def real_baseline_confusion(samples_by_class: dict[VucetichClass,
                                                    tuple[np.ndarray, list[str]]]
                            ) -> np.ndarray:
    """Misma matriz (4, 3) que en sample.py pero usando muestras REALES."""
    matrix = np.zeros((NUM_CLASSES, len(CONFUSION_COLS)), dtype=np.int32)
    col_idx = {sym: i for i, sym in enumerate(CONFUSION_COLS)}
    for row_idx, klass in enumerate(VucetichClass):
        imgs, hands = samples_by_class[klass]
        for img, hand in zip(imgs, hands):
            result = classify(img, hand=hand)
            matrix[row_idx, col_idx[_detected_bucket(result.klass)]] += 1
    return matrix


def save_side_by_side(real_by_class: dict[VucetichClass,
                                           tuple[np.ndarray, list[str]]],
                      synth_by_class: dict[VucetichClass, np.ndarray],
                      out_path: Path,
                      n_per_side: int) -> None:
    """Grid 4 filas x (2*n_per_side + 1) cols: REAL | sep | SYNTH por clase.

    La columna del medio queda en blanco para separar visualmente los
    dos bloques.
    """
    sep_col = 1
    cols = 2 * n_per_side + sep_col
    fig, axes = plt.subplots(NUM_CLASSES, cols,
                             figsize=(cols * 1.4, NUM_CLASSES * 1.6))

    for i, klass in enumerate(VucetichClass):
        real_imgs, _ = real_by_class[klass]
        synth_imgs = synth_by_class[klass]
        sym = LABEL_TO_SYMBOL[klass]

        # bloque REAL (cols 0..n_per_side-1)
        for j in range(n_per_side):
            ax = axes[i, j]
            if j < real_imgs.shape[0]:
                ax.imshow(real_imgs[j], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(sym, fontsize=12)
            if i == 0 and j == n_per_side // 2:
                ax.set_title("REAL", fontsize=11)

        # columna separadora
        ax_sep = axes[i, n_per_side]
        ax_sep.axis("off")

        # bloque SYNTH (cols n_per_side+1..)
        for j in range(n_per_side):
            ax = axes[i, n_per_side + 1 + j]
            if j < synth_imgs.shape[0]:
                ax.imshow(synth_imgs[j], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0 and j == n_per_side // 2:
                ax.set_title("SYNTH", fontsize=11)

    fig.suptitle("Real vs sintetica — filas=clase Vucetich, "
                 f"{n_per_side} muestras por bloque", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_baseline_confusion(matrix: np.ndarray,
                            csv_path: Path,
                            png_path: Path) -> None:
    """Igual al de sample.py pero rotulado como baseline real."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_symbols = [LABEL_TO_SYMBOL[c] for c in VucetichClass]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pedida \\ detectada", *CONFUSION_COLS, "total"])
        for sym, row in zip(row_symbols, matrix):
            writer.writerow([sym, *row.tolist(), int(row.sum())])

    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    pct = matrix / row_totals * 100.0

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(pct, cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(CONFUSION_COLS)))
    ax.set_xticklabels(CONFUSION_COLS)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(row_symbols)
    ax.set_xlabel("Detectada (Poincare)")
    ax.set_ylabel("Etiqueta del dataset")
    for i in range(NUM_CLASSES):
        for j in range(len(CONFUSION_COLS)):
            ax.text(j, i, f"{int(matrix[i, j])}\n{pct[i, j]:.0f}%",
                    ha="center", va="center",
                    color="white" if pct[i, j] < 50 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="% por fila")
    fig.suptitle("Baseline Poincare sobre REALES — sanity del classifier",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_hit_rates(matrix: np.ndarray, label: str) -> None:
    expected_bucket = {VucetichClass.ARCO: "A",
                       VucetichClass.PRESILLA_INTERNA: "P",
                       VucetichClass.PRESILLA_EXTERNA: "P",
                       VucetichClass.VERTICILO: "V"}
    col_idx = {sym: i for i, sym in enumerate(CONFUSION_COLS)}
    print(f"\n[{label}] hit-rate por clase pedida (clase -> bucket esperado):")
    for row_idx, klass in enumerate(VucetichClass):
        sym = LABEL_TO_SYMBOL[klass]
        bucket = expected_bucket[klass]
        hits = matrix[row_idx, col_idx[bucket]]
        total = matrix[row_idx].sum()
        pct = (100.0 * hits / total) if total else 0.0
        print(f"  {sym} -> {bucket}: {hits}/{total} ({pct:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                        help=f"Muestras por clase usadas para grid Y baseline "
                             f"(default: {DEFAULT_N_PER_CLASS})")
    parser.add_argument("--baseline-n", type=int, default=200,
                        help="Muestras reales por clase para el baseline "
                             "Poincare (default: 200)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print(f"[device] {device}")
    print(f"[weights] {args.weights}")
    print(f"[images]  {args.images}")
    print(f"[out-dir] {args.out_dir}")

    images, meta = _load_real_dataset(args.images, args.metadata)
    print(f"[ok] dataset real cargado: {len(images)} imagenes")

    generator, z_dim = _load_generator(args.weights, device)
    print(f"[ok] generator cargado (z_dim={z_dim})")

    # 1) reales para grid lado a lado
    real_for_grid = _sample_real_per_class(images, meta,
                                           args.n_per_class, args.seed)
    # 2) sinteticas para grid lado a lado (mismo seed que sample.py default)
    synth_for_grid = generate_per_class(generator, args.n_per_class, device,
                                        z_dim, args.seed)

    grid_path = args.out_dir / "real_vs_synth.png"
    save_side_by_side(real_for_grid, synth_for_grid, grid_path,
                      n_per_side=args.n_per_class)
    print(f"[grid] {grid_path}")

    # 3) baseline Poincare con N reales mas grandes para ruido bajo
    real_for_baseline = _sample_real_per_class(images, meta,
                                               args.baseline_n,
                                               args.seed + 1)
    print(f"[poincare] clasificando {args.baseline_n} reales por clase...")
    baseline_matrix = real_baseline_confusion(real_for_baseline)
    csv_path = args.out_dir / "poincare_baseline_real.csv"
    png_path = args.out_dir / "poincare_baseline_real.png"
    save_baseline_confusion(baseline_matrix, csv_path, png_path)
    print(f"[csv] {csv_path}")
    print(f"[heatmap] {png_path}")
    _print_hit_rates(baseline_matrix, "baseline real")


if __name__ == "__main__":
    main()
