"""Generacion de muestras y sanity check del Generator entrenado.

Carga `generator.pt` (state_dict guardado por `_save_checkpoint` al final del
training), genera N muestras por clase Vucetich y produce tres artefactos
que sirven para validar la Fase 5:

  1. **Grids por clase** (`samples_<symbol>.png`): grilla 8 cols x ceil(N/8)
     filas a 128x128 reales (1 px = 1 px), una por clase. Sirve para mirar
     en detalle nitidez de crestas, deltas y nucleos.
  2. **Showcase** (`showcase.png`): 4 filas (clases) x 8 cols, primer batch
     de cada clase. Comparacion visual rapida entre clases.
  3. **Sanity check Poincare** (`poincare_confusion.{csv,png}`): aplica el
     mismo clasificador heuristico que etiqueto SOCOFing a las muestras
     generadas y arma una matriz de confusion (clase pedida vs clase
     detectada). Como el clasificador necesita la mano para distinguir
     I de E pero el GAN no condiciona en mano, agrupamos I+E -> "P"
     (presilla) y reportamos solo {A, P, V} en columnas. Filas siguen
     siendo las 4 clases pedidas.

Por que evaluar asi y no FID/IS:
  - Visual + estructural es lo que importa para una app de practica de
    dactiloscopia: necesitamos crestas legibles y conteo correcto de cores.
  - FID requiere descargar pesos InceptionV3 (~100 MB) y no diferencia
    bien entre clases de huellas (entrenado en ImageNet). Lo dejamos como
    Fase 5b si hace falta.
  - El Poincare check usa exactamente el mismo criterio con el que se
    armaron los labels del dataset, asi que mide consistencia interna del
    pipeline (si el GAN aprendio "lo que el classifier ve", funciona).

Uso:
  python -m src.evaluation.sample
  python -m src.evaluation.sample --n-per-class 32
  python -m src.evaluation.sample --weights ruta/a/generator.pt
  python -m src.evaluation.sample --no-confusion        # salta sanity check
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..data.label_vucetich import classify
from ..data.vucetich import LABEL_TO_SYMBOL, NUM_CLASSES, VucetichClass
from ..models.gan import Generator
from ..training.config import PROJECT_ROOT

DEFAULT_WEIGHTS = (PROJECT_ROOT.parent
                   / "huellas_out_final" / "huellas_gan"
                   / "final" / "generator.pt")
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_N_PER_CLASS = 64
DEFAULT_GRID_COLS = 8
DEFAULT_SEED = 1234

# Para la matriz de confusion: el classifier no devuelve I vs E sin mano,
# asi que colapsamos las dos presillas a "P".
CONFUSION_COLS = ["A", "P", "V"]


def _resolve_device(preference: str | None) -> torch.device:
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_generator(weights_path: Path,
                    device: torch.device) -> tuple[Generator, int]:
    """Instancia un Generator y carga el state_dict guardado al final del training.

    El archivo final tiene formato {"generator": state_dict, "z_dim": int,
    "num_classes": int}. Tambien acepta checkpoints intermedios que ademas
    traen "discriminator", "opt_g", "opt_d", "epoch": ignoramos esas keys.
    Devuelve (generator, z_dim) para que el caller use el z_dim guardado en
    el checkpoint en vez de re-derivarlo de la config.
    """
    if not weights_path.exists():
        raise SystemExit(f"No existe el archivo de pesos: {weights_path}")
    bundle = torch.load(weights_path, map_location=device, weights_only=True)
    if "generator" not in bundle:
        raise SystemExit(
            f"Bundle invalido en {weights_path}: falta key 'generator'. "
            f"Keys presentes: {list(bundle.keys())}")
    z_dim = int(bundle.get("z_dim", 100))
    g = Generator(z_dim=z_dim).to(device)
    g.load_state_dict(bundle["generator"])
    g.eval()
    return g, z_dim


@torch.no_grad()
def generate_per_class(generator: Generator,
                       n_per_class: int,
                       device: torch.device,
                       z_dim: int,
                       seed: int) -> dict[VucetichClass, np.ndarray]:
    """Genera n_per_class muestras por cada clase Vucetich.

    Devuelve un dict {clase: array uint8 [N, 128, 128]} listo para guardar
    o pasar al classifier (que espera uint8).
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out: dict[VucetichClass, np.ndarray] = {}
    for klass in VucetichClass:
        z = torch.randn(n_per_class, z_dim, generator=gen).to(device)
        labels = torch.full((n_per_class,), int(klass), dtype=torch.long,
                            device=device)
        fake = generator(z, labels).cpu().numpy()  # (N, 1, 128, 128) en [-1, 1]
        # Tanh -> [0, 1] -> uint8. clip por seguridad numerica.
        fake = ((fake[:, 0] * 0.5 + 0.5).clip(0.0, 1.0) * 255.0
                ).round().astype(np.uint8)
        out[klass] = fake
    return out


def save_per_class_grid(samples: np.ndarray,
                        klass: VucetichClass,
                        out_path: Path,
                        cols: int) -> None:
    """Tile-grid 1px=1px de N muestras 128x128 de una sola clase.

    Evitamos matplotlib aca: pegamos las imagenes en un canvas grande y
    guardamos PNG con plt.imsave (lossless, sin reescalado), asi se conserva
    la nitidez exacta para inspeccion de crestas.
    """
    n = samples.shape[0]
    rows = (n + cols - 1) // cols
    h, w = samples.shape[1], samples.shape[2]
    canvas = np.full((rows * h, cols * w), 255, dtype=np.uint8)
    for idx in range(n):
        r, c = divmod(idx, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = samples[idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, canvas, cmap="gray", vmin=0, vmax=255)


def save_showcase(samples_by_class: dict[VucetichClass, np.ndarray],
                  out_path: Path,
                  cols: int) -> None:
    """Grid 4 filas (clases) x cols cols con las primeras `cols` muestras."""
    h, w = next(iter(samples_by_class.values())).shape[1:]
    fig, axes = plt.subplots(NUM_CLASSES, cols,
                             figsize=(cols * 1.4, NUM_CLASSES * 1.4))
    for i, klass in enumerate(VucetichClass):
        for j in range(cols):
            ax = axes[i, j] if NUM_CLASSES > 1 else axes[j]
            ax.imshow(samples_by_class[klass][j], cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(LABEL_TO_SYMBOL[klass], fontsize=12)
    fig.suptitle("Showcase Generator final — filas=clase pedida, "
                 f"cols=primeras {cols} muestras", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _detected_bucket(klass: VucetichClass) -> str:
    """Mapea la clase devuelta por classify() a {A, P, V}.

    classify() devuelve I o E segun la mano, pero como el GAN no condiciona
    en mano y al sanity check le pasamos hand fijo, agrupamos I+E -> P.
    """
    if klass == VucetichClass.ARCO:
        return "A"
    if klass == VucetichClass.VERTICILO:
        return "V"
    return "P"


def poincare_confusion(samples_by_class: dict[VucetichClass, np.ndarray]
                       ) -> np.ndarray:
    """Devuelve matriz (4, 3) con conteos clase-pedida vs bucket detectado.

    Filas: A, I, E, V (orden de VucetichClass).
    Cols : A, P, V    (CONFUSION_COLS).
    """
    matrix = np.zeros((NUM_CLASSES, len(CONFUSION_COLS)), dtype=np.int32)
    col_idx = {sym: i for i, sym in enumerate(CONFUSION_COLS)}
    for row_idx, klass in enumerate(VucetichClass):
        for img in samples_by_class[klass]:
            # hand fijo en "Right": como agrupamos I+E -> P, no afecta el bucket.
            result = classify(img, hand="Right")
            matrix[row_idx, col_idx[_detected_bucket(result.klass)]] += 1
    return matrix


def save_confusion(matrix: np.ndarray,
                   csv_path: Path,
                   png_path: Path) -> None:
    """Vuelca la matriz a CSV y a un heatmap PNG con porcentajes por fila."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_symbols = [LABEL_TO_SYMBOL[c] for c in VucetichClass]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pedida \\ detectada", *CONFUSION_COLS, "total"])
        for sym, row in zip(row_symbols, matrix):
            writer.writerow([sym, *row.tolist(), int(row.sum())])

    # heatmap con porcentajes por fila (hit-rate condicional).
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    pct = matrix / row_totals * 100.0

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(pct, cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(CONFUSION_COLS)))
    ax.set_xticklabels(CONFUSION_COLS)
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(row_symbols)
    ax.set_xlabel("Detectada (Poincare)")
    ax.set_ylabel("Pedida al G")
    for i in range(NUM_CLASSES):
        for j in range(len(CONFUSION_COLS)):
            ax.text(j, i, f"{int(matrix[i, j])}\n{pct[i, j]:.0f}%",
                    ha="center", va="center",
                    color="white" if pct[i, j] < 50 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="% por fila")
    fig.suptitle("Sanity check Poincare — pedida vs detectada", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help=f"Path al state_dict del Generator "
                             f"(default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Directorio de salida (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS,
                        help=f"Muestras a generar por clase (default: {DEFAULT_N_PER_CLASS})")
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS,
                        help=f"Columnas en los grids por clase (default: {DEFAULT_GRID_COLS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Semilla del ruido z (default: {DEFAULT_SEED})")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu | cuda (default: auto)")
    parser.add_argument("--no-confusion", action="store_true",
                        help="Salta el sanity check Poincare (mas rapido)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)

    print(f"[device] {device}")
    print(f"[weights] {args.weights}")
    print(f"[out-dir] {args.out_dir}")

    generator, z_dim = _load_generator(args.weights, device)
    print(f"[ok] generator cargado (z_dim={z_dim})")

    samples = generate_per_class(generator, args.n_per_class, device,
                                 z_dim, args.seed)
    print(f"[ok] generadas {args.n_per_class} muestras por clase "
          f"({args.n_per_class * NUM_CLASSES} en total)")

    for klass in VucetichClass:
        sym = LABEL_TO_SYMBOL[klass]
        out_path = args.out_dir / f"samples_{sym}.png"
        save_per_class_grid(samples[klass], klass, out_path, args.grid_cols)
        print(f"[grid] {out_path}")

    showcase_path = args.out_dir / "showcase.png"
    save_showcase(samples, showcase_path, cols=DEFAULT_GRID_COLS)
    print(f"[showcase] {showcase_path}")

    if args.no_confusion:
        print("[skip] sanity check Poincare (--no-confusion)")
        return

    print("[poincare] clasificando muestras generadas...")
    matrix = poincare_confusion(samples)
    csv_path = args.out_dir / "poincare_confusion.csv"
    png_path = args.out_dir / "poincare_confusion.png"
    save_confusion(matrix, csv_path, png_path)
    print(f"[csv] {csv_path}")
    print(f"[heatmap] {png_path}")

    # resumen por consola: hit-rate de cada clase pedida hacia su bucket esperado
    print("\n[resumen] hit-rate por clase pedida (clase -> bucket esperado):")
    expected_bucket = {VucetichClass.ARCO: "A",
                       VucetichClass.PRESILLA_INTERNA: "P",
                       VucetichClass.PRESILLA_EXTERNA: "P",
                       VucetichClass.VERTICILO: "V"}
    col_idx = {sym: i for i, sym in enumerate(CONFUSION_COLS)}
    for row_idx, klass in enumerate(VucetichClass):
        sym = LABEL_TO_SYMBOL[klass]
        bucket = expected_bucket[klass]
        hits = matrix[row_idx, col_idx[bucket]]
        total = matrix[row_idx].sum()
        pct = (100.0 * hits / total) if total else 0.0
        print(f"  {sym} -> {bucket}: {hits}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
