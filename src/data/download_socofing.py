"""Descarga el dataset SOCOFing desde Kaggle a data/raw/socofing/.

Uso (desde la raiz del repo, con el venv activo):
    python -m src.data.download_socofing

Pre-requisito: tener configurado el token API de Kaggle. Ver README,
seccion "Datasets", para como generar y ubicar kaggle.json.
"""

import argparse
import shutil
import sys
from pathlib import Path

from . import socofing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = PROJECT_ROOT / "data" / "raw" / "socofing"


def download(dest_dir: Path = DEFAULT_DEST, force: bool = False) -> Path:
    """Baja SOCOFing desde Kaggle y lo descomprime en dest_dir.

    El cliente `kaggle` se importa adentro porque requiere kaggle.json
    presente para instanciarse: si lo importamos arriba, todo el modulo
    falla al cargarse en maquinas que todavia no configuraron el token.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    real_dir = dest_dir / "Real"
    if real_dir.exists() and not force:
        count = sum(1 for _ in real_dir.glob(f"*{socofing.IMAGE_EXT}"))
        if count == socofing.EXPECTED_REAL_COUNT:
            print(f"[ok] SOCOFing ya presente en {real_dir} ({count} imagenes). "
                  f"Pasar --force para re-descargar.")
            return dest_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except OSError as e:
        # kaggle levanta OSError si no encuentra kaggle.json en ~/.kaggle/
        print(f"[error] No se encontro el token de Kaggle: {e}", file=sys.stderr)
        print("Ver README > Datasets para configurar kaggle.json.", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"[..] Descargando {socofing.KAGGLE_DATASET_SLUG} en {dest_dir} ...")
    api.dataset_download_files(
        socofing.KAGGLE_DATASET_SLUG,
        path=str(dest_dir),
        unzip=True,
        quiet=False,
    )

    # el zip se descomprime en dest_dir/SOCOFing/{Real,Altered}
    # aplanamos para que queden en dest_dir/{Real,Altered}
    nested = dest_dir / "SOCOFing"
    if nested.exists():
        for item in nested.iterdir():
            target = dest_dir / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
        nested.rmdir()

    count = sum(1 for _ in (dest_dir / "Real").glob(f"*{socofing.IMAGE_EXT}"))
    if count != socofing.EXPECTED_REAL_COUNT:
        print(f"[warn] Se esperaban {socofing.EXPECTED_REAL_COUNT} imagenes en Real/, "
              f"se encontraron {count}.", file=sys.stderr)
    else:
        print(f"[ok] Descarga completa: {count} imagenes en {dest_dir / 'Real'}")

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"destino (default: {DEFAULT_DEST})")
    parser.add_argument("--force", action="store_true",
                        help="re-descargar aunque ya este presente")
    args = parser.parse_args()
    download(args.dest, force=args.force)


if __name__ == "__main__":
    main()
