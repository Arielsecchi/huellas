"""Dataset PyTorch: huellas SOCOFing + etiqueta Vucetich.

Lee el preproceso generado por `src.data.preprocess` y `src.data.label_vucetich`:

  data/processed/images.npz     key "images" -> uint8 (N, 128, 128)
  data/processed/metadata.csv   columnas ..., vucetich (int 0..3)

Normaliza las imagenes a float32 en [-1, 1] (range requerido por la salida
Tanh del Generator).

Devuelve tuplas (img, label):
  img   : FloatTensor (1, 128, 128) en [-1, 1]
  label : LongTensor escalar en [0, 4)
"""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGES_PATH = PROJECT_ROOT / "data" / "processed" / "images.npz"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"


class HuellasDataset(Dataset):
    """Dataset en memoria. El npz entero cabe facil en RAM (~90 MB)."""

    def __init__(self,
                 images_path: Path = IMAGES_PATH,
                 metadata_path: Path = METADATA_PATH) -> None:
        if not images_path.exists():
            raise FileNotFoundError(
                f"Falta {images_path}. Correr antes: python -m src.data.preprocess")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Falta {metadata_path}. Correr antes: "
                "python -m src.data.label_vucetich --full")

        images = np.load(images_path)["images"]   # (N, 128, 128) uint8
        with open(metadata_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if "vucetich" not in rows[0]:
            raise RuntimeError(
                "metadata.csv no tiene columna 'vucetich'. Correr antes: "
                "python -m src.data.label_vucetich --full")

        if len(rows) != len(images):
            raise RuntimeError(
                f"Mismatch: {len(images)} imagenes vs {len(rows)} filas de metadata")

        labels = np.array([int(r["vucetich"]) for r in rows], dtype=np.int64)

        # normalizamos a [-1, 1] en float32 y agregamos dim de canal
        imgs_f = (images.astype(np.float32) / 127.5) - 1.0
        self.images = torch.from_numpy(imgs_f).unsqueeze(1)   # (N, 1, 128, 128)
        self.labels = torch.from_numpy(labels)                 # (N,)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
