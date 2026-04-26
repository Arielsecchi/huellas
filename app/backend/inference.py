"""Singleton del Generator + render a PNG bytes.

Carga `generator.pt` UNA vez al startup (ver `main.py`). Cada request hace
forward en `torch.no_grad()` -> Tanh -> [0,1] -> uint8 -> PIL PNG.

En CPU una sample tarda ~30-50 ms a 128x128, suficiente para una app de
practica. Si en algun momento hace falta lo migramos a CUDA o batcheamos
varias en paralelo.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.vucetich import NUM_CLASSES, VucetichClass
from src.models.gan import Generator

# Path por defecto: el v1 shipping. Override via HUELLAS_WEIGHTS env var en main.py.
# PROJECT_ROOT apunta a huellas-gan/. Los outputs del entrenamiento estan
# UN nivel arriba (huellas_out_final/ es hermano de huellas-gan/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = (PROJECT_ROOT.parent
                   / "huellas_out_final" / "huellas_gan"
                   / "final" / "generator.pt")


class GANSampler:
    """Wrapper en memoria del Generator entrenado.

    Uso tipico:
        sampler = GANSampler.load(weights_path)
        png_bytes = sampler.sample_png(VucetichClass.PRESILLA_INTERNA)
    """

    def __init__(self, generator: Generator, z_dim: int,
                 device: torch.device) -> None:
        self.generator = generator
        self.z_dim = z_dim
        self.device = device

    @classmethod
    def load(cls, weights_path: Path,
             device: torch.device | None = None) -> "GANSampler":
        if not weights_path.exists():
            raise FileNotFoundError(
                f"No existe el archivo de pesos: {weights_path}. "
                f"Override con APP_WEIGHTS env var.")
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(weights_path, map_location=device,
                            weights_only=True)
        if "generator" not in bundle:
            raise RuntimeError(
                f"Bundle invalido en {weights_path}: falta 'generator'.")
        z_dim = int(bundle.get("z_dim", 100))
        g = Generator(z_dim=z_dim).to(device)
        g.load_state_dict(bundle["generator"])
        g.eval()
        return cls(g, z_dim, device)

    @torch.no_grad()
    def sample_uint8(self, klass: VucetichClass) -> np.ndarray:
        """Genera UNA sample (128, 128) uint8 condicional en `klass`."""
        z = torch.randn(1, self.z_dim, device=self.device)
        labels = torch.tensor([int(klass)], dtype=torch.long,
                              device=self.device)
        fake = self.generator(z, labels).cpu().numpy()[0, 0]
        # de [-1, 1] a [0, 255] uint8
        img = ((fake * 0.5 + 0.5).clip(0.0, 1.0) * 255.0
               ).round().astype(np.uint8)
        return img

    def sample_png(self, klass: VucetichClass) -> bytes:
        """Genera una sample y la devuelve como PNG bytes (sin recompresion)."""
        img = self.sample_uint8(klass)
        return _uint8_to_png_bytes(img)


def _uint8_to_png_bytes(img: np.ndarray) -> bytes:
    """Encodea un array uint8 (H, W) o (H, W, 3) como PNG en memoria."""
    if img.ndim == 2:
        pil = Image.fromarray(img, mode="L")
    elif img.ndim == 3 and img.shape[2] == 3:
        pil = Image.fromarray(img, mode="RGB")
    else:
        raise ValueError(f"shape de imagen no soportada: {img.shape}")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
