"""Hiperparametros del entrenamiento.

Los valores default siguen la receta DCGAN (Radford et al. 2015):

  - Adam con betas (0.5, 0.999): beta1 bajo evita que el G oscile
    demasiado en las primeras epocas.
  - Learning rate 2e-4 parejo para G y D.
  - Batch size 64: entra comodo en una T4 de Colab Free a 128x128.

Otros defaults razonables para este dataset (~6k huellas 128x128):

  - 50 epocas: ~4700 steps con batch 64. En T4 son ~30-40 minutos.
  - Sampling cada 2 epocas para ver evolucion sin saturar Drive.
  - Checkpoint cada 10 epocas.
  - Label smoothing 0.9 en reales (truco clasico para estabilizar el D).
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    # duracion
    epochs: int = 50
    batch_size: int = 64

    # optimizador (Adam DCGAN)
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # ruido + arquitectura
    z_dim: int = 100

    # trucos de estabilidad
    real_label_smooth: float = 0.9   # 1.0 -> sin smoothing
    fake_label: float = 0.0

    # logging / sampling / checkpoints
    sample_every_epochs: int = 2
    ckpt_every_epochs: int = 10
    samples_per_class: int = 4       # grilla de sampling: 4 clases x N = 16 imgs
    sample_seed: int = 42

    # data loading
    num_workers: int = 2
    pin_memory: bool = True

    # rutas (relativas al root del proyecto; pueden sobreescribirse en Colab)
    checkpoints_dir: Path = PROJECT_ROOT / "models" / "checkpoints"
    final_model_path: Path = PROJECT_ROOT / "models" / "final" / "generator.pt"
    samples_dir: Path = PROJECT_ROOT / "outputs" / "training_samples"
    log_path: Path = PROJECT_ROOT / "outputs" / "training_samples" / "train_log.csv"

    # device: "cuda" si hay GPU, "cpu" si no. None = auto-detect.
    device: str | None = None

    # semilla global
    seed: int = 42

    # corte temprano para smoke tests (None = entrenar completo)
    max_steps: int | None = None
