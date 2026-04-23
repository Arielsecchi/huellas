"""Hiperparametros del entrenamiento.

Los valores default siguen la receta DCGAN (Radford et al. 2015) con dos
ajustes modernos que estabilizan mucho en datasets chicos:

  - **Hinge loss** en vez de BCE: empuja los logits de reales hacia +1 y los
    de fakes hacia -1, con saturacion via ReLU. Mucho mas estable que BCE
    cuando el D se vuelve muy seguro y el gradiente del G colapsa.
  - **Adam betas (0.5, 0.999)**: beta1 bajo evita que el G oscile demasiado
    en las primeras epocas.
  - Learning rate 2e-4 parejo para G y D.
  - Batch size 64: entra comodo en una T4 de Colab Free a 128x128.

Defaults razonables para este dataset (~6k huellas 128x128):

  - 150 epocas: con hinge + cBN + horizontal flip, a las 50 epocas todavia
    no habia aprendido crestas bien definidas. 150 dejan margen y en T4
    tardan ~2-2.5 h (factible en Colab Free de una sentada).
  - Sampling cada 5 epocas (30 grillas en total, Drive contento).
  - Checkpoint cada 25 epocas.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    # duracion
    epochs: int = 150
    batch_size: int = 64

    # optimizador (Adam DCGAN)
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # ruido + arquitectura
    z_dim: int = 100

    # data augmentation
    # Horizontal flip aleatorio en reales. Como flippear cambia la clase
    # de las presillas (Interna <-> Externa), el loop tambien intercambia
    # las etiquetas cuando flippea. Arcos y Verticilos quedan igual.
    hflip_prob: float = 0.5

    # Balanceo de clases via WeightedRandomSampler.
    # SOCOFing trae A 8.3% / I 33.1% / E 37.1% / V 21.5%: sin balanceo, el G
    # sufre mode-collapse en Arcos (~498 muestras unicas, la cBN + hinge loss
    # tira al G hacia clases mayoritarias). Con balanceo, cada batch ve las 4
    # clases con frecuencia ~uniforme (oversampling de Arcos ~12x).
    # Default True: experimentalmente comprobado en Fase 5 que sin esto el G
    # nunca aprende Arcos.
    balance_classes: bool = True

    # logging / sampling / checkpoints
    sample_every_epochs: int = 5
    ckpt_every_epochs: int = 25
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

    # Resumir entrenamiento desde un checkpoint (ckpt_NNN.pt o generator.pt
    # guardado por _save_checkpoint). Si se define, se cargan G/D/opt_G/opt_D
    # y se continua desde epoch+1. El log CSV se appendea (no se reescribe).
    resume_from: Path | None = None
