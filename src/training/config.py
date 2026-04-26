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
    # tira al G hacia clases mayoritarias).
    #
    # Historial de Fase 5:
    #   v1 (balance_classes=False, strength=irrelevante): 3/4 clases OK pero
    #     A 0% (mode-collapse limpio en la minoria). Recomendado para shipping.
    #   v2 (balance_classes=True, balance_strength=1.0 -> pesos 1/freq, Arcos
    #     oversampleadas 4.4x): empeoro. Diferenciacion en ep.50 pero colapso
    #     tardio total para ep.150 (las 4 clases generan el mismo patron).
    #   v3 (deuda tecnica, NO entrenado aun): probar balance_classes=True con
    #     balance_strength=0.5 (pesos 1/sqrt(freq), Arcos ~2.1x) + epochs=100
    #     en vez de 150. Idea: el oversampling agresivo de v2 desestabilizo
    #     el cBN; medio-paso puede preservar diversidad inter-clase sin matar
    #     la convergencia.
    balance_classes: bool = False

    # Exponente del balanceo: pesos por muestra = 1 / (freq_clase ** strength).
    #   strength=0.0 -> sin oversampling (equivale a balance_classes=False).
    #   strength=0.5 -> 1/sqrt(freq), oversampling moderado (receta v3).
    #   strength=1.0 -> 1/freq, oversampling fuerte (receta v2, colapso tardio).
    balance_strength: float = 1.0

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
