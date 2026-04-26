"""Hiperparametros del entrenamiento.

Receta "camino B" (defaults actuales): DCGAN base + ajustes para datasets
chicos:

  - **Hinge loss** en vez de BCE: empuja los logits de reales hacia +1 y los
    de fakes hacia -1, con saturacion via ReLU. Mucho mas estable que BCE
    cuando el D se vuelve muy seguro y el gradiente del G colapsa.
  - **Adam betas (0.5, 0.999)**: beta1 bajo evita que el G oscile demasiado
    en las primeras epocas.
  - **TTUR** (Heusel et al. 2017): lr_d > lr_g. Default lr_g=1e-4, lr_d=4e-4
    (ratio 4:1). En setups con SN+hinge sin TTUR el D suele quedarse
    "lento" a partir de cierto punto y el G empieza a generar ruido suave
    sin detalle. Subir lr_d arregla eso sin tener que duplicar pasos del D.
  - Batch size 64: entra comodo en una T4 de Colab Free a 128x128.
  - **DiffAugment** (Zhao et al. 2020) con policy "translation,cutout" para
    multiplicar el dataset efectivo. Imprescindible bajo 10k samples.
  - **EMA del Generator** con decay 0.999: el modelo de inferencia es el
    promedio movil, no el G crudo. Suaviza oscilaciones, mejora calidad
    visible.

Defaults razonables para este dataset (~6k huellas 128x128):

  - 150 epocas: con hinge + cBN + horizontal flip + DiffAugment + EMA, a
    las 50 epocas todavia falta detalle de crestas. 150 dejan margen y en
    T4 tardan ~2.5-3 h (DiffAug agrega ~10-15% por step).
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

    # optimizador (Adam DCGAN + TTUR: lr_d > lr_g)
    lr_g: float = 1e-4
    lr_d: float = 4e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # ruido + arquitectura
    z_dim: int = 100

    # DiffAugment: augs diferenciables aplicadas a real Y fake antes del D.
    # "translation,cutout" es la receta default del paper para datasets chicos.
    # "none" o "" desactiva DiffAugment (vuelve a la receta clasica).
    diffaug_policy: str = "translation,cutout"

    # EMA del Generator: g_ema = decay * g_ema + (1 - decay) * g.
    # 0.999 promedia los ultimos ~1000 steps, que con batch=64 y ~6k samples
    # son ~10 epocas. Es el default de StyleGAN2 y funciona bien aca.
    ema_decay: float = 0.999

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
