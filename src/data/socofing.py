"""Metadata del dataset SOCOFing (Sokoto Coventry Fingerprint Dataset).

Contenido:
  - 6000 huellas originales en Real/ -> estas son las que nos interesan
  - versiones alteradas (Easy/Medium/Hard) para research de anti-spoofing:
    las ignoramos, no aportan para entrenar la GAN
  - 600 sujetos africanos adultos, 10 dedos por sujeto
  - BMP 8-bit gris, aprox 96x103 px (tamanio chico, vamos a resamplear)
  - sin clase Vucetich: hay que etiquetar aparte (fase de etiquetado)

Convencion de nombre de archivo:
  "{subject_id}__{gender}_{hand}_{finger}_finger.BMP"
  ejemplo: "1__M_Left_index_finger.BMP"

Fuente oficial: https://www.kaggle.com/datasets/ruizgara/socofing
Paper:          Shehu et al. 2018 (arXiv:1807.10609)
Licencia:       libre para uso academico (ver README del dataset)
"""

from dataclasses import dataclass
from pathlib import Path

KAGGLE_DATASET_SLUG = "ruizgara/socofing"

# subcarpetas dentro del zip descomprimido
REAL_SUBDIR = "SOCOFing/Real"
ALTERED_SUBDIR = "SOCOFing/Altered"

EXPECTED_REAL_COUNT = 6000
IMAGE_EXT = ".BMP"


@dataclass(frozen=True)
class FingerprintMeta:
    """Campos parseados del filename SOCOFing."""

    subject_id: int
    gender: str  # "M" | "F"
    hand: str    # "Left" | "Right"
    finger: str  # "thumb" | "index" | "middle" | "ring" | "little"
    path: Path

    @property
    def filename(self) -> str:
        return self.path.name


def parse_filename(path: Path) -> FingerprintMeta:
    """Devuelve la metadata embebida en el filename SOCOFing.

    Formato esperado: "{id}__{gender}_{hand}_{finger}_finger.BMP"
    """
    stem = path.stem  # sin extension
    subject_part, rest = stem.split("__", 1)
    gender, hand, finger, _finger_word = rest.split("_")
    return FingerprintMeta(
        subject_id=int(subject_part),
        gender=gender,
        hand=hand,
        finger=finger,
        path=path,
    )
