"""DiffAugment: augmentaciones diferenciables aplicadas a real Y fake.

Implementacion siguiendo Zhao et al. 2020, "Differentiable Augmentation for
Data-Efficient GAN Training" (NeurIPS), pensado especificamente para
datasets chicos (<10k imagenes), que es nuestro caso (~6k SOCOFing).

La clave del paper: el D ve siempre versiones augmentadas, tanto de la
imagen real como de la fake. Como las augmentaciones son DIFERENCIABLES,
el gradiente del G atraviesa la augmentacion y el G nunca aprende a
imitar los artefactos de aug; solo aprende a imitar la distribucion real.

Comparado con augmentar solo el real (lo que se haria en un clasificador):
augmentar solo el real le ensena al D que las imagenes augmentadas son
"reales", y el G empieza a generar artefactos de aug. Hay que augmentar
ambos lados con la misma policy en cada step.

Policy elegida para huellas grayscale:
  - "translation"  : shift hasta 1/8 de la imagen (16px en 128x128). Es la
                     aug mas barata y la que mas baja FID en el paper en
                     todos los datasets que probaron.
  - "cutout"       : enmascarar un cuadrado del 50% del area de la imagen
                     en una posicion aleatoria. Empuja al D a mirar la
                     imagen entera, no a memorizar pixeles puntuales.

Descartamos "color" (brightness/saturation/contrast del paper original)
porque las huellas son grayscale y la saturation no aplica; brightness
y contrast estarian, pero el preproceso ya normaliza a [-1, 1] consistente
y agregar variacion de brillo enturbia mas que ayudar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diff_augment(x: torch.Tensor, policy: str) -> torch.Tensor:
    """Aplica la policy de DiffAugment a un batch.

    `policy` es un string con augs separadas por coma, ej. "translation,cutout".
    Si policy es vacio o "none" devuelve el tensor sin tocar.
    """
    if not policy or policy.lower() == "none":
        return x
    for name in policy.split(","):
        name = name.strip()
        if name == "translation":
            x = _rand_translation(x)
        elif name == "cutout":
            x = _rand_cutout(x)
        else:
            raise ValueError(f"DiffAugment policy desconocida: {name!r}. "
                             "Validas: translation, cutout.")
    return x


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Shift aleatorio hasta `ratio` * dimension, con padding cero.

    Para 128x128 y ratio=0.125 los shifts son [-16, +16] px en cada eje.
    Cada imagen del batch recibe un shift independiente.

    Implementacion via gather de coordenadas: padea la imagen con 1 px
    en cada lado, arma una grilla de coords desplazada por el shift y
    selecciona los pixeles correspondientes. Es lo que hace la impl
    oficial del paper y mantiene el grafo diferenciable end-to-end.
    """
    b, c, h, w = x.shape
    shift_x = int(h * ratio + 0.5)
    shift_y = int(w * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(b, 1, 1),
                                  device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(b, 1, 1),
                                  device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(b, dtype=torch.long, device=x.device),
        torch.arange(h, dtype=torch.long, device=x.device),
        torch.arange(w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, h + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return (x_pad.permute(0, 2, 3, 1).contiguous()
            [grid_batch, grid_x, grid_y]
            .permute(0, 3, 1, 2).contiguous())


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Enmascara un cuadrado de `ratio` * lado, con centro aleatorio.

    Para 128x128 y ratio=0.5 el cuadrado es 64x64. El centro puede caer
    sobre el borde (cuadrado parcialmente fuera de la imagen), eso esta
    bien y le da variabilidad al patron de oclusion.
    """
    b, c, h, w = x.shape
    cutout_h = int(h * ratio + 0.5)
    cutout_w = int(w * ratio + 0.5)
    offset_x = torch.randint(0, h + (1 - cutout_h % 2),
                             size=(b, 1, 1), device=x.device)
    offset_y = torch.randint(0, w + (1 - cutout_w % 2),
                             size=(b, 1, 1), device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(b, dtype=torch.long, device=x.device),
        torch.arange(cutout_h, dtype=torch.long, device=x.device),
        torch.arange(cutout_w, dtype=torch.long, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_h // 2, 0, h - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_w // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


def _smoke_test() -> None:
    """Sanity check: shapes preservadas, gradiente fluye end-to-end."""
    torch.manual_seed(0)
    x = torch.randn(8, 1, 128, 128, requires_grad=True)
    y = diff_augment(x, "translation,cutout")
    assert y.shape == x.shape, f"shape mismatch: {x.shape} -> {y.shape}"
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, \
        "el gradiente no fluye a traves de DiffAugment"
    print(f"[ok] diffaug smoke test: {x.shape} -> {y.shape}, grad ok")


if __name__ == "__main__":
    _smoke_test()
