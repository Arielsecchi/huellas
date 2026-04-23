"""Clases del sistema Vucetich y su mapeo a labels numericos.

Vucetich clasifica los dactilogramas en cuatro tipos fundamentales:
  A - Arco           (sin deltas, flujo transversal)
  I - Presilla Interna  (un delta, nucleo desplazado hacia el lado interno)
  E - Presilla Externa  (un delta, nucleo desplazado hacia el lado externo)
  V - Verticilo      (dos deltas, lineas en espiral o circulares)

En la ficha decadactilar los pulgares van numerados (1/2/3/4) y los
otros dedos con letra (A/I/E/V). Para el modelo es indistinto: usamos
indices 0..3.
"""

from enum import IntEnum


class VucetichClass(IntEnum):
    ARCO = 0
    PRESILLA_INTERNA = 1
    PRESILLA_EXTERNA = 2
    VERTICILO = 3


LABEL_TO_SYMBOL = {
    VucetichClass.ARCO: "A",
    VucetichClass.PRESILLA_INTERNA: "I",
    VucetichClass.PRESILLA_EXTERNA: "E",
    VucetichClass.VERTICILO: "V",
}

LABEL_TO_NAME = {
    VucetichClass.ARCO: "Arco",
    VucetichClass.PRESILLA_INTERNA: "Presilla Interna",
    VucetichClass.PRESILLA_EXTERNA: "Presilla Externa",
    VucetichClass.VERTICILO: "Verticilo",
}

SYMBOL_TO_LABEL = {v: k for k, v in LABEL_TO_SYMBOL.items()}

NUM_CLASSES = len(VucetichClass)
