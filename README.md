# Huellas-GAN

Generador de huellas dactilares sintéticas con una **GAN condicional** + app web local de práctica para clasificación bajo el sistema **Vucetich** (Arco / Presilla Interna / Presilla Externa / Verticilo).

## Requisitos

- Windows / macOS / Linux
- **Python 3.12.x**, instalado desde [python.org](https://www.python.org/downloads/) — **no** desde Microsoft Store (la versión Store rompe `venv` y algunas instalaciones de PyTorch)
- Cuenta de Google (el entrenamiento corre en Google Colab, ya que esta máquina no tiene GPU NVIDIA)
- ~5 GB libres para dataset + modelos + venv

## Setup desde cero

### 1. Crear el entorno virtual

Desde el directorio `huellas-gan/`:

```bash
python -m venv venv
```

Activarlo:

| Shell | Comando |
|---|---|
| Windows — Git Bash | `source venv/Scripts/activate` |
| Windows — PowerShell | `venv\Scripts\Activate.ps1` |
| Windows — CMD | `venv\Scripts\activate.bat` |
| macOS / Linux | `source venv/bin/activate` |

Sabés que está activo cuando ves `(venv)` adelante del prompt.

### 2. Instalar PyTorch (versión CPU para entrenar local *no* — sólo inferencia)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> **Por qué CPU:** esta máquina no tiene GPU NVIDIA, así que la versión con CUDA sería un peso muerto. El entrenamiento real corre en Colab (donde PyTorch ya viene con CUDA pre-instalado). La versión CPU local nos sirve para correr la app web y para hacer pruebas rápidas del código de la GAN.

### 3. Instalar el resto de las dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar que todo cargó OK

```bash
python -c "import torch, torchvision, fastapi, cv2; print('todo OK, torch', torch.__version__)"
```

### 5. Reactivar el entorno en una sesión nueva

Sólo el comando de activación (paso 1, segunda parte). No reinstalar nada.

## Estructura del proyecto

```
huellas-gan/
├── data/                    # datasets (no van al repo)
│   ├── raw/                 # tal cual descargados
│   └── processed/           # listos para entrenar
├── models/
│   ├── checkpoints/         # pesos durante entrenamiento (no van al repo)
│   └── final/               # modelo exportado (no va al repo)
├── outputs/
│   ├── generated/           # huellas sintéticas generadas
│   ├── training_samples/    # samples visuales del entrenamiento
│   └── evaluation/          # grillas y métricas
├── src/
│   ├── data/                # loaders, preproceso, etiquetado Vucetich
│   ├── models/              # arquitectura de la GAN
│   ├── training/            # loop de entrenamiento
│   └── utils/
├── app/                     # app web de práctica
│   ├── backend/             # FastAPI
│   └── frontend/            # vanilla HTML/JS/CSS
├── notebooks/               # exploración + evaluación
├── requirements.txt
└── README.md
```

## Datasets

### SOCOFing (Sokoto Coventry Fingerprint Dataset)

6000 huellas reales, libres, en Kaggle. No trae clase Vucetich — la etiquetamos nosotros en el paso siguiente.

**Conseguir el token de Kaggle (una sola vez):**

1. Crearse cuenta en [kaggle.com](https://www.kaggle.com/) (gratis).
2. Ir a tu avatar → **Settings** → bajar hasta **API** → click **Create New Token**. Se baja un archivo `kaggle.json`.
3. Moverlo a `C:\Users\<tu-usuario>\.kaggle\kaggle.json` (crear la carpeta si no existe).

**Descargar el dataset:**

```bash
python -m src.data.download_socofing
```

Baja ~300 MB a `data/raw/socofing/`. Si ya está presente no re-descarga (pasar `--force` para forzar).

## Arquitectura de la GAN

cDCGAN condicional (128x128 grayscale, 4 clases Vucetich). Detalles en el docstring de [src/models/gan.py](src/models/gan.py). Smoke test rápido:

```bash
python -m src.models.gan
```

Debería imprimir shapes del forward pass y `[ok] smoke test paso`.

## Entrenamiento

El entrenamiento corre en **Google Colab Free (T4)** porque esta máquina no tiene GPU NVIDIA. En CPU el loop arranca pero una época tarda varios minutos — sólo sirve para smoke tests.

### Smoke test local (CPU)

```bash
python -m src.training.train --max-steps 2 --epochs 1 --batch-size 8 --num-workers 0
```

Corre dos iteraciones, guarda un checkpoint, una grilla de samples y el modelo final. Sirve para chequear que todo enchufa antes de ir a Colab.

### Entrenamiento completo (Colab)

1. Tené tu `kaggle.json` a mano (cómo conseguirlo: sección [Datasets](#datasets) abajo).
2. Abrí el notebook directamente en Colab: [train_colab.ipynb en Colab](https://colab.research.google.com/github/Arielsecchi/huellas/blob/main/huellas-gan/notebooks/train_colab.ipynb).
3. *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
4. Correr las celdas en orden.

El notebook:

- verifica la GPU,
- **clona el repo desde GitHub** (siempre la última versión),
- instala dependencias,
- te pide que **subas `kaggle.json` una vez** (con el botón "Elegir archivos"),
- **regenera el dataset dentro de la VM** (descarga SOCOFing + preproceso + etiquetado Vucetich, ~10 min),
- corre un smoke test de 2 iteraciones,
- entrena 50 épocas (batch 64, ~30-40 min en T4),
- muestra la grilla final de samples y la curva de pérdida,
- **arma un ZIP con el modelo + samples y te lo baja a tu PC**.

En tu PC vas a recibir `huellas_out.zip` con:

- `final/generator.pt` — el modelo entrenado (lo usa la app de Fase 6).
- `training_samples/` — grillas por época + `train_log.csv`.

La configuración vive en [src/training/config.py](src/training/config.py) — si querés cambiar épocas, lr, etc., editá ahí o pasá flags por CLI.

## Estado actual

**Fase 4 — Entrenamiento:** código listo. Falta correrlo en Colab y volcar los resultados.

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Diagnóstico de hardware | ✅ |
| 1 | Setup del proyecto | ✅ |
| 2 | Dataset + preproceso + etiquetado Vucetich | ✅ |
| 3 | Arquitectura de la GAN | ✅ |
| 4 | Entrenamiento (Colab) | 🟡 en progreso |
| 5 | Evaluación | ⏳ |
| 6 | App web de práctica | ⏳ |
| 7 | Documentación final | ⏳ |

## Experimentos descartados

### Camino B (2026-04-26) — Upsample+Conv + DiffAugment + EMA + TTUR

**Hipótesis:** la receta v1 (cDCGAN baseline) podía mejorarse para datasets chicos combinando 4 ajustes modernos:

- **Upsample(nearest) + Conv 3x3** en vez de `ConvTranspose2d 4x4 stride=2` en el Generator (Odena et al. 2016, evita checkerboard).
- **DiffAugment** policy `"translation,cutout"` aplicada a real y fake antes del D (Zhao et al. 2020, multiplica el dataset efectivo).
- **EMA del Generator** decay 0.999 (StyleGAN2/BigGAN) — el modelo de inferencia es el promedio móvil.
- **TTUR** `lr_d=4*lr_g` (Heusel et al. 2017).

**Resultado:** descartada. Las samples salieron **notablemente peores que el v1**: bajo contraste, crestas grises tenues, aspecto de "boceto a lápiz" en lugar de huella escaneada. Comparación visual confirmada en las 4 clases Vucetich.

**Diagnóstico realizado:**

- Las samples del camino B ya estaban "lavadas" en `epoch_020.png`, mucho antes de que el EMA acumulara historia significativa → **EMA descartado** como causa.
- Comparando G crudo vs G_EMA cargados desde el mismo `ckpt_150.pt`, ambos igual de lavados; el EMA solo suaviza un toque más → **EMA confirmado descartado**.
- Causa más probable: **DiffAugment-cutout** (mask del 50% del área) sobre huellas grayscale. En CelebA/CIFAR donde la información es redundante el cutout funciona; en huellas la señal es el contraste binario fuerte de las crestas, y enmascarar la mitad le baja el "techo de calidad" al D — el G aprende texturas blandas que pasan el test pero no parecen huellas reales.
- Causa secundaria probable: Upsample-nearest + Conv 3x3 entrega texturas más blandas que ConvT 4x4. Sumado al cutout, el efecto se compone.

**Decisión:** revertir los 4 cambios. El `generator.pt` shipping de Fase 5 (v1) sigue sirviendo a la app sin tocar.

**Para futuros intentos** (en orden de mayor a menor probabilidad de mejora):

1. **lightweight-GAN / FastGAN** (Liu et al. 2020) — arquitectura específicamente diseñada para datasets <10k y compute limitado. Requiere refactor grande pero el upside es alto.
2. **R1 regularization** (gradient penalty en reales para el D) sin DiffAugment — cambio chico al D, mejora modesta documentada en StyleGAN2.
3. Si se quiere reintentar DiffAugment, **probar solo `policy="translation"` (sin cutout)** — el cutout es la pieza más sospechosa de haber roto el contraste.

## Licencia

MIT © Ariel
