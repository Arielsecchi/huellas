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

1. Subí la carpeta `huellas-gan/` entera a tu Google Drive en `MyDrive/huellas-gan/` (sin `venv/`, con `data/processed/` lleno).
2. Abrí [notebooks/train_colab.ipynb](notebooks/train_colab.ipynb) en Colab.
3. *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
4. Correr las celdas en orden.

El notebook:

- verifica la GPU,
- monta Drive,
- instala dependencias,
- corre un smoke test de 2 iteraciones,
- entrena 50 épocas (batch 64, ~30-40 min en T4),
- muestra la grilla final de samples y la curva de pérdida.

Los checkpoints, samples y el modelo final quedan persistidos en Drive. La configuración vive en [src/training/config.py](src/training/config.py) — si querés cambiar épocas, lr, etc., editá ahí o pasá flags por CLI.

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

## Licencia

MIT © Ariel
