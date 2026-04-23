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

## Estado actual

**Fase 1 — Setup del proyecto:** terminada. Próximo paso: Fase 2 (dataset).

| Fase | Descripción | Estado |
|---|---|---|
| 0 | Diagnóstico de hardware | ✅ |
| 1 | Setup del proyecto | ✅ |
| 2 | Dataset + preproceso + etiquetado Vucetich | 🟡 siguiente |
| 3 | Arquitectura de la GAN | ⏳ |
| 4 | Entrenamiento (Colab) | ⏳ |
| 5 | Evaluación | ⏳ |
| 6 | App web de práctica | ⏳ |
| 7 | Documentación final | ⏳ |

## Licencia

MIT © Ariel
