"""Loop de entrenamiento de la cDCGAN condicional.

Pensado para correr en Colab Free (T4) porque esta maquina no tiene GPU
NVIDIA. Tambien corre en CPU, pero unicamente para smoke tests (cada
epoca tarda ~varios minutos).

Receta ("camino B"): DCGAN clasica + 6 ajustes modernos para datasets
chicos (~6k imagenes):
  - Adam(betas=0.5, 0.999) con **TTUR** (Heusel et al. 2017): lr_d > lr_g.
    Default lr_g=1e-4, lr_d=4e-4. Ayuda al D a mantener el ritmo en
    setups con SN+hinge sin necesidad de mas pasos del D.
  - **Hinge loss**:
      D real -> ReLU(1 - D(real)).mean()
      D fake -> ReLU(1 + D(fake)).mean()
      G      -> -D(fake).mean()
    Mas estable que BCE en datasets chicos y ayuda a que el G aprenda
    detalle fino (crestas) en vez de quedarse en blobs suaves.
  - **Horizontal flip aleatorio** en reales. El flip cambia la clase de
    las presillas (I <-> E), asi que cuando flippeamos tambien intercambiamos
    la etiqueta. Arcos y Verticilos quedan con su clase original.
  - **Class-conditional BatchNorm** en el G (ver src/models/gan.py). Esto
    es lo que hace que el G *tenga que* diferenciar clases en cada capa,
    no solo en la entrada.
  - **DiffAugment** (Zhao et al. 2020) sobre real Y fake antes del D, con
    policy "translation,cutout". Multiplica el dataset efectivo y mantiene
    el gradiente del G fluyendo a traves de la augmentacion.
  - **EMA del Generator**: mantenemos un G "promedio movil" (decay 0.999)
    en paralelo al G de entrenamiento. El sampling y el modelo final usan
    el G_EMA. Suaviza oscilaciones del adversarial training y suele bajar
    FID notablemente, sin costo de GPU adicional (StyleGAN2/BigGAN lo usan
    de forma estandar).

Para cada epoca:
  1. D step: real + DiffAug -> D, fake + DiffAug -> D, hinge loss, optimizer step.
  2. G step: fake (sin detach) + DiffAug -> D, hinge loss, optimizer step.
  3. EMA update del Generator.
  4. Log de losses medios por epoca.
  5. Cada sample_every_epochs: grilla de samples fija por clase, usando G_EMA.
  6. Cada ckpt_every_epochs: guardar checkpoint completo (G + G_EMA + D + opts).

Al final se guarda el state_dict del **G_EMA** en models/final/generator.pt
bajo la key "generator" (lo que necesita la app web; la app no sabe ni
necesita saber que es el promedio movil).

Uso:
  python -m src.training.train                    # defaults (150 epocas)
  python -m src.training.train --epochs 5         # corto
  python -m src.training.train --max-steps 4      # smoke test (2 iters)
"""

import argparse
import copy
import csv
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from ..data.vucetich import LABEL_TO_SYMBOL, NUM_CLASSES, VucetichClass
from ..models.gan import Discriminator, Generator, init_weights
from .config import TrainConfig
from .dataset import HuellasDataset
from .diffaug import diff_augment

# codigos de clase para el swap I<->E al flippear horizontalmente
_PRESILLA_I = int(VucetichClass.PRESILLA_INTERNA)
_PRESILLA_E = int(VucetichClass.PRESILLA_EXTERNA)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _hflip_with_label_swap(imgs: torch.Tensor,
                           labels: torch.Tensor,
                           prob: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip aleatorio por imagen.

    El flip horizontal convierte una presilla interna en externa (y vice
    versa) porque invierte la posicion del core respecto al centro. Para
    arcos y verticilos el flip conserva la clase. Por eso cuando flippeamos
    tambien intercambiamos la etiqueta de las presillas.
    """
    if prob <= 0.0:
        return imgs, labels
    b = imgs.shape[0]
    flip_mask = torch.rand(b, device=imgs.device) < prob
    if not flip_mask.any():
        return imgs, labels

    imgs = imgs.clone()
    imgs[flip_mask] = torch.flip(imgs[flip_mask], dims=[3])

    new_labels = labels.clone()
    swap_i = flip_mask & (labels == _PRESILLA_I)
    swap_e = flip_mask & (labels == _PRESILLA_E)
    new_labels[swap_i] = _PRESILLA_E
    new_labels[swap_e] = _PRESILLA_I
    return imgs, new_labels


def _build_balanced_sampler(dataset: HuellasDataset,
                            cfg: TrainConfig) -> WeightedRandomSampler:
    """Sampler que sub/sobre-muestrea segun frecuencia de clase.

    Pesos por muestra = 1 / count(clase) ** balance_strength.
      strength 1.0 -> 1/freq, balance "duro" (intentado en v2, colapso tardio).
      strength 0.5 -> 1/sqrt(freq), balance moderado (receta v3 sugerida).
    Con reemplazo: cada epoca ve len(dataset) indices muestreados con esos
    pesos. Sin reemplazo nos quedariamos en min(class_count)*NUM_CLASSES ~2000
    indices, demasiado poco para una epoca util.
    """
    labels = dataset.labels.numpy()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    if (class_counts == 0).any():
        missing = [int(c) for c, n in enumerate(class_counts) if n == 0]
        raise RuntimeError(
            f"WeightedRandomSampler: faltan clases en el dataset: {missing}")
    per_class_w = 1.0 / np.power(class_counts, cfg.balance_strength)
    sample_weights = per_class_w[labels]
    print(f"[sampler] class counts: {class_counts.astype(int).tolist()} "
          f"| strength={cfg.balance_strength} "
          f"-> pesos por clase {np.round(per_class_w / per_class_w.min(), 2).tolist()}")
    # generator local para reproducibilidad sin pisar el seed global
    g = torch.Generator().manual_seed(cfg.seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(dataset),
        replacement=True,
        generator=g,
    )


@torch.no_grad()
def _ema_update(g_ema: nn.Module, g: nn.Module, decay: float) -> None:
    """Actualiza pesos y buffers del G_EMA hacia los del G.

    Pesos: ema = decay * ema + (1 - decay) * g  (lerp).
    Buffers (running_mean/var de los BN sin affine, etc.): copia directa.
    Los buffers no son aprendidos -> no tiene sentido promediarlos, copiamos.
    """
    for ema_p, p in zip(g_ema.parameters(), g.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(g_ema.buffers(), g.buffers()):
        ema_b.data.copy_(b.data)


def _build_fixed_samples(cfg: TrainConfig, device: torch.device
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Arma (z, labels) fijos para visualizar progreso siempre con el mismo input.

    samples_per_class por cada clase Vucetich, total = 4 * N imagenes.
    """
    gen = torch.Generator(device="cpu").manual_seed(cfg.sample_seed)
    n = cfg.samples_per_class * NUM_CLASSES
    z = torch.randn(n, cfg.z_dim, generator=gen).to(device)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(
        cfg.samples_per_class).to(device)
    return z, labels


def _save_sample_grid(generator: Generator,
                      fixed_z: torch.Tensor,
                      fixed_labels: torch.Tensor,
                      out_path: Path,
                      samples_per_class: int,
                      epoch: int) -> None:
    """Guarda una grilla (NUM_CLASSES filas x samples_per_class columnas)."""
    generator.eval()
    with torch.no_grad():
        fake = generator(fixed_z, fixed_labels).cpu().numpy()
    # de [-1, 1] a [0, 1]
    fake = (fake * 0.5 + 0.5).clip(0.0, 1.0)

    rows, cols = NUM_CLASSES, samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j] if rows > 1 else axes[j]
            ax.imshow(fake[i * cols + j, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(LABEL_TO_SYMBOL[VucetichClass(i)], fontsize=11)
    fig.suptitle(f"epoca {epoch:03d} — filas=clase, cols=samples", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_checkpoint(path: Path,
                     generator: Generator,
                     generator_ema: Generator,
                     discriminator: Discriminator,
                     opt_g: optim.Optimizer,
                     opt_d: optim.Optimizer,
                     epoch: int,
                     cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator": generator.state_dict(),
        "generator_ema": generator_ema.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "epoch": epoch,
        "z_dim": cfg.z_dim,
        "num_classes": NUM_CLASSES,
    }, path)


def train(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    device = _resolve_device(cfg.device)
    _seed_all(cfg.seed)

    print(f"[device] {device}")

    # data
    dataset = HuellasDataset()
    sampler = _build_balanced_sampler(dataset, cfg) if cfg.balance_classes else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        # sampler y shuffle son mutuamente excluyentes: si hay sampler,
        # shuffle=False; si no, shuffle=True (orden aleatorio dentro de la epoca).
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device.type == "cuda",
        drop_last=True,
    )
    sampler_tag = "balanced" if sampler is not None else "uniforme"
    print(f"[data] {len(dataset)} imagenes | {len(loader)} steps por epoca "
          f"| sampler={sampler_tag}")

    # modelos + init
    generator = Generator(z_dim=cfg.z_dim).to(device)
    discriminator = Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    # G_EMA: copia del Generator que se actualiza como promedio movil tras
    # cada G step. Siempre en eval, sin grad (no participa en backward).
    # El sampling y el modelo final salen de aca, no del G "crudo".
    generator_ema = copy.deepcopy(generator).eval()
    for p in generator_ema.parameters():
        p.requires_grad_(False)

    # TTUR (Heusel et al. 2017): lr_d > lr_g por default.
    opt_g = optim.Adam(generator.parameters(), lr=cfg.lr_g,
                       betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(discriminator.parameters(), lr=cfg.lr_d,
                       betas=(cfg.beta1, cfg.beta2))

    print(f"[lr] g={cfg.lr_g}  d={cfg.lr_d}  (TTUR ratio d/g={cfg.lr_d/cfg.lr_g:.2f})")
    print(f"[diffaug] policy={cfg.diffaug_policy!r}")
    print(f"[ema] decay={cfg.ema_decay}")

    # resume desde checkpoint si se pidio
    start_epoch = 1
    if cfg.resume_from is not None:
        ck_path = Path(cfg.resume_from)
        if not ck_path.exists():
            raise FileNotFoundError(f"resume_from no existe: {ck_path}")
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        generator.load_state_dict(ck["generator"])
        if "discriminator" in ck:
            discriminator.load_state_dict(ck["discriminator"])
            opt_g.load_state_dict(ck["opt_g"])
            opt_d.load_state_dict(ck["opt_d"])
            # Si el ckpt trae G_EMA lo usamos; si no (ckpt previo a camino B),
            # arrancamos el EMA desde el G recien cargado.
            if "generator_ema" in ck:
                generator_ema.load_state_dict(ck["generator_ema"])
            else:
                generator_ema.load_state_dict(generator.state_dict())
                print("[resume] ckpt sin G_EMA -> inicializado desde G")
            start_epoch = int(ck.get("epoch", 0)) + 1
            print(f"[resume] desde {ck_path} -> continuando en epoca {start_epoch}")
        else:
            # checkpoint "final" (solo G). No se puede continuar entrenamiento
            # porque falta el D y los optimizer states.
            raise RuntimeError(
                f"{ck_path} es un checkpoint 'final' (solo G). Para resume "
                "usar un ckpt_NNN.pt de models/checkpoints/.")

    fixed_z, fixed_labels = _build_fixed_samples(cfg, device)

    # log csv: modo append si estamos resumiendo, write si es corrida nueva
    cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_is_new = not (cfg.resume_from is not None and cfg.log_path.exists())
    log_file = open(cfg.log_path, "a" if not log_is_new else "w",
                    newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    if log_is_new:
        log_writer.writerow(["epoch", "loss_d", "loss_g", "seconds"])

    global_step = 0
    stopped_early = False
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            generator.train()
            discriminator.train()
            t0 = time.time()
            sum_d, sum_g, n_batches = 0.0, 0.0, 0

            pbar = tqdm(loader, desc=f"epoca {epoch:03d}")
            for real_imgs, real_labels in pbar:
                real_imgs = real_imgs.to(device, non_blocking=True)
                real_labels = real_labels.to(device, non_blocking=True)

                # horizontal flip aleatorio con swap I<->E para presillas
                real_imgs, real_labels = _hflip_with_label_swap(
                    real_imgs, real_labels, cfg.hflip_prob)
                b = real_imgs.shape[0]

                # --- D step (hinge) ---
                # DiffAugment: aplicamos la MISMA policy a real y a fake antes
                # de que las vea el D. Cada llamada usa randoms independientes
                # (no es necesario que la aug sea la misma para real y fake;
                # solo importa que ambos se augmenten para que el D no aprenda
                # a usar la "augmentacion" como senal de "fake").
                opt_d.zero_grad(set_to_none=True)
                # reales: queremos logits > 1
                real_aug = diff_augment(real_imgs, cfg.diffaug_policy)
                d_real = discriminator(real_aug, real_labels)
                loss_d_real = F.relu(1.0 - d_real).mean()
                # falsos: queremos logits < -1. G genera con clases uniformes.
                z = torch.randn(b, cfg.z_dim, device=device)
                fake_labels = torch.randint(0, NUM_CLASSES, (b,), device=device)
                fake = generator(z, fake_labels).detach()
                fake_aug = diff_augment(fake, cfg.diffaug_policy)
                d_fake = discriminator(fake_aug, fake_labels)
                loss_d_fake = F.relu(1.0 + d_fake).mean()
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                opt_d.step()

                # --- G step (hinge) ---
                # Importante: el fake del G step TAMBIEN pasa por DiffAugment.
                # Como la aug es diferenciable, el gradiente vuelve al G a
                # traves de ella sin que el G aprenda a "imitar" los artefactos
                # de aug.
                opt_g.zero_grad(set_to_none=True)
                z = torch.randn(b, cfg.z_dim, device=device)
                fake_labels = torch.randint(0, NUM_CLASSES, (b,), device=device)
                fake = generator(z, fake_labels)
                fake_aug_g = diff_augment(fake, cfg.diffaug_policy)
                d_fake_for_g = discriminator(fake_aug_g, fake_labels)
                # maximizar D(fake) => minimizar -D(fake)
                loss_g = -d_fake_for_g.mean()
                loss_g.backward()
                opt_g.step()

                # --- EMA update del Generator ---
                # Despues del G step: empujamos g_ema un poquito hacia g.
                # Decay alto (0.999) -> el EMA promedia ~1000 steps recientes.
                _ema_update(generator_ema, generator, cfg.ema_decay)

                sum_d += float(loss_d.detach())
                sum_g += float(loss_g.detach())
                n_batches += 1
                global_step += 1
                pbar.set_postfix(d=f"{loss_d.item():.3f}",
                                 g=f"{loss_g.item():.3f}")

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    stopped_early = True
                    break

            mean_d = sum_d / max(n_batches, 1)
            mean_g = sum_g / max(n_batches, 1)
            dt = time.time() - t0
            print(f"[epoca {epoch:03d}] loss_d={mean_d:.4f}  loss_g={mean_g:.4f}  "
                  f"t={dt:.1f}s")
            log_writer.writerow([epoch, f"{mean_d:.6f}", f"{mean_g:.6f}", f"{dt:.2f}"])
            log_file.flush()

            # sampling: SIEMPRE con G_EMA, que es lo que termina sirviendo
            # la app. Asi lo que vemos en disco es la calidad real de
            # inferencia, no el G "crudo" que oscila mas.
            if epoch % cfg.sample_every_epochs == 0:
                sample_path = cfg.samples_dir / f"epoch_{epoch:03d}.png"
                _save_sample_grid(generator_ema, fixed_z, fixed_labels,
                                  sample_path, cfg.samples_per_class, epoch)

            # checkpoint: guarda G + G_EMA + D + opts para poder resumir
            if epoch % cfg.ckpt_every_epochs == 0:
                ckpt_path = cfg.checkpoints_dir / f"ckpt_{epoch:03d}.pt"
                _save_checkpoint(ckpt_path, generator, generator_ema,
                                 discriminator, opt_g, opt_d, epoch, cfg)
                print(f"[ckpt] {ckpt_path}")

            if stopped_early:
                print(f"[stop] max_steps={cfg.max_steps} alcanzado en epoca {epoch}")
                break

    finally:
        log_file.close()

    # modelo final: guardamos el G_EMA bajo la key "generator". La app
    # carga genericamente bundle["generator"] y no le importa que sea el
    # promedio movil; le importa que la arquitectura coincida con la que
    # importa de src.models.gan, cosa que se cumple.
    cfg.final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "generator": generator_ema.state_dict(),
        "z_dim": cfg.z_dim,
        "num_classes": NUM_CLASSES,
    }, cfg.final_model_path)
    print(f"[final] {cfg.final_model_path}  (G_EMA, decay={cfg.ema_decay})")


def _parse_args() -> TrainConfig:
    cfg = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr-g", type=float, default=cfg.lr_g)
    p.add_argument("--lr-d", type=float, default=cfg.lr_d)
    p.add_argument("--num-workers", type=int, default=cfg.num_workers)
    p.add_argument("--hflip-prob", type=float, default=cfg.hflip_prob)
    p.add_argument("--device", type=str, default=None,
                   help='"cuda" | "cpu". None = auto-detect.')
    p.add_argument("--max-steps", type=int, default=None,
                   help="corta el entrenamiento tras N steps (smoke test)")
    p.add_argument("--sample-every", type=int, default=cfg.sample_every_epochs)
    p.add_argument("--ckpt-every", type=int, default=cfg.ckpt_every_epochs)
    p.add_argument("--resume", type=str, default=None,
                   help="path a ckpt_NNN.pt para continuar entrenamiento")
    # flag bool con default heredado del config (True) y --no-balance para apagar
    bal = p.add_mutually_exclusive_group()
    bal.add_argument("--balance-classes", dest="balance_classes",
                     action="store_true",
                     help="WeightedRandomSampler que iguala las 4 clases Vucetich")
    bal.add_argument("--no-balance-classes", dest="balance_classes",
                     action="store_false",
                     help="DataLoader uniforme (orden estricto del dataset)")
    p.set_defaults(balance_classes=cfg.balance_classes)
    p.add_argument("--balance-strength", type=float, default=cfg.balance_strength,
                   help="Exponente de los pesos: 1.0 = 1/freq (v2, agresivo), "
                        "0.5 = 1/sqrt(freq) (v3, moderado). Solo aplica si "
                        "--balance-classes esta activo.")
    p.add_argument("--diffaug-policy", type=str, default=cfg.diffaug_policy,
                   help='DiffAugment policy ("translation,cutout" | "none").')
    p.add_argument("--ema-decay", type=float, default=cfg.ema_decay,
                   help="Decay del promedio movil del Generator (default 0.999).")
    args = p.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        num_workers=args.num_workers,
        hflip_prob=args.hflip_prob,
        device=args.device,
        max_steps=args.max_steps,
        sample_every_epochs=args.sample_every,
        ckpt_every_epochs=args.ckpt_every,
        resume_from=Path(args.resume) if args.resume else None,
        balance_classes=args.balance_classes,
        balance_strength=args.balance_strength,
        diffaug_policy=args.diffaug_policy,
        ema_decay=args.ema_decay,
    )


if __name__ == "__main__":
    train(_parse_args())
