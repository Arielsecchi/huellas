"""Arquitectura de la GAN condicional (cDCGAN) para huellas 128x128.

Basada en DCGAN (Radford et al. 2015) + condicionamiento mediante
**class-conditional BatchNorm** (de Vries et al. 2017, Miyato et al. 2018).

Por que cBN y no concat-en-z como en cDCGAN original: con solo concatenar la
etiqueta al ruido z en la entrada del G, la señal de clase se diluye en capas
profundas y el G termina ignorandola. cBN inyecta la etiqueta en CADA bloque
reemplazando los (gamma, beta) globales del BatchNorm por (gamma, beta) por
clase. Asi la condicion llega con fuerza a todas las capas del G.

Mantenemos el concat-en-z TAMBIEN como señal auxiliar al primer bloque: el
ruido y la clase se mezclan desde la entrada y se refuerzan capa por capa.

Up-sampling del Generator: **Upsample(nearest) + Conv2d 3x3** en cada bloque
en vez de ConvTranspose2d 4x4 stride=2. La ConvT con stride=2 produce
artefactos de checkerboard / blur sistematico cuando los strides no dividen
el kernel (Odena, Dumoulin & Olah 2016, "Deconvolution and Checkerboard
Artifacts"). El reemplazo por up-nearest + conv 3x3 (que es lo que usan
StyleGAN/BigGAN/PGAN) elimina el checkerboard y produce texturas mas
limpias, al mismo costo de parametros. El primer bloque (1x1 -> 4x4) sigue
siendo ConvT 4x4 stride=1 padding=0 porque ahi no hay stride>1, no hay
checkerboard, y es la unica forma sensata de subir desde 1x1.

Discriminator: condicionamiento por mapa espacial aprendible concatenado
como canal extra (Mirza-Osindero) + **spectral normalization** en cada
Conv2d (Miyato et al. 2018). SN reemplaza al BN del D original: mantiene
la red 1-Lipschitz, evita que el D explote sus logits y aplaste al G en
las primeras epocas. Es la receta estandar para hinge GAN.

Por que cDCGAN-modificado y no algo mas moderno (SAGAN, StyleGAN, BigGAN):

  1. Es la arquitectura condicional mas simple que funciona bien a 128x128
     con datasets chicos (~6k). Ideal para entrenar en Colab Free (T4, 12 GB).
  2. Es pedagogica: el codigo se lee de arriba abajo sin capas
     sofisticadas (attention, style blocks, noise injection, etc.).
  3. Si los resultados no convencen pasamos a algo mas pesado en Fase 5,
     pero primero queremos una baseline honesta.

Dimensionado:

  z         : 100-d  (estandar DCGAN)
  embed     : 100-d  (match con z, para que condicion y ruido pesen igual)
  G input   : (B, 200, 1, 1)
  G path    : 1x1 -> 4x4 -> 8 -> 16 -> 32 -> 64 -> 128
  canales G : 200 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 1
  D input   : (B, 1+1, 128, 128)   # imagen + canal de condicion
  D path    : 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 1
  canales D : 2 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 1

Convenciones:

  - Imagenes en [-1, 1]  (salida Tanh del G, preproc del dataset).
  - Discriminator devuelve LOGITS, no probabilidades: el loop de
    entrenamiento usa hinge loss sobre los logits directo.
  - Init de pesos: N(0, 0.02) en Conv/ConvT, N(1, 0.02) en BatchNorm
    (receta original DCGAN).
"""

import torch
from torch import nn

from ..data.vucetich import NUM_CLASSES

Z_DIM = 100
EMBED_DIM = 100
IMG_SIZE = 128
IMG_CHANNELS = 1

G_BASE_CHANNELS = 64   # multiplicador base del Generator
D_BASE_CHANNELS = 64   # multiplicador base del Discriminator

WEIGHT_INIT_MEAN = 0.0
WEIGHT_INIT_STD = 0.02
BN_INIT_MEAN = 1.0
BN_INIT_STD = 0.02


class ConditionalBatchNorm2d(nn.Module):
    """BatchNorm con (gamma, beta) aprendibles por clase.

    Ecuacion: y = gamma(clase) * normalize(x) + beta(clase)

    La normalizacion (resta media, divide std) es identica a un BN estandar;
    cambia solo la parte affine (scale/shift), que depende de la etiqueta.
    Init: gamma=1, beta=0 por clase -> arranca como un BN afin trivial.
    """

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        # BN sin affine: solo normaliza. El affine lo mete cada clase.
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_embed = nn.Embedding(num_classes, num_features)
        self.beta_embed = nn.Embedding(num_classes, num_features)
        nn.init.ones_(self.gamma_embed.weight)
        nn.init.zeros_(self.beta_embed.weight)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        gamma = self.gamma_embed(y).view(-1, x.size(1), 1, 1)
        beta = self.beta_embed(y).view(-1, x.size(1), 1, 1)
        return gamma * out + beta


class _GInitBlock(nn.Module):
    """Proyecta (B, in_dim, 1, 1) -> (B, out_ch, 4, 4) con cBN + ReLU.

    Usa ConvT 4x4 stride 1 padding 0 porque arrancamos desde 1x1.
    """

    def __init__(self, in_dim: int, out_ch: int, num_classes: int) -> None:
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_dim, out_ch, kernel_size=4,
                                        stride=1, padding=0, bias=False)
        self.cbn = ConditionalBatchNorm2d(out_ch, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.act(self.cbn(self.convt(x), y))


class _GUpBlock(nn.Module):
    """Bloque up del Generator: Upsample(nearest) 2x + Conv 3x3 + cBN + ReLU.

    El Upsample-nearest duplica H y W sin parametros y sin artefactos de
    interpolacion, y la Conv 3x3 padding=1 mantiene esas dimensiones y
    aprende a "limpiar" la salida. Esto evita el checkerboard que produce
    ConvTranspose 4x4 stride=2 (Odena, Dumoulin & Olah 2016).
    """

    def __init__(self, in_ch: int, out_ch: int, num_classes: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.cbn = ConditionalBatchNorm2d(out_ch, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.act(self.cbn(self.conv(self.up(x)), y))


def _sn(module: nn.Module) -> nn.Module:
    """Wrapper: aplica spectral normalization a un modulo con .weight.

    La SN reparametriza W como W / sigma(W), con sigma = mayor valor
    singular estimado via power iteration. Asi la capa queda 1-Lipschitz
    respecto a la entrada y el D no puede "explotar" sus logits, lo que
    evita que aplaste al G en las primeras epocas (problema crucial con
    hinge loss en datasets chicos).
    """
    return nn.utils.parametrizations.spectral_norm(module)


def _d_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Bloque estandar del Discriminator: Conv 4x4 stride 2 + SN + LReLU.

    Reduce H y W a la mitad. Spectral norm reemplaza el BN original de
    DCGAN: controla la escala de los logits sin mezclar estadisticas
    entre ejemplos (cosa que le traia problemas al D condicional).
    """
    conv = _sn(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2,
                         padding=1, bias=True))
    return nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True))


class Generator(nn.Module):
    """Generator condicional: (z, clase) -> imagen 1x128x128 en [-1, 1].

    Doble via de condicionamiento: (a) embedding concatenado a z en la
    entrada, (b) cBN en cada bloque. La (b) es la que realmente fuerza al
    G a diferenciar clases en capas profundas.
    """

    def __init__(self,
                 z_dim: int = Z_DIM,
                 embed_dim: int = EMBED_DIM,
                 num_classes: int = NUM_CLASSES,
                 base_channels: int = G_BASE_CHANNELS,
                 img_channels: int = IMG_CHANNELS) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # embedding de clase para el concat-en-z
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        c = base_channels
        in_dim = z_dim + embed_dim

        self.init_block = _GInitBlock(in_dim, 16 * c, num_classes)   # 1 -> 4
        self.up1 = _GUpBlock(16 * c, 8 * c, num_classes)             # 4  -> 8
        self.up2 = _GUpBlock(8 * c, 4 * c, num_classes)              # 8  -> 16
        self.up3 = _GUpBlock(4 * c, 2 * c, num_classes)              # 16 -> 32
        self.up4 = _GUpBlock(2 * c, c, num_classes)                  # 32 -> 64

        # bloque final: Upsample 2x + Conv 3x3 a img_channels + Tanh. Sin BN.
        # Mismo motivo que _GUpBlock: evitar checkerboard de ConvT stride=2.
        self.to_img = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c, img_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z:      (B, z_dim)
        # labels: (B,)   enteros en [0, num_classes)
        embed = self.label_embed(labels)                       # (B, embed_dim)
        x = torch.cat([z, embed], dim=1).unsqueeze(-1).unsqueeze(-1)
        # x: (B, z_dim + embed_dim, 1, 1)
        x = self.init_block(x, labels)   # (B, 16c, 4, 4)
        x = self.up1(x, labels)          # (B,  8c, 8, 8)
        x = self.up2(x, labels)          # (B,  4c, 16, 16)
        x = self.up3(x, labels)          # (B,  2c, 32, 32)
        x = self.up4(x, labels)          # (B,   c, 64, 64)
        return self.to_img(x)            # (B, img_channels, 128, 128)


class Discriminator(nn.Module):
    """Discriminator condicional: (imagen, clase) -> logit real/fake."""

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 img_size: int = IMG_SIZE,
                 img_channels: int = IMG_CHANNELS,
                 base_channels: int = D_BASE_CHANNELS) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.img_channels = img_channels

        # embedding: un mapa espacial 128x128 aprendible por clase, que se
        # concatena a la imagen como canal extra. Asi la condicion llega
        # con la misma resolucion que la imagen.
        self.label_embed = nn.Embedding(num_classes, img_size * img_size)

        c = base_channels
        in_ch = img_channels + 1   # imagen + canal de condicion

        self.down1 = _d_block(in_ch, c)        # 128 -> 64
        self.down2 = _d_block(c, 2 * c)        # 64  -> 32
        self.down3 = _d_block(2 * c, 4 * c)    # 32  -> 16
        self.down4 = _d_block(4 * c, 8 * c)    # 16  -> 8
        self.down5 = _d_block(8 * c, 16 * c)   # 8   -> 4

        # bloque final: Conv 4x4 stride 1 padding 0 -> logit escalar.
        # SN tambien aca para mantener la cadena 1-Lipschitz hasta el logit.
        self.to_logit = _sn(nn.Conv2d(16 * c, 1, kernel_size=4, stride=1,
                                      padding=0, bias=True))

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # img:    (B, img_channels, 128, 128)
        # labels: (B,)
        b = img.shape[0]
        cond = self.label_embed(labels).view(b, 1, self.img_size, self.img_size)
        x = torch.cat([img, cond], dim=1)   # (B, img_channels+1, 128, 128)
        x = self.down1(x)   # (B,  c, 64, 64)
        x = self.down2(x)   # (B, 2c, 32, 32)
        x = self.down3(x)   # (B, 4c, 16, 16)
        x = self.down4(x)   # (B, 8c, 8, 8)
        x = self.down5(x)   # (B, 16c, 4, 4)
        x = self.to_logit(x)   # (B, 1, 1, 1)
        return x.view(b)       # (B,)


def init_weights(module: nn.Module) -> None:
    """Inicializacion estandar DCGAN aplicada via model.apply(init_weights).

    Conv/ConvTranspose: N(0, 0.02); BatchNorm: weight N(1, 0.02), bias=0.
    Los embeddings los dejamos intactos: el `label_embed` del G en default
    de PyTorch, y los gamma/beta de cBN ya se inicializaron a 1/0 en el
    constructor (arranque como BN afin trivial).
    """
    classname = module.__class__.__name__
    if "Conv" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=WEIGHT_INIT_MEAN,
                            std=WEIGHT_INIT_STD)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
    elif "BatchNorm" in classname:
        # cBN tiene bn.affine=False -> no tiene weight/bias que inicializar.
        # Solo BN "full affine" del D entra aca.
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight, mean=BN_INIT_MEAN,
                            std=BN_INIT_STD)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def _smoke_test() -> None:
    """Instancia G y D, corre un forward y printea shapes + #params.

    Sirve como sanity check rapido: si las shapes salen bien, la
    arquitectura esta enchufada correctamente y podemos pasar a escribir
    el loop de entrenamiento.
    """
    torch.manual_seed(0)
    batch = 4

    g = Generator()
    d = Discriminator()
    g.apply(init_weights)
    d.apply(init_weights)

    z = torch.randn(batch, Z_DIM)
    labels = torch.randint(0, NUM_CLASSES, (batch,))

    fake = g(z, labels)
    logits = d(fake, labels)

    def _n_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"z      : {tuple(z.shape)}")
    print(f"labels : {tuple(labels.shape)}  (values={labels.tolist()})")
    print(f"fake   : {tuple(fake.shape)}  range=[{fake.min():.3f}, {fake.max():.3f}]")
    print(f"logits : {tuple(logits.shape)}")
    print(f"G params: {_n_params(g):,}")
    print(f"D params: {_n_params(d):,}")

    assert fake.shape == (batch, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), \
        f"Generator output shape incorrecto: {fake.shape}"
    assert logits.shape == (batch,), \
        f"Discriminator output shape incorrecto: {logits.shape}"
    assert fake.min() >= -1.0 - 1e-5 and fake.max() <= 1.0 + 1e-5, \
        "Generator deberia devolver valores en [-1, 1] (Tanh)"
    print("[ok] smoke test paso")


if __name__ == "__main__":
    _smoke_test()
