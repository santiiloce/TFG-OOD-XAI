# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT AND LGPL-3.0-or-later AND Apache-2.0
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
# Las clases marcadas como «Código derivado de terceros» están bajo LGPL-3.0-or-later; el resto, bajo MIT.
"""LRP para el ResNet-50 de timm y variantes de ConvNeXt de la fase 2."""
import torch
from timm.models.resnet import Bottleneck as TimmBottleneck
from zennit.canonizers import AttributeCanonizer, SequentialMergeBatchNorm
from zennit.composites import LAYER_MAP_BASE, EpsilonPlusFlat, SpecialFirstLayerMapComposite
from zennit.core import stabilize
from zennit.layer import Sum
from zennit.rules import Epsilon, Flat, Pass, ZPlus
from zennit.types import BatchNorm, Convolution, Linear

from comun2 import BASE_A
from lrp_convnext_v2 import EpsilonSinSesgo, ReglaLN, make_composite


# --- Código derivado de terceros (ver XAI/lrp/LICENSES/) ---------------------------------------------------
# Origen 1: zennit 0.4.6, zennit/torchvision.py, clase ResNetBottleneckCanonizer (https://github.com/chr5tphr/zennit),
#   Copyright (C) 2019-2021 Christopher J. Anders,
#   licencia LGPL-3.0-or-later. Se toma la estructura: __init__, _attribute_map (forward + módulo Sum) y la suma
#   residual con torch.stack(...) + canonizer_sum.
# Origen 2: timm 1.0.25, timm/models/resnet.py, Bottleneck.forward (https://github.com/huggingface/pytorch-image-models),
#   Copyright 2019 Ross Wightman,
#   licencia Apache-2.0. El cuerpo de forward es el de timm.
# Cambios: se aplica a timm.models.resnet.Bottleneck en lugar del de torchvision, y `x += shortcut` se sustituye por
#   torch.stack([shortcut, x], dim=-1) + self.canonizer_sum(x).
# Esta clase se distribuye bajo LGPL-3.0-or-later (incluye código Apache-2.0); el resto del fichero, bajo MIT.
class TimmBottleneckCanonizer(AttributeCanonizer):
    """Como zennit.torchvision.ResNetBottleneckCanonizer, pero para timm.models.resnet.Bottleneck (esa clase solo
    reconoce el Bottleneck de torchvision). Replica el forward de timm 1.0.25 con la suma residual como módulo Sum."""

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, TimmBottleneck):
            return {'forward': cls.forward.__get__(module), 'canonizer_sum': Sum()}
        return None

    @staticmethod
    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.se is not None:
            x = self.se(x)
        if self.drop_path is not None:
            x = self.drop_path(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        return self.act3(x)


def composite_resnet(residual=True, bn_pass=True):
    """Método estándar: EpsilonPlusFlat + fusión de BatchNorm + suma residual como módulo; Pass en BatchNorm de la cabeza."""
    canon = [SequentialMergeBatchNorm()] + ([TimmBottleneckCanonizer()] if residual else [])
    extra = [(BatchNorm, Pass())] if bn_pass else []
    return SpecialFirstLayerMapComposite(extra + LAYER_MAP_BASE + [(Convolution, ZPlus()), (torch.nn.Linear, Epsilon())],
                                         [(Linear, Flat())], canonizers=canon)


def composite_tutor():
    """Montaje del código de STOOD-X: EpsilonPlusFlat con SequentialMergeBatchNorm, sin canonizer de la residual."""
    return EpsilonPlusFlat(canonizers=[SequentialMergeBatchNorm()])


class ReglaLNEstabilizada(ReglaLN):
    """Regla LN con un estabilizador mayor en el denominador z (no conservativa en sentido estricto)."""

    def __init__(self, epsilon=1e-2):
        super().__init__()
        self.epsilon = epsilon

    def copy(self):
        return ReglaLNEstabilizada(self.epsilon)

    def backward(self, module, grad_input, grad_output):
        from timm.layers import LayerNorm2d
        x = self.stored_tensors['input'][0].detach().requires_grad_()
        dim = 1 if isinstance(module, LayerNorm2d) else -1
        forma = [1] * x.dim()
        forma[dim] = -1
        with torch.autograd.enable_grad():
            sigma = (x.var(dim=dim, keepdim=True, unbiased=False) + module.eps).sqrt().detach()
            z = (x - x.mean(dim=dim, keepdim=True)) / sigma * module.weight.reshape(forma)
        g = torch.autograd.grad(z, x, grad_outputs=grad_output[0] / stabilize(z, self.epsilon))[0]
        r = x * g
        return tuple(r if o.shape == r.shape else None for o in grad_input)


def composite_A(**kw):
    return make_composite(**{**BASE_A, **kw})


def composite_A_ln_est(epsilon=1e-2):
    comp = make_composite(**{**BASE_A, 'ln': None})
    comp.layer_map.insert(0, (torch.nn.LayerNorm, ReglaLNEstabilizada(epsilon)))
    return comp


def composite_A_eps_etapa3(solo_mlp=False):
    """Paso 3: A con epsilon (sin sesgo) en stages.3 y z+ en las etapas inferiores."""
    comp = make_composite(**BASE_A)
    base = comp.module_map
    tipos = (torch.nn.Linear,) if solo_mlp else (torch.nn.Linear, torch.nn.Conv2d)
    regla = EpsilonSinSesgo()

    def mapa(ctx, name, module):
        if '.stages.3.' in name and isinstance(module, tipos):
            return regla
        return base(ctx, name, module)
    comp.module_map = mapa
    return comp


def composite_resnet_como_A():
    """R_estandar con las reglas de A: z+ y epsilon sin sesgo (la fusión de BatchNorm deja sesgos) y suma residual por |x|."""
    from lrp_convnext_v2 import SumaAbs, ZPlusSinSesgo
    return SpecialFirstLayerMapComposite(
        [(BatchNorm, Pass()), (Sum, SumaAbs())] + LAYER_MAP_BASE + [(Convolution, ZPlusSinSesgo()), (torch.nn.Linear, EpsilonSinSesgo())],
        [(Linear, Flat())], canonizers=[SequentialMergeBatchNorm(), TimmBottleneckCanonizer()])
