# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT AND LGPL-3.0-or-later AND Apache-2.0
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
# Las clases marcadas como «Código derivado de terceros» están bajo LGPL-3.0-or-later; el resto, bajo MIT.
"""
LRP con zennit para el ConvNeXt-S del TFG (timm convnext_small + cabeza de fastai). Versión 2 del boceto
OpenOOD/lrp_convnext.py, comprobada con zennit 0.4.6, zennit-crp 0.6.0 y timm 1.0.25 (ver results_crp/resumen.md).

Cambios respecto al boceto:
  1. make_composite: EpsilonPlusFlat de zennit 0.4.6 NO acepta layer_map (su __init__ solo admite canonizers); se
     construye el mismo composite con SpecialFirstLayerMapComposite.
  2. El forward del canonizer conserva la rama de gamma: replica exactamente ConvNeXtBlock.forward de timm 1.0.25 y
     vale también si gamma no se ha absorbido (variante de ablación d).
  3. Regla para BatchNorm1d de la cabeza de fastai (opcional, bn_pass): zennit no le asigna regla y no se puede
     fusionar con la Linear porque va antes de ella.
  4. copia_para_lrp: desactiva ReLU(inplace=True) en la copia (misma función, sin escritura in situ).
  5. Reglas epsilon y z+ sin sesgo (opcional, sin_sesgo) y regla LN de Ali et al. (opcional, ln='regla_ln').
  6. Pass en el GELU de timm (gelu_pass): el boceto suponía que zennit ya lo cubría, pero solo reconoce torch.nn.GELU.
"""
import copy

import torch
from timm.layers import LayerNorm2d
from timm.layers.activations import GELU as TimmGELU
from timm.models.convnext import ConvNeXtBlock
from zennit.canonizers import AttributeCanonizer
from zennit.composites import LAYER_MAP_BASE, SpecialFirstLayerMapComposite
from zennit.core import BasicHook, Hook, stabilize
from zennit.layer import Sum
from zennit.rules import Epsilon, Flat, Pass, ZPlus
from zennit.types import BatchNorm, Convolution, Linear


def copia_para_lrp(model, plegar_gamma=True):
    """Copia en modo eval; el modelo original no se toca. Absorbe gamma en fc2 si plegar_gamma."""
    model = copy.deepcopy(model).eval()
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False
    if plegar_gamma:
        with torch.no_grad():
            for block in model.modules():
                if isinstance(block, ConvNeXtBlock) and block.gamma is not None:
                    g = block.gamma.detach().clone()
                    fc2 = block.mlp.fc2
                    # Linear: weight (out, in); Conv2d 1x1: weight (out, in, 1, 1)
                    shape = (-1,) + (1,) * (fc2.weight.dim() - 1)
                    fc2.weight.mul_(g.reshape(shape))
                    if fc2.bias is not None:
                        fc2.bias.mul_(g)
                    block.gamma = None
    return model


# --- Código derivado de terceros (ver XAI/lrp/LICENSES/) ---------------------------------------------------
# Origen 1: zennit 0.4.6, zennit/torchvision.py, clase ResNetBottleneckCanonizer (https://github.com/chr5tphr/zennit),
#   Copyright (C) 2019-2021 Christopher J. Anders,
#   licencia LGPL-3.0-or-later. Se toma la estructura: __init__, _attribute_map (forward + módulo Sum) y la suma
#   residual con torch.stack(...) + canonizer_sum.
# Origen 2: timm 1.0.25, timm/models/convnext.py, ConvNeXtBlock.forward (https://github.com/huggingface/pytorch-image-models),
#   Copyright 2019 Ross Wightman,
#   licencia Apache-2.0. El cuerpo de forward es el de timm.
# Cambios: se aplica a ConvNeXtBlock de timm en lugar del Bottleneck de torchvision, y la suma final
#   `self.drop_path(x) + self.shortcut(shortcut)` se sustituye por torch.stack(...) + self.canonizer_sum(x).
# Esta clase se distribuye bajo LGPL-3.0-or-later (incluye código Apache-2.0); el resto del fichero, bajo MIT.
class ConvNeXtBlockCanonizer(AttributeCanonizer):
    """Sustituye la suma residual funcional por un módulo Sum, que zennit sí ve."""

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, ConvNeXtBlock):
            return {'forward': cls.forward.__get__(module), 'canonizer_sum': Sum()}
        return None

    @staticmethod
    def forward(self, x):
        # Igual que ConvNeXtBlock.forward de timm 1.0.25, salvo la suma final
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = torch.stack([self.drop_path(x), self.shortcut(shortcut)], dim=-1)
        return self.canonizer_sum(x)


class EpsilonSinSesgo(BasicHook):
    """Regla epsilon sin el sesgo: suma R_entrada = R. Con sesgo, suma R_entrada = R (z - b) / z, que se dispara
    cuando |b| es comparable a |z| (fc1 y fc2 de los bloques de ConvNeXt tienen sesgo)."""

    def __init__(self, epsilon=1e-6):
        super().__init__(
            input_modifiers=[lambda input: input],
            param_modifiers=[lambda param, name: torch.zeros_like(param) if name == 'bias' else param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0], epsilon)),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]))


# --- Código derivado de terceros (ver XAI/lrp/LICENSES/) ---------------------------------------------------
# Origen: zennit 0.4.6, zennit/rules.py, clase ZPlus (https://github.com/chr5tphr/zennit),
#   Copyright (C) 2019-2021 Christopher J. Anders, licencia LGPL-3.0-or-later.
# Cambios: en la primera rama (pesos positivos) el sesgo se anula en lugar de recortarse a su parte positiva;
#   el resto (modificadores de entrada, gradient_mapper y reducer) es el de ZPlus.
# Esta clase se distribuye bajo LGPL-3.0-or-later; el resto del fichero, bajo MIT.
class ZPlusSinSesgo(BasicHook):
    """Regla z+ de zennit sin la parte positiva del sesgo, que ZPlus conserva en su primera rama."""

    def __init__(self):
        sin_sesgo = (lambda param, name, f: torch.zeros_like(param) if name == 'bias' else f(param))
        super().__init__(
            input_modifiers=[lambda input: input.clamp(min=0), lambda input: input.clamp(max=0)],
            param_modifiers=[lambda param, name: sin_sesgo(param, name, lambda p: p.clamp(min=0)),
                             lambda param, name: sin_sesgo(param, name, lambda p: p.clamp(max=0))],
            output_modifiers=[lambda output: output] * 2,
            gradient_mapper=(lambda out_grad, outputs: [out_grad / stabilize(sum(outputs))] * 2),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0] + inputs[1] * gradients[1]))


class ReglaLN(Hook):
    """Regla LN (Ali et al., ICML 2022): la desviación típica se trata como constante, de modo que la normalización
    es lineal en x: z_j = gamma_j (x_j - media(x)) / sigma. Se reparte con LRP-0 sobre esa aplicación lineal, sin el
    desplazamiento beta, así que es conservativa. timm: LayerNorm normaliza la última dimensión y LayerNorm2d, la 1."""

    def forward(self, module, input, output):
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors['input'][0].detach().requires_grad_()
        dim = 1 if isinstance(module, LayerNorm2d) else -1
        forma = [1] * x.dim()
        forma[dim] = -1
        with torch.autograd.enable_grad():
            sigma = (x.var(dim=dim, keepdim=True, unbiased=False) + module.eps).sqrt().detach()
            z = (x - x.mean(dim=dim, keepdim=True)) / sigma * module.weight.reshape(forma)
        gradiente = torch.autograd.grad(z, x, grad_outputs=grad_output[0] / stabilize(z))[0]
        relevancia = x * gradiente
        return tuple(relevancia if o.shape == relevancia.shape else None for o in grad_input)


class SumaAbs(BasicHook):
    """Suma residual repartida por |x|: R_i = |x_i| R / sum|x|. Es conservativa y acotada aunque rama y atajo tengan
    signos opuestos y su suma se acerque a 0, caso en el que la regla Norm de zennit se dispara."""

    def __init__(self):
        super().__init__(
            input_modifiers=[lambda input: input.abs()],
            param_modifiers=[lambda param, _: param],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / stabilize(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]))


class ComposicionConvNeXt(SpecialFirstLayerMapComposite):
    """Permite una regla distinta para fc1 y fc2 de los bloques (son convoluciones 1x1 sobre channels-last)."""

    def __init__(self, layer_map, first_map, regla_mlp=None, regla_cabeza=None, canonizers=None):
        super().__init__(layer_map, first_map, canonizers=canonizers)
        self.regla_mlp = regla_mlp
        self.regla_cabeza = regla_cabeza

    def mapping(self, ctx, name, module):
        if self.regla_mlp is not None and isinstance(module, torch.nn.Linear) and '.mlp.fc' in name:
            return self.regla_mlp
        if self.regla_cabeza is not None and name == '1.4':      # primera Linear de la cabeza de fastai
            return self.regla_cabeza
        return super().mapping(ctx, name, module)


def make_composite(ln='pass', canonizer=True, bn_pass=False, sin_sesgo=False, mlp=None, suma='norm', epsilon=1e-6,
                   gelu_pass=False, cabeza_zplus=False):
    """EpsilonPlusFlat (flat en la primera capa, z+ en convoluciones, epsilon en Linear) con reglas adicionales.
    ln: 'pass', 'regla_ln' o None (sin regla: gradiente normal). bn_pass: Pass en BatchNorm1d de la cabeza.
    sin_sesgo: epsilon y z+ sin sesgos. mlp: 'zplus' para fc1/fc2 de los bloques (por defecto, la regla de Linear).
    suma: 'norm' (zennit) o 'abs' (SumaAbs). epsilon: valor de la regla epsilon.
    gelu_pass: Pass en el GELU de timm, que NO es torch.nn.GELU y zennit no reconoce como activación: sin esta regla
    la relevancia se multiplica por la derivada de GELU, sin ningún aviso."""
    extra = []
    if gelu_pass:
        extra.append((TimmGELU, Pass()))
    if ln == 'pass':
        extra.append((torch.nn.LayerNorm, Pass()))    # incluye LayerNorm y LayerNorm2d de timm (heredan de ella)
    elif ln == 'regla_ln':
        extra.append((torch.nn.LayerNorm, ReglaLN()))
    if bn_pass:
        extra.append((BatchNorm, Pass()))
    if suma == 'abs':
        extra.append((Sum, SumaAbs()))
    capas = [(Convolution, ZPlusSinSesgo()), (torch.nn.Linear, EpsilonSinSesgo(epsilon))] if sin_sesgo else \
            [(Convolution, ZPlus()), (torch.nn.Linear, Epsilon(epsilon))]
    regla_mlp = (ZPlusSinSesgo() if sin_sesgo else ZPlus()) if mlp == 'zplus' else None
    return ComposicionConvNeXt(extra + LAYER_MAP_BASE + capas, [(Linear, Flat())], regla_mlp=regla_mlp,
                               regla_cabeza=(ZPlusSinSesgo() if cabeza_zplus else None),
                               canonizers=[ConvNeXtBlockCanonizer()] if canonizer else None)
