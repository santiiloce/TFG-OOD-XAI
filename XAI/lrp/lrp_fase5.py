# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Reglas de la fase 5: LRP-gamma por contribuciones por grupo de capas y tratamientos H1-H4 de la BatchNorm de la cabeza."""
import os
import re
import sys

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
FASE5 = os.path.join(RESULTS, 'fase5')
sys.path.insert(0, os.path.join(RESULTS, 'fase3', 'scripts'))

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
from comun3 import BASE_A, boot_ic, cargar, fusionar_bn, preprocesar, seleccion, semilla_clase   # noqa: E402,F401
from lrp_convnext_v2 import (ComposicionConvNeXt, EpsilonSinSesgo, ZPlusSinSesgo, copia_para_lrp,  # noqa: E402,F401
                             make_composite)
from pathlib import Path                                                 # noqa: E402
from zennit.core import BasicHook, Hook, stabilize                       # noqa: E402
from zennit.rules import Epsilon                                         # noqa: E402

INF = float('inf')
GAMMAS = [0.25, 1.0, INF]
HS = ['H1', 'H2', 'H3', 'H4']
FUSIONADA = {'H1': False, 'H2': True, 'H3': True, 'H4': False}
GRUPO = {0: 0, 1: 0, 2: 1, 3: 2}     # etapa -> índice de grupo (G1, G2, G3)
EPS_BN = 1e-6


class GammaContrib(BasicHook):
    """LRP-gamma por contribuciones, sin sesgo: R_j = sum_k (z_jk + gamma z_jk^+) / sum_j (z_jk + gamma z_jk^+) R_k, con z_jk = x_j w_jk.
    z^+ = x^+ w^+ + x^- w^- y z^- = x^+ w^- + x^- w^+, de modo que una contribución positiva lo es por su signo real, no por el del peso."""

    def __init__(self, gamma, epsilon=1e-6):
        s = 1.0 + gamma

        def peso(f):
            return lambda param, name: torch.zeros_like(param) if name == 'bias' else f(param)
        pos, neg = (lambda p: p.clamp(min=0)), (lambda p: p.clamp(max=0))
        xpos, xneg = (lambda i: i.clamp(min=0)), (lambda i: i.clamp(max=0))

        def mapa(g, o):
            den = stabilize(s * (o[0] + o[1]) + o[2] + o[3], epsilon)
            return [s * g / den, s * g / den, g / den, g / den]
        super().__init__(input_modifiers=[xpos, xneg, xpos, xneg], param_modifiers=[peso(pos), peso(neg), peso(neg), peso(pos)],
                         output_modifiers=[lambda o: o] * 4, gradient_mapper=mapa,
                         reducer=lambda i, g: i[0] * g[0] + i[1] * g[1] + i[2] * g[2] + i[3] * g[3])


def parametros_bn(bn):
    """y_i = w_i x_i + b_i en evaluación: w = gamma/sqrt(var + eps_BN), b = beta - mu w."""
    w = bn.weight / (bn.running_var + bn.eps).sqrt()
    return w, bn.bias - bn.running_mean * w


class ReglaAbsZ(Hook):
    """Regla |z| aplicada en la BN1d, según la fórmula del prompt (Hui y Binder, 2019, ec. 9): R(x) = |x w| / (|x w| + |b| + eps) R(y)."""

    def __init__(self, epsilon=EPS_BN):
        super().__init__()
        self.epsilon = epsilon

    def copy(self):
        return ReglaAbsZ(self.epsilon)

    def forward(self, module, input, output):
        self.stored_tensors['input'] = input

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors['input'][0].detach()
        w, b = parametros_bn(module)
        a = (x * w).abs()
        r = a / (a + b.abs() + self.epsilon) * grad_output[0]
        return tuple(r if o is not None and o.shape == r.shape else None for o in grad_input)


class ComposicionFase5(ComposicionConvNeXt):
    def __init__(self, layer_map, first_map, reglas_grupo, regla_lin, regla_bn, canonizers):
        super().__init__(layer_map, first_map, canonizers=canonizers)
        self.reglas_grupo, self.regla_lin, self.regla_bn = reglas_grupo, regla_lin, regla_bn

    def mapping(self, ctx, name, module):
        m = re.match(r'0\.model\.stages\.(\d)\.', name)
        if m and isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            return self.reglas_grupo[GRUPO[int(m.group(1))]]
        if name in ('1.4', '1.8') and isinstance(module, torch.nn.Linear):
            return self.regla_lin
        if isinstance(module, torch.nn.BatchNorm1d):
            return self.regla_bn
        return super().mapping(ctx, name, module)


def regla_gamma(g):
    return ZPlusSinSesgo() if g == INF else GammaContrib(g)


def composite(g1=INF, g2=INF, g3=INF, h='H1'):
    """Composite de una configuración; usar sobre preparar_modelo(model, h)."""
    base = make_composite(**BASE_A)
    regla_lin = Epsilon(1e-6) if h == 'H3' else EpsilonSinSesgo(1e-6)
    regla_bn = ReglaAbsZ() if h == 'H4' else base.layer_map[[isinstance(r, Hook) and t is not None and getattr(t, '__name__', '') == 'BatchNorm' for t, r in base.layer_map].index(True)][1]
    return ComposicionFase5(base.layer_map, base.first_map, [regla_gamma(g1), regla_gamma(g2), regla_gamma(g3)], regla_lin, regla_bn, base.canonizers)


def preparar_modelo(model, h, device=None):
    m = copia_para_lrp(model)
    if device is not None:
        m = m.to(device)
    return fusionar_bn(m) if FUSIONADA[h] else m


def nombre_cfg(g1, g2, g3, h):
    f = lambda g: '∞' if g == INF else f'{g:g}'    # noqa: E731
    return f'{h} γ=({f(g1)},{f(g2)},{f(g3)})'


def seleccion_val(n_por_clase=4, semilla=0):
    d = np.load('/mnt/homeGPU/slcerro/OpenOOD/features/completo/id_val.npz')
    rng = np.random.default_rng(semilla)
    idx = []
    for c in range(14):
        idx.extend(sorted(rng.choice(np.where(d['label'] == c)[0], n_por_clase, replace=False).tolist()))
    idx = np.array(idx)
    return idx, [str(p) for p in d['paths'][idx]], d['label'][idx]


def capas_sesgo(h):
    return {'H3': ['1.4', '1.8'], 'H4': ['1.2', '1.6']}.get(h, [])


class EntradasSesgo:
    """Captura entrada y salida de los módulos con sesgo para calcular, de forma analítica, la relevancia absorbida."""

    def __init__(self, m, h):
        self.m, self.h, self.d, self.hs = m, h, {}, []

    def __enter__(self):
        for n in capas_sesgo(self.h):
            mod = dict(self.m.named_modules())[n]
            self.hs.append(mod.register_forward_hook(lambda mo, a, s, n=n: self.d.__setitem__(n, (mo, a[0].detach(), s.detach()))))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


def relevancia_sesgo(h, ent, relevancias):
    """Relevancia absorbida por capa: H3, R_k b_k / stabilize(z_k); H4, R_i (|b_i| + eps) / (|x_i w_i| + |b_i| + eps)."""
    r = {}
    for n in capas_sesgo(h):
        mo, x, s = ent[n]
        R = relevancias[n]
        if h == 'H3':
            r[n] = float((R * mo.bias / stabilize(s, 1e-6)).sum())
        else:
            w, b = parametros_bn(mo)
            a = (x * w).abs()
            r[n] = float((R * (b.abs() + EPS_BN) / (a + b.abs() + EPS_BN)).sum())
    return r


# ---- Utilidades de evaluación (mismos cálculos que en la fase 3)
import copy                                                              # noqa: E402

from scipy.stats import rankdata, spearmanr                              # noqa: E402

LOTE = 64


@torch.no_grad()
def curva(model, xi, fondo, c, orden, L, n, P=16):
    x, sal, lote = xi.clone(), [model(xi)[:, c]], []
    for k in range(L):
        f, g = int(orden[k]) // n, int(orden[k]) % n
        x[:, :, f * P:(f + 1) * P, g * P:(g + 1) * P] = fondo[:, :, f * P:(f + 1) * P, g * P:(g + 1) * P]
        lote.append(x.clone())
        if len(lote) == LOTE or k == L - 1:
            sal.append(model(torch.cat(lote))[:, c])
            lote = []
    return torch.cat(sal).cpu().numpy()


def ordenes(h, P=16):
    n = h.shape[-1] // P
    s = torch.nn.functional.avg_pool2d(torch.as_tensor(h)[None, None], P)[0, 0].numpy()
    o = np.argsort(-s.ravel(), kind='stable')
    return {'MoRF': o, 'LeRF': o[::-1], 'volteada': (n - 1 - o // n) * n + (n - 1 - o % n)}, n


def aopc(y):
    y = np.atleast_2d(y)
    return (y[:, :1] - y[:, 1:]).mean(1)


def rho(a, b):
    return float(spearmanr(a.ravel(), b.ravel()).correlation)


def aleatorizado(m, nombres):
    m = copy.deepcopy(m)
    for n, mod in m.named_modules():
        if any(n == k or n.startswith(k + '.') for k in nombres) and getattr(mod, 'weight', None) is not None:
            with torch.no_grad():
                mod.weight.copy_(torch.randn_like(mod.weight) * mod.weight.std())
    return m
