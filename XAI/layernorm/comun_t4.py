# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""T4 (parte afín de la LayerNorm, centrado incluido, fusionada en la capa lineal siguiente), Expected Gradients e IG.
Importa sin modificarlos los módulos de las fases 1, 2, 3 y 5."""
import os
import sys

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
T4 = os.path.join(RESULTS, 't4_ig')
sys.path.insert(0, os.path.join(RESULTS, 'fase5', 'scripts'))

import numpy as np                                                           # noqa: E402
import torch                                                                 # noqa: E402
from composite_final import composite_final, preparar_modelo_final           # noqa: E402,F401
from lrp_fase5 import (aleatorizado, aopc, boot_ic, cargar, curva, ordenes,  # noqa: E402,F401
                       preprocesar, rho, seleccion, semilla_clase)
from lrp_convnext_v2 import copia_para_lrp                                   # noqa: E402
from timm.layers import LayerNorm2d                                          # noqa: E402
from timm.models.convnext import ConvNeXtBlock, ConvNeXtStage                # noqa: E402
from zennit.rules import Pass                                                # noqa: E402

PROFUNDIDAD = [3, 3, 27, 3]


class NormSigma(torch.nn.Module):
    """Lo que queda de la LayerNorm en T4: n = x / sigma, sigma = sqrt(Var(x) + eps) por posición. Regla: Pass."""

    def __init__(self, ln):
        super().__init__()
        self.eps, self.dim = ln.eps, (1 if isinstance(ln, LayerNorm2d) else -1)

    def forward(self, x):
        return x / (x.var(dim=self.dim, keepdim=True, unbiased=False) + self.eps).sqrt()


def _centrado(w, gamma, dim_in):
    """W' = W diag(gamma) C a lo largo de la dimensión de entrada dim_in: W'_i = W_i gamma_i - media_j(W_j gamma_j)."""
    forma = [1] * w.dim()
    forma[dim_in] = -1
    wg = w * gamma.reshape(forma)
    return wg - wg.mean(dim=dim_in, keepdim=True)


@torch.no_grad()
def fusionar_ln(m):
    """Aplica T4 sobre m (ya copiado con copia_para_lrp); lo modifica y devuelve (m, tratamiento por LayerNorm)."""
    trat = {}
    for nombre, mod in list(m.named_modules()):
        if isinstance(mod, ConvNeXtBlock):
            ln, fc1 = mod.norm, mod.mlp.fc1
            assert isinstance(ln, torch.nn.LayerNorm) and not isinstance(ln, LayerNorm2d) and isinstance(fc1, torch.nn.Linear)
            nueva = torch.nn.Linear(fc1.in_features, fc1.out_features, bias=True).to(fc1.weight.device)
            nueva.weight.copy_(_centrado(fc1.weight, ln.weight, 1))
            nueva.bias.copy_(fc1.weight @ ln.bias + (fc1.bias if fc1.bias is not None else 0))
            mod.mlp.fc1, mod.norm = nueva.eval(), NormSigma(ln)
            trat[nombre + '.norm'] = f'fusionada en {nombre}.mlp.fc1 (Linear {fc1.in_features}->{fc1.out_features}); queda x/σ con Pass'
        elif isinstance(mod, ConvNeXtStage) and isinstance(mod.downsample, torch.nn.Sequential):
            ln, conv = mod.downsample[0], mod.downsample[1]
            assert isinstance(ln, LayerNorm2d) and isinstance(conv, torch.nn.Conv2d)
            nueva = torch.nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                                    bias=True).to(conv.weight.device)
            nueva.weight.copy_(_centrado(conv.weight, ln.weight, 1))
            nueva.bias.copy_(conv.weight.sum((2, 3)) @ ln.bias + (conv.bias if conv.bias is not None else 0))
            mod.downsample[0], mod.downsample[1] = NormSigma(ln), nueva.eval()
            trat[nombre + '.downsample.0'] = f'fusionada en {nombre}.downsample.1 (Conv2d {conv.kernel_size}, paso {conv.stride}); queda x/σ con Pass'
    for nombre, mod in m.named_modules():
        if isinstance(mod, torch.nn.LayerNorm):
            trat[nombre] = 'Pass (sin fusionar)' + ('; no se ejecuta: TimmBody usa forward_features' if '.head.' in nombre else '')
    return m, trat


def preparar_t4(model, device=None, ya_copiado=False):
    m = model if ya_copiado else copia_para_lrp(model)
    if device is not None:
        m = m.to(device)
    return fusionar_ln(m)


def composite_t4():
    comp = composite_final()
    comp.layer_map.insert(0, (NormSigma, Pass()))
    return comp


def reglas_registradas(m, comp, x1):
    """Para cada LayerNorm o NormSigma, qué regla registra el composite (comprobación de que ninguna queda sin regla)."""
    out = {}
    with comp.context(m) as mh:
        for n, mod in mh.named_modules():
            if isinstance(mod, (torch.nn.LayerNorm, NormSigma)):
                hooks = list(mod._forward_hooks.values()) + list(mod._forward_pre_hooks.values()) + list(mod._backward_hooks.values())
                reglas = {type(getattr(h, '__self__', h)).__name__ for h in hooks}
                out[n] = sorted(reglas) if hooks else []
    return out


# ---- Expected Gradients e Integrated Gradients (a mano)

def grad_logit(model, xs, c, lote=16):
    g = []
    for i in range(0, len(xs), lote):
        xb = xs[i:i + lote].detach().clone().requires_grad_(True)
        g.append(torch.autograd.grad(model(xb)[:, c].sum(), xb)[0])
    return torch.cat(g)


def eg(model, xi, c, refs, alfas, lote=16):
    """Expected Gradients: media de (x - x~) * grad f_c(x~ + a (x - x~)). Devuelve mapa (suma RGB), suma total y E[f(x~)]."""
    a = alfas.reshape(-1, 1, 1, 1)
    puntos = refs + a * (xi - refs)
    phi = ((xi - refs) * grad_logit(model, puntos, c, lote)).mean(0)
    with torch.no_grad():
        fref = torch.cat([model(refs[i:i + lote])[:, c] for i in range(0, len(refs), lote)]).mean()
    return phi.sum(0).cpu().numpy(), float(phi.sum()), float(fref)


def ig(model, xi, c, base, pasos=32, lote=16):
    """IG con regla del punto medio, alfa = (k + 0.5) / pasos."""
    a = ((torch.arange(pasos, device=xi.device, dtype=xi.dtype) + 0.5) / pasos).reshape(-1, 1, 1, 1)
    phi = ((xi - base) * grad_logit(model, base + a * (xi - base), c, lote).mean(0, keepdim=True))[0]
    with torch.no_grad():
        fref = float(model(base)[0, c])
    return phi.sum(0).cpu().numpy(), float(phi.sum()), fref


def gxe(model, xi, c):
    xi = xi.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(model(xi)[0, c], xi)[0]
    return (g * xi).sum(1)[0].detach().cpu().numpy()


def sorteo_eg(n_img, k=32, semilla=0):
    """Índices de train y alfas de EG, fijados de antemano (decisiones.md)."""
    rng = np.random.default_rng(semilla)
    n_train = len(np.load('/mnt/homeGPU/slcerro/OpenOOD/features/completo/id_train.npz')['label'])
    idx, alf = [], []
    for _ in range(n_img):
        idx.append(rng.integers(0, n_train, k))
        alf.append(rng.random(k))
    return np.stack(idx), np.stack(alf)


def rutas_train(indices):
    d = np.load('/mnt/homeGPU/slcerro/OpenOOD/features/completo/id_train.npz')
    return [str(p) for p in d['paths'][indices]]


def resumen_cociente(q):
    q = np.asarray(q, float)
    return dict(n=len(q), media=q.mean(), mediana=np.median(q), min=q.min(), p5=np.percentile(q, 5),
                p95=np.percentile(q, 95), max=q.max(), fuera_05_15=int(((q < 0.5) | (q > 1.5)).sum()))
