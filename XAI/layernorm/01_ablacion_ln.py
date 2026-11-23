# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Ablación de la LayerNorm sobre `A` (28 imágenes de la fase 1), con relevancia bloque a bloque y LN a LN.
Variantes: A (Pass), T1 = A-LN (sin regla), T3 = A-reglaLN (ε = 1e-6·signo(z), el de stabilize por defecto),
T3-eps-abs (ε = 1e-2, la de la fase 2) y T3-eps-rel (ε = 1e-2·std(z) de la capa, nueva).
Además, medida de cancelación en el forward de cada LN (no depende de la regla)."""
import json
import os
import sys

sys.path.insert(0, '/mnt/homeGPU/slcerro/OpenOOD/results_crp/fase2/scripts')
from comun2 import BASE_A, RESULTS, cargar, preprocesar, seleccion   # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402
import torch   # noqa: E402
from crp.attribution import CondAttribution   # noqa: E402
from lrp_convnext_v2 import copia_para_lrp, make_composite   # noqa: E402
from lrp_fase2 import ReglaLNEstabilizada, composite_A_ln_est   # noqa: E402
from timm.layers import LayerNorm2d   # noqa: E402
from zennit.core import stabilize   # noqa: E402

SALIDA = os.path.join(RESULTS, 'ablacion_ln')


class ReglaLNRelativa(ReglaLNEstabilizada):
    """Como ReglaLNEstabilizada, con ε = factor·std(z) de la capa (imagen a imagen)."""

    def __init__(self, factor=1e-2):
        super().__init__(factor)

    def copy(self):
        return ReglaLNRelativa(self.epsilon)

    def backward(self, module, grad_input, grad_output):
        x = self.stored_tensors['input'][0].detach().requires_grad_()
        dim = 1 if isinstance(module, LayerNorm2d) else -1
        forma = [1] * x.dim()
        forma[dim] = -1
        with torch.autograd.enable_grad():
            sigma = (x.var(dim=dim, keepdim=True, unbiased=False) + module.eps).sqrt().detach()
            z = (x - x.mean(dim=dim, keepdim=True)) / sigma * module.weight.reshape(forma)
        eps = self.epsilon * float(z.detach().std())
        g = torch.autograd.grad(z, x, grad_outputs=grad_output[0] / stabilize(z, eps))[0]
        r = x * g
        return tuple(r if o.shape == r.shape else None for o in grad_input)


def composite_rel():
    comp = make_composite(**{**BASE_A, 'ln': None})
    comp.layer_map.insert(0, (torch.nn.LayerNorm, ReglaLNRelativa(1e-2)))
    return comp


VARIANTES = {
    'A': lambda: make_composite(**BASE_A),
    'T1': lambda: make_composite(**{**BASE_A, 'ln': None}),
    'T3': lambda: make_composite(**{**BASE_A, 'ln': 'regla_ln'}),
    'T3-eps-abs': lambda: composite_A_ln_est(1e-2),
    'T3-eps-rel': composite_rel,
}
PROFUNDIDAD = [3, 3, 27, 3]


def capas_ln():
    """(nombre LN, capa con R en la entrada de la LN, capa con R en la salida de la LN)."""
    p = '0.model.'
    out = [('stem.1', p + 'stem.0', p + 'stem.1')]
    for s, n in enumerate(PROFUNDIDAD):
        if s > 0:
            out.append((f'stages.{s}.downsample.0', p + f'stages.{s - 1}', p + f'stages.{s}.downsample.0'))
        for b in range(n):
            q = f'stages.{s}.blocks.{b}'
            out.append((q + '.norm', p + q + '.conv_dw', p + q + '.norm'))
    return out


def capas_bloque():
    p = '0.model.'
    out = ['stem']
    for s, n in enumerate(PROFUNDIDAD):
        if s > 0:
            out.append(f'stages.{s}.downsample')
        out += [f'stages.{s}.blocks.{b}' for b in range(n)]
    return [(c, p + c) for c in out]


def cancelacion(model, x):
    """Por LN e imagen: fracción de salidas con |y_i| < 1 % de la media de |x| (misma posición) y, aparte, con
    |z_i| < 1 % de la media de |z| (z = parte sin β, el denominador de la regla)."""
    filas, ganchos = [], []
    for nombre, mod in model.named_modules():
        if isinstance(mod, torch.nn.LayerNorm) and not nombre.startswith('0.model.head'):
            def h(m, inp, out, nombre=nombre):
                xi = inp[0]
                dim = 1 if isinstance(m, LayerNorm2d) else -1
                forma = [1] * xi.dim()
                forma[dim] = -1
                sigma = (xi.var(dim=dim, keepdim=True, unbiased=False) + m.eps).sqrt()
                z = (xi - xi.mean(dim=dim, keepdim=True)) / sigma * m.weight.reshape(forma)
                fy = (out.abs() < 0.01 * xi.abs().mean(dim=dim, keepdim=True)).float().mean().item()
                fz = (z.abs() < 0.01 * z.abs().mean(dim=dim, keepdim=True)).float().mean().item()
                fyz = (out.abs() < 0.01 * out.abs().mean(dim=dim, keepdim=True)).float().mean().item()
                filas.append({'ln': nombre.replace('0.model.', ''), 'frac_|y|<1%media|x|': fy,
                              'frac_|z|<1%media|z|': fz, 'frac_|y|<1%media|y|': fyz})
            ganchos.append(mod.register_forward_hook(h))
    with torch.no_grad():
        for i in range(len(x)):
            n0 = len(filas)
            model(x[i:i + 1])
            for f in filas[n0:]:
                f['imagen'] = i
    for g in ganchos:
        g.remove()
    return pd.DataFrame(filas)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn, model = cargar('convnext', device)
    _, rutas, _ = seleccion(2)
    x = preprocesar(learn, rutas, device)
    with torch.no_grad():
        pred = model(x).argmax(1).cpu().numpy()
    m = copia_para_lrp(model).to(device)
    lns, bloques = capas_ln(), capas_bloque()
    registro = sorted({c for _, a, b in lns for c in (a, b)} | {c for _, c in bloques})
    attribution = CondAttribution(m, no_param_grad=True)
    por_imagen, por_bloque, por_ln = [], [], []
    for v, hacer in VARIANTES.items():
        comp = hacer()
        for i in range(len(x)):
            c = int(pred[i])

            def seed(p):
                s = torch.zeros_like(p)
                s[:, c] = p[:, c]
                return s
            xi = x[i:i + 1].clone().requires_grad_(True)
            a = attribution(xi, [{'y': [c]}], comp, record_layer=registro, init_rel=seed)
            fc = float(a.prediction[0, c].detach())
            por_imagen.append({'variante': v, 'imagen': i, 'clase': c, 'f_c': fc,
                               'cociente': float(a.heatmap.sum()) / fc, 'ruta': rutas[i]})
            for nombre, capa in bloques:
                r = a.relevances[capa]
                por_bloque.append({'variante': v, 'imagen': i, 'capa': nombre, 'R/f_c': float(r.sum()) / fc,
                                   '|R|/f_c': float(r.abs().sum()) / abs(fc)})
            for nombre, ent, sal in lns:
                re, rs = a.relevances[ent], a.relevances[sal]
                por_ln.append({'variante': v, 'imagen': i, 'ln': nombre, 'R_sal/f_c': float(rs.sum()) / fc,
                               'R_ent/f_c': float(re.sum()) / fc, '|R_ent|/f_c': float(re.abs().sum()) / abs(fc),
                               '|R_sal|/f_c': float(rs.abs().sum()) / abs(fc)})
        q = np.array([f['cociente'] for f in por_imagen if f['variante'] == v])
        print(f'{v:>11}: mediana {np.median(q):.4g} min {q.min():.4g} max {q.max():.4g} '
              f'p5 {np.percentile(q, 5):.4g} p95 {np.percentile(q, 95):.4g}', flush=True)
    pd.DataFrame(por_imagen).to_csv(os.path.join(SALIDA, 'conservacion_por_imagen.csv'), index=False)
    pd.DataFrame(por_bloque).to_csv(os.path.join(SALIDA, 'relevancia_por_bloque.csv'), index=False)
    pd.DataFrame(por_ln).to_csv(os.path.join(SALIDA, 'relevancia_por_ln.csv'), index=False)
    cancelacion(model, x).to_csv(os.path.join(SALIDA, 'cancelacion_por_ln.csv'), index=False)
    with open(os.path.join(SALIDA, 'logs', 'pred.json'), 'w') as f:
        json.dump({'pred': pred.tolist(), 'rutas': rutas}, f, indent=1)


if __name__ == '__main__':
    main()
