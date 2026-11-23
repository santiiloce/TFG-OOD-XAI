# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Distribución de |denominador| de la regla de Ali en cada LayerNorm: z = γ (x − μ)/σ (sin β), el tensor que divide
ReglaLN (stabilize(z, ε)). 28 imágenes de la fase 1, solo forward. Por LN: mediana, p1, p5 de |z| y fracción por
debajo de 1e-6, 1e-2 y 1e-2·std(z) (los ε de T3, T3-eps-abs y T3-eps-rel)."""
import os
import sys

sys.path.insert(0, '/mnt/homeGPU/slcerro/OpenOOD/results_crp/fase2/scripts')
from comun2 import RESULTS, cargar, preprocesar, seleccion   # noqa: E402

import numpy as np   # noqa: E402
import pandas as pd   # noqa: E402
import torch   # noqa: E402
from timm.layers import LayerNorm2d   # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
learn, model = cargar('convnext', 'cpu')
_, rutas, _ = seleccion(2)
x = preprocesar(learn, rutas, 'cpu')
z_por_ln, rel_por_ln, ganchos = {}, {}, []
for nombre, mod in model.named_modules():
    if isinstance(mod, torch.nn.LayerNorm) and '.head.' not in nombre:
        def h(m, inp, out, nombre=nombre):
            xi = inp[0]
            dim = 1 if isinstance(m, LayerNorm2d) else -1
            forma = [1] * xi.dim()
            forma[dim] = -1
            sigma = (xi.var(dim=dim, keepdim=True, unbiased=False) + m.eps).sqrt()
            zs = (xi - xi.mean(dim=dim, keepdim=True)) / sigma * m.weight.reshape(forma)
            z = zs.abs()
            z_por_ln.setdefault(nombre, []).append(z.flatten().numpy())
            rel_por_ln.setdefault(nombre, []).append(float((z < 1e-2 * zs.std()).float().mean()))   # ε de T3-eps-rel: 1e-2·std(z), imagen a imagen
        ganchos.append(mod.register_forward_hook(h))
with torch.no_grad():
    for i in range(len(x)):
        model(x[i:i + 1])
filas = []
for nombre, zs in z_por_ln.items():
    z = np.concatenate(zs)
    tipo = 'stem' if 'stem' in nombre else ('downsample' if 'downsample' in nombre else 'bloque')
    filas.append(dict(ln=nombre.replace('0.model.', ''), tipo=tipo, n=z.size, mediana=np.median(z), p1=np.percentile(z, 1),
                      p5=np.percentile(z, 5), frac_lt_1e6=float((z < 1e-6).mean()), frac_lt_1e2=float((z < 1e-2).mean()),
                      frac_lt_1e2_std=float(np.mean(rel_por_ln[nombre]))))
d = pd.DataFrame(filas)
d.to_csv(os.path.join(RESULTS, 'ablacion_ln', 'denominadores_por_ln.csv'), index=False)
todo = np.concatenate([np.concatenate(v) for v in z_por_ln.values()])
print('todas las LN: mediana', np.median(todo), 'p1', np.percentile(todo, 1), 'p5', np.percentile(todo, 5),
      'frac<1e-6', (todo < 1e-6).mean(), 'frac<1e-2', (todo < 1e-2).mean(), flush=True)
pd.set_option('display.width', 200)
print(d.groupby('tipo')[['mediana', 'p1', 'p5', 'frac_lt_1e6', 'frac_lt_1e2']].median().to_string(), flush=True)
print(d.round(5).to_string(index=False), flush=True)
