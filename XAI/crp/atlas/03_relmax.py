# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 2: RelMax con el train completo. Uso: 03_relmax.py A|GxE. R = f_c(x) de la clase real (CRP por defecto). Reanudable por tramos de 4.800 imágenes."""
import json
import os
import sys
import time

from comun4 import CAPA, FASE4, ChannelConcept, DatasetRutas, FVGxE, FeatureVisualization, cargar, hacer_preprocesa, metodo, np, torch

nom = sys.argv[1]
LOTE, TRAMO = {'A': 16, 'GxE': 32}[nom], 4800
device = torch.device('cuda')
learn, model = cargar('convnext', device)
pp = hacer_preprocesa(learn, device)
m, at, comp = metodo(nom, model, device)
del model
d = np.load(os.path.join(FASE4, 'features', 'convnext_train.npz'))
ds = DatasetRutas(learn, [str(p) for p in d['paths']], d['label'])
S = os.path.join(FASE4, 'relmax', nom)
os.makedirs(S, exist_ok=True)
F = os.path.join(S, 'progreso.json')
fv = (FeatureVisualization if nom == 'A' else FVGxE)(at, ds, {CAPA: ChannelConcept()}, preprocess_fn=pp, path=S, max_target='sum', abs_norm=True)
prog = json.load(open(F)) if os.path.exists(F) else dict(hecho=0, chk={'r_max': [], 'a_max': [], 'r_stats': [], 'a_stats': []})
N, t = len(ds), time.perf_counter()
while prog['hecho'] < N:
    a, b = prog['hecho'], min(prog['hecho'] + TRAMO, N)
    c = fv.run_distributed(comp, a, b, batch_size=LOTE, checkpoint=10 ** 9)
    # Los nombres de CRP empiezan en 0 en cada tramo y terminan en el último índice: no se repiten entre tramos
    for k in prog['chk']:
        prog['chk'][k].extend(c[k])
    prog['hecho'] = b
    json.dump(prog, open(F, 'w'))
    print('TRAMO', nom, b, round(time.perf_counter() - t), 's', flush=True)
fv.collect_results(prog['chk'])
print('RELMAX_TERMINADO', nom, round(time.perf_counter() - t), 's', flush=True)
