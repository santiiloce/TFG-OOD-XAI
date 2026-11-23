# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4 (GPU): (1) estabilidad: C con la semilla 1 para las 50 candidatas de cada concepto; (2) mapas φ^k (float32) de las 8 referencias
de cada concepto, con la clase real de la referencia y el mismo sorteo (semilla 0) que su C.
Uso: 06_estabilidad_mapas.py <dir de 05 (resultados|prueba_parcial)> [tarea ntareas]   (reparte las imágenes entre tareas)"""
import json
import os
import sys
import time

from comun10 import F10, RESULTS, cargar, cargar_reserva, conductancia, mapas_canal, preprocesar, sorteo_reserva

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
O = os.path.join(F10, 'paso4', sys.argv[1])
t_, nt = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)), int(os.environ.get('NTAREAS', 1)))
tr = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))
learn, model = cargar('convnext', device)
pool = cargar_reserva(learn, device)
cand = {int(k): v for k, v in json.load(open(os.path.join(O, 'candidatas50.json'))).items()}
ids = sorted({i for v in cand.values() for i in v})[t_::nt]
t0 = time.perf_counter()
C1 = np.zeros((len(ids), 768), np.float32)
for j, i in enumerate(ids):
    x = preprocesar(learn, [str(tr['paths'][i])], device)
    sel, al = sorteo_reserva('train', i, 1, device)
    C1[j] = conductancia(model, x, int(tr['label'][i]), pool[sel], al, con_fref=False)[0].cpu().numpy()
    if j % 100 == 0:
        print('semilla 1', j, '/', len(ids), round(time.perf_counter() - t0), 's', flush=True)
np.savez(os.path.join(O, f'C_semilla1_parte{t_}.npz'), C=C1, ids=np.array(ids))
print('estabilidad', len(ids), 'imágenes', round(time.perf_counter() - t0), 's', flush=True)
ref = pd.read_csv(os.path.join(O, 'referencias.csv'))
mine = ref.iloc[t_::nt]
mapas = {}
for _, r in mine.iterrows():
    i = int(r.idx_train)
    x = preprocesar(learn, [str(tr['paths'][i])], device)
    sel, al = sorteo_reserva('train', i, 0, device)
    mapas[f'{r.clase_figura}|{r.puesto_concepto}|{r.canal}|{r.puesto_ref}|{i}'] = mapas_canal(model, x, int(tr['label'][i]), pool[sel], al, [int(r.canal)])[0].cpu().numpy().astype(np.float32)
np.savez_compressed(os.path.join(O, f'mapas_referencias_parte{t_}.npz'), **mapas)
print('mapas', len(mapas), round(time.perf_counter() - t0), 's', flush=True)
# Mapas condicionados de A (CRP) de las referencias de A, si existen (muestra estratificada; para comparar en las figuras)
fA = os.path.join(O, 'referencias_A.csv')
if os.path.exists(fA):
    sys.path.insert(0, os.path.join(RESULTS, 'fase5', 'scripts'))
    from composite_final import composite_final, preparar_modelo_final   # noqa: E402
    from crp.attribution import CondAttribution                          # noqa: E402
    from comun10 import CAPA, semilla_clase                              # noqa: E402
    mA, compA = preparar_modelo_final(model, device), composite_final()
    atA = CondAttribution(mA, no_param_grad=True)
    mapasA = {}
    for _, r in pd.read_csv(fA).iloc[t_::nt].iterrows():
        i, c = int(r.idx_train), int(tr['label'][int(r.idx_train)])
        x = preprocesar(learn, [str(tr['paths'][i])], device).clone().requires_grad_(True)
        a = atA(x, [{'y': [c], CAPA: [int(r.canal)]}], compA, init_rel=semilla_clase(c))
        mapasA[f'{r.clase_figura}|{r.puesto_concepto}|{r.canal}|{r.puesto_ref}|{i}'] = a.heatmap[0].detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(O, f'mapas_referencias_A_parte{t_}.npz'), **mapasA)
    print('mapas A', len(mapasA), round(time.perf_counter() - t0), 's', flush=True)
print('ESTABILIDAD_MAPAS_TERMINADO', flush=True)
sys.stdout.flush()
os._exit(0)
