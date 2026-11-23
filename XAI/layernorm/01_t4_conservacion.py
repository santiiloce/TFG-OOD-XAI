# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Pasos 1 y 2: T4 sobre las 28 imágenes de la fase 1. Equivalencia de logits, regla de cada LayerNorm, conservación de
A y T4 (por imagen, por etapa y por bloque). Termina con código 3 si se cumple la condición de parada (decisiones.md).
    python 01_t4_conservacion.py --hilos 8 [--limite 2]      # --limite: solo mide tiempo, no escribe resultados"""
import argparse
import json
import os
import sys
import time

from comun_t4 import (T4, cargar, composite_final, copia_para_lrp, composite_t4, fusionar_ln, preprocesar,
                      reglas_registradas, resumen_cociente, seleccion, semilla_clase)

import numpy as np
import pandas as pd
import torch
from crp.attribution import CondAttribution

ap = argparse.ArgumentParser()
ap.add_argument('--hilos', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
ap.add_argument('--limite', type=int, default=0)
args = ap.parse_args()
torch.set_num_threads(args.hilos)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('dispositivo', device, 'hilos', torch.get_num_threads(), flush=True)
SALIDA = os.path.join(T4, 'conservacion')
os.makedirs(SALIDA, exist_ok=True)

learn, model = cargar('convnext', device)
_, rutas, _ = seleccion(2)
if args.limite:
    rutas = rutas[:args.limite]
x = preprocesar(learn, rutas, device)
with torch.no_grad():
    logits = torch.cat([model(x[i:i + 7]) for i in range(0, len(x), 7)])
pred = logits.argmax(1).cpu().numpy()

mA = copia_para_lrp(model).to(device)
mT, trat = fusionar_ln(copia_para_lrp(model).to(device))
compA, compT = composite_final(), composite_t4()
equiv = {}
with torch.no_grad():
    equiv['T4_sin_hooks'] = float(max((mT(x[i:i + 7]) - logits[i:i + 7]).abs().max() for i in range(0, len(x), 7)))
    equiv['A_sin_hooks'] = float(max((mA(x[i:i + 7]) - logits[i:i + 7]).abs().max() for i in range(0, len(x), 7)))
for nom, m, comp in [('T4_con_composite', mT, compT), ('A_con_composite', mA, compA)]:
    with comp.context(m) as mh:
        equiv[nom] = float(max((mh(x[i:i + 7].clone().requires_grad_(True)).detach() - logits[i:i + 7]).abs().max()
                               for i in range(0, len(x), 7)))
print('equivalencia', json.dumps(equiv), flush=True)
reglas = {'A': reglas_registradas(mA, compA, x[:1]), 'T4': reglas_registradas(mT, compT, x[:1])}

p = '0.model.'
CAPAS = {'stem': p + 'stem'}
for s, n in enumerate([3, 3, 27, 3]):
    if s > 0:
        CAPAS[f'stages.{s}.downsample'] = p + f'stages.{s}.downsample'
    for b in range(n):
        CAPAS[f'stages.{s}.blocks.{b}'] = p + f'stages.{s}.blocks.{b}'
CAPAS.update({'cabeza.ConcatPool': '1.0', 'cabeza.Linear1': '1.4', 'cabeza.Linear2': '1.8'})

por_img, por_capa, tiempos = [], [], {}
for nom, m, comp in [('A', mA, compA), ('T4', mT, compT)]:
    at = CondAttribution(m, no_param_grad=True)
    ini = time.perf_counter()
    for i in range(len(x)):
        c = int(pred[i])
        a = at(x[i:i + 1].clone().requires_grad_(True), [{'y': [c]}], comp, init_rel=semilla_clase(c),
               record_layer=list(CAPAS.values()))
        fc = float(a.prediction[0, c].detach())
        hm = a.heatmap[0].detach()
        por_img.append(dict(variante=nom, imagen=i, clase=c, f_c=fc, cociente=float(hm.sum()) / fc,
                            mapa_finito=bool(torch.isfinite(hm).all()), ruta=rutas[i]))
        for k, capa in CAPAS.items():
            r = a.relevances[capa]
            por_capa.append(dict(variante=nom, imagen=i, capa=k, R_f=float(r.sum()) / fc, absR_f=float(r.abs().sum()) / abs(fc)))
    tiempos[nom] = (time.perf_counter() - ini) / len(x)
    print(nom, f'{tiempos[nom]:.2f} s/imagen', flush=True)

if args.limite:
    print('SOLO_TIEMPO', json.dumps(tiempos), flush=True)
    sys.exit(0)

d = pd.DataFrame(por_img)
d.to_csv(os.path.join(SALIDA, 'conservacion_por_imagen.csv'), index=False)
pc = pd.DataFrame(por_capa)
pc.to_csv(os.path.join(SALIDA, 'relevancia_por_capa.csv'), index=False)
res = pd.DataFrame([dict(variante=v, **resumen_cociente(g.cociente)) for v, g in d.groupby('variante', sort=False)])
res.to_csv(os.path.join(SALIDA, 'conservacion_resumen.csv'), index=False)
med = pc.groupby(['capa', 'variante'], sort=False)[['R_f', 'absR_f']].median().unstack()
med.to_csv(os.path.join(SALIDA, 'relevancia_por_capa_mediana.csv'))
t = res.set_index('variante').loc['T4']
parar = bool(t.mediana < 0.9 or t['min'] < -1 or t['max'] > 2)
json.dump(dict(equivalencia=equiv, tratamiento_ln=trat, reglas_registradas=reglas, s_por_imagen=tiempos,
               condicion_parada=parar), open(os.path.join(SALIDA, 'comprobaciones.json'), 'w'), indent=1, ensure_ascii=False)
pd.set_option('display.width', 250)
print(res.round(4).to_string(index=False))
print(med.round(4).to_string())
if equiv['T4_sin_hooks'] >= 1e-4:
    print('AVISO: |Δ logit| de T4 >= 1e-4', flush=True)
if parar:
    print('PARADA: T4 no cumple la condición de conservación', flush=True)
    sys.exit(3)
print('PASO2_TERMINADO', flush=True)
