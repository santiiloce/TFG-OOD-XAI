# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4 (CPU): estabilidad del top-8 de cada concepto entre semillas, sobre sus 50 candidatas. Solape bajo: ≤ 4 de 8 (fijado en decisiones.md)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
O = os.path.join(R, 'fase10_eg', 'paso4', sys.argv[1])
tr = np.load(os.path.join(R, 'fase4', 'features', 'convnext_train.npz'))
cand = {int(k): v for k, v in json.load(open(os.path.join(O, 'candidatas50.json'))).items()}
c1 = {}
for f in glob.glob(os.path.join(O, 'C_semilla1_parte*.npz')):
    z = np.load(f)
    c1.update({int(i): c for i, c in zip(z['ids'], z['C'])})
C0 = {}
for f in glob.glob(os.path.join(R, 'fase10_eg', 'paso4', 'C', 'train_*_s0.npz')):
    z = np.load(f)
    C0.update({int(i): c for i, c in zip(z['ids'], z['C']) if int(i) in {j for v in cand.values() for j in v}})
fm = os.path.join(R, 'fase10_eg', 'paso4', 'muestra', 'C_muestra_faltantes_s0.npz')     # C de la muestra calculadas aparte (17-11)
if os.path.exists(fm):
    z = np.load(fm)
    C0.update({int(i): c for i, c in zip(z['ids'], z['C']) if int(i) in {j for v in cand.values() for j in v}})
con = pd.read_csv(os.path.join(O, 'conceptos.csv'))
if 'metodo' in con.columns:
    con = con[con.metodo.str.startswith('EG')]
filas = []
for k, ids in cand.items():
    n0 = np.array([C0[i][k] / np.abs(C0[i]).sum() for i in ids])
    n1 = np.array([c1[i][k] / np.abs(c1[i]).sum() for i in ids])
    t0, t1 = set(np.array(ids)[np.argsort(-n0, kind='stable')[:8]]), set(np.array(ids)[np.argsort(-n1, kind='stable')[:8]])
    filas.append(dict(canal=k, clases=';'.join(con[con.canal == k].clase), solape_top8=len(t0 & t1), spearman_50=float(spearmanr(n0, n1).correlation)))
d = pd.DataFrame(filas)
d['bajo'] = d.solape_top8 <= 4
d.to_csv(os.path.join(O, 'estabilidad_top8.csv'), index=False)
res = dict(conceptos=len(d), solape_mediana=float(d.solape_top8.median()), solape_min=int(d.solape_top8.min()), solape_max=int(d.solape_top8.max()),
           n_bajo_le4=int(d.bajo.sum()), spearman_mediana=float(d.spearman_50.median()), spearman_min=float(d.spearman_50.min()))
json.dump(res, open(os.path.join(O, 'estabilidad_resumen.json'), 'w'), indent=1)
print(d.to_string()); print(res)
