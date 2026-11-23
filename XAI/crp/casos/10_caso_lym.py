# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Caso nuevo: OOD NCT-CRC LYM aceptada como ID por STOOD-X en M (regla de ood_aceptada de coherencia_f9/M/scripts/casos_M.py:
la aceptada más cercana a la mediana de las aceptadas, desde arriba). Vecinos y canales como en casos_M.py. Solo CPU.
Escribe ../nuevos/caso_lym.json.  Uso: /mnt/homeGPU/slcerro/fastai_env_ood/bin/python 10_caso_lym.py"""
import json, os
import numpy as np
import pandas as pd

R = '/mnt/homeGPU/slcerro/OpenOOD'
F9M = f'{R}/results_ood_convnext/espacio_principal'
S = f'{R}/results_crp/explicaciones_f9/M/nuevos'
os.makedirs(S, exist_ok=True)
CLASES = pd.read_csv(f'{R}/results_crp/fase4/explicaciones/caracteristicas_por_clase.csv')['clase'].tolist()
def fM(n, k): return np.load(f'{F9M}/features_M/{n}_{k}.npy', allow_pickle=True)

st = dict(np.load(f'{F9M}/espacio_M/partes_stoodx_100/afin_nct_lym.npz', allow_pickle=True))
te = np.load(f'{F9M}/espacio_M/partes_stoodx_100/id_test.npz', allow_pickle=True)['conf']
M, L = fM('afin_nct_lym', 'M'), fM('afin_nct_lym', 'logits').astype(np.float64)
assert (fM('afin_nct_lym', 'paths') == st['paths']).all() and (fM('afin_nct_lym', 'pred') == st['pred']).all()
thr = float(np.percentile(te, 5))
acc = st['conf'] >= thr
pool = np.where(acc)[0]
med = float(np.median(st['conf'][pool]))
arriba = pool[st['conf'][pool] >= med]
i = int(arriba[np.argmin(st['conf'][arriba] - med)])
c = int(st['pred'][i])

def nrm(X): X = np.asarray(X, np.float64); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
Mtr, p9tr = fM('id_train', 'M'), fM('id_train', 'pred')
mu, sd = Mtr.mean(0), Mtr.std(0)
idx = np.where(p9tr == c)[0]
d = 1 - nrm(Mtr[idx]) @ nrm(M[i][None])[0]
o = np.argsort(d, kind='stable')[:5]
assert (idx[o] == st['vecinos_idx'][i]).all()          # = vecinos de la puntuación de la fase 9b
vec, dv = idx[o[:3]].tolist(), d[o[:3]].tolist()
Mx = np.vstack([M[i][None], Mtr[vec]])
def presencia(z):
    P = np.abs((Mx - mu) / sd if z else Mx).sum(0); top = np.argsort(-P)[:4]; return top.tolist(), P[top].tolist()
can, pres = presencia(False); canz, presz = presencia(True)
e = np.exp(L[i] - L[i].max())
caso = dict(grupo='ood_aceptada', conjunto='nct_lym', indice=i, ruta=str(st['paths'][i]), clase_real='OOD',
            clase_pred=CLASES[c], c=c, softmax=float((e / e.sum()).max()), vecinos=vec, dist_vecinos=dv,
            canales=can, presencia=pres, canales_z=canz, presencia_z=presz,
            stoodx_menos_z=float(st['conf'][i]), stoodx_p=float(st['p'][i]), saturada=bool(st['saturada'][i]),
            percentil_id_test=float((te <= st['conf'][i]).mean() * 100), decision='ID (aceptada)',
            motivo='nueva: LYM aceptada más cercana a la mediana desde arriba',
            info=dict(n=int(len(acc)), n_pool=int(len(pool)), frac_aceptadas=float(acc.mean()), mediana_pool=med, umbral=thr))
json.dump(caso, open(f'{S}/caso_lym.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(caso, indent=1, ensure_ascii=False))
