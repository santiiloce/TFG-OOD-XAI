# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4 (cambio del 17-11, muestra estratificada): (1) C de EG para las imágenes de la muestra que faltan, con la misma reserva y el
mismo sorteo (s = 0, default_rng([0, i])) que 04_relmax_bloque.py; (2) relevancia de canal de A (CRP, composite_final, R = f_c de la
clase real, sin init_rel, como el RelMax de la fase 4) para las 4.654 imágenes de la muestra.
Uso: 10_muestra_C_A.py [medir]   (medir: 20 imágenes de cada parte y estimación)"""
import os
import sys
import time
from pathlib import Path

from comun10 import CAPA, F10, RESULTS, cargar, cargar_reserva, conductancia, sorteo_reserva

import numpy as np
import torch

sys.path.insert(0, os.path.join(RESULTS, 'fase5', 'scripts'))
from composite_final import composite_final, preparar_modelo_final   # noqa: E402
from crp.attribution import CondAttribution                          # noqa: E402
from crp.concepts import ChannelConcept                              # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cuda')
MEDIR = len(sys.argv) > 1 and sys.argv[1] == 'medir'
S = os.path.join(F10, 'paso4', 'muestra')
tr = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))
mu, falta = np.load(os.path.join(S, 'muestra_idx.npy')), np.load(os.path.join(S, 'por_calcular_idx.npy'))
if MEDIR:                                   # medir [nC nA]: imágenes de cada parte (por defecto 20 y 20)
    nC, nA = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (20, 20)
    mu, falta = mu[:nA], falta[:nC]
learn, model = cargar('convnext', device)
hilos = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
t0 = time.perf_counter()
# (1) C de EG para las que faltan
fC = os.path.join(S, 'C_muestra_faltantes_s0.npz')
if MEDIR or not os.path.exists(fC):
    pool = cargar_reserva(learn, device)
    print('reserva', round(time.perf_counter() - t0), 's', flush=True)
    C, FC, j, t1 = np.zeros((len(falta), 768), np.float32), np.zeros(len(falta), np.float32), 0, time.perf_counter()
    for b in learn.dls.test_dl([Path(str(p)) for p in tr['paths'][falta]], bs=16, num_workers=hilos):
        xb = b[0].as_subclass(torch.Tensor).to(device)
        for u in range(len(xb)):
            i, c = int(falta[j]), int(tr['label'][falta[j]])
            sel, al = sorteo_reserva('train', i, 0, device)
            C[j] = conductancia(model, xb[u:u + 1], c, pool[sel], al, con_fref=False)[0].cpu().numpy()
            with torch.no_grad():
                FC[j] = float(model(xb[u:u + 1])[0, c])
            j += 1
        if j % 160 < 16:
            print('C', j, '/', len(falta), round(time.perf_counter() - t1), 's', flush=True)
    assert j == len(falta)
    tC = (time.perf_counter() - t1) / len(falta)
    del pool
    torch.cuda.empty_cache()
    if not MEDIR:
        np.savez(fC, C=C, f_c=FC, ids=falta)
    print(f'C: {tC:.2f} s/imagen', flush=True)
# (2) Relevancia de canal de A en toda la muestra
mA = preparar_modelo_final(model, device)
at, comp = CondAttribution(mA, no_param_grad=True), composite_final()
RA, j, t2 = np.zeros((len(mu), 768), np.float32), 0, time.perf_counter()
for b in learn.dls.test_dl([Path(str(p)) for p in tr['paths'][mu]], bs=16, num_workers=hilos):
    x = b[0].as_subclass(torch.Tensor).to(device).clone().requires_grad_(True)
    ys = tr['label'][mu[j:j + len(x)]]
    a = at(x, [{'y': [int(y)]} for y in ys], comp, record_layer=[CAPA])          # sin init_rel: R = f_c(x), como en el RelMax de la fase 4
    RA[j:j + len(x)] = ChannelConcept().attribute(a.relevances[CAPA].detach(), abs_norm=False).cpu().numpy()
    j += len(x)
assert j == len(mu)
tA = (time.perf_counter() - t2) / len(mu)
print(f'A: {tA:.3f} s/imagen', flush=True)
if MEDIR:
    n_f, n_m = len(np.load(os.path.join(S, 'por_calcular_idx.npy'))), len(np.load(os.path.join(S, 'muestra_idx.npy')))
    print(f'ESTIMACION: C {n_f} × {tC:.2f} s = {n_f * tC / 60:.0f} min; A {n_m} × {tA:.3f} s = {n_m * tA / 60:.0f} min; + reserva ≈ 1 min', flush=True)
    sys.exit(0)
np.savez(os.path.join(S, 'R_A_muestra.npz'), R=RA, ids=mu)
print('MUESTRA_TERMINADA', round(time.perf_counter() - t0), 's', flush=True)
sys.stdout.flush()
os._exit(0)
