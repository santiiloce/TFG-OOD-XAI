# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4: conductancia C (768) de EG para train (90.800) y test (12.618), con la clase real, por bloques de 2.000 imágenes.
Referencias: reserva fija de 4.096 imágenes de train (default_rng(10)); para la imagen i, 128 índices de la reserva y 128 α con default_rng([s, i]).
Uso: 04_relmax_bloque.py <bloque|medir> [s]   (bloque 0-45 = train, 46-52 = test; s = semilla, 0 por defecto)"""
import os
import sys
import time

from comun10 import F10, RESULTS, cargar, cargar_refs, conductancia, rutas_train

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cuda')
TAM, P, K = 2000, 4096, 128
arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('SLURM_ARRAY_TASK_ID')
s = int(sys.argv[2]) if len(sys.argv) > 2 else 0
S = os.path.join(F10, 'paso4', 'C')
os.makedirs(S, exist_ok=True)
tr = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))
te = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_test.npz'))
medir = arg == 'medir'
if medir:
    conj, ids = 'train', np.arange(100)
else:
    b = int(arg)
    nb_tr = int(np.ceil(len(tr['label']) / TAM))
    conj, base = ('train', b) if b < nb_tr else ('test', b - nb_tr)
    n_tot = len((tr if conj == 'train' else te)['label'])
    ids = np.arange(base * TAM, min((base + 1) * TAM, n_tot))
D = tr if conj == 'train' else te
f = os.path.join(S, f'{conj}_{ids[0]:06d}_{ids[-1]:06d}_s{s}.npz')
if os.path.exists(f) and not medir:
    print('YA_HECHO', f, flush=True)
    sys.exit(0)
if not medir:                          # candado: evita que dos GPU calculen el mismo bloque (atenea asciende, dionisio/zeus descienden)
    try:
        os.close(os.open(os.path.join(S, f'bloque{arg}_s{s}.lock'), os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print('EN_CURSO_EN_OTRA_GPU', arg, flush=True)
        sys.exit(0)
learn, model = cargar('convnext', device)
pool_idx = np.sort(np.random.default_rng(10).choice(len(tr['label']), P, replace=False))
t0 = time.perf_counter()
fpool = os.path.join(F10, 'paso4', 'reserva_idx.npy')
if not os.path.exists(fpool):
    np.save(fpool, pool_idx)
pool = torch.cat([cargar_refs(learn, [str(p) for p in tr['paths'][pool_idx[a:a + 512]]], 'cpu') for a in range(0, P, 512)])   # CPU, fp32
print('reserva', tuple(pool.shape), round(time.perf_counter() - t0), 's', flush=True)
pool = pool.to(device)
from pathlib import Path    # noqa: E402
dl = learn.dls.test_dl([Path(str(p)) for p in D['paths'][ids]], bs=16, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
C, FC, j, t1 = np.zeros((len(ids), 768), np.float32), np.zeros(len(ids), np.float32), 0, time.perf_counter()
for bt in dl:
    xb = bt[0].as_subclass(torch.Tensor).to(device)
    for u in range(len(xb)):
        i = int(ids[j])
        rng = np.random.default_rng([s, i] if conj == 'train' else [s, 1_000_000 + i])
        sel_, al = rng.integers(0, P, K), torch.as_tensor(rng.random(K), dtype=xb.dtype, device=device)
        c = int(D['label'][i])
        Ck, _ = conductancia(model, xb[u:u + 1], c, pool[sel_], al, con_fref=False)
        C[j] = Ck.cpu().numpy()
        with torch.no_grad():
            FC[j] = float(model(xb[u:u + 1])[0, c])
        j += 1
    if medir or j % 400 < 16:
        print(conj, j, '/', len(ids), round(time.perf_counter() - t1), 's', flush=True)
assert j == len(ids)
if medir:
    v = (time.perf_counter() - t1) / len(ids)
    n = len(tr['label']) + len(te['label'])
    print(f'MEDIDA: {v:.2f} s/imagen; {n} imágenes = {v * n / 3600:.1f} h de GPU; con 4 GPU, {v * n / 3600 / 4:.1f} h de reloj (+ carga de la reserva {round(time.perf_counter() - t0 - (time.perf_counter() - t1))} s por trabajo)', flush=True)
    sys.exit(0)
np.savez(f, C=C, f_c=FC, ids=ids, conjunto=conj, semilla=s)
print('BLOQUE_TERMINADO', f, round(time.perf_counter() - t0), 's', flush=True)
sys.stdout.flush()
os._exit(0)
