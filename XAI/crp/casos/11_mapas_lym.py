# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Mapas A y EG del caso LYM (../nuevos/caso_lym.json), con las mismas funciones que 01_mapas_M.py.
EG: referencias de sorteo_eg(80, K=128, semilla=0), fila f -> q = 76 + f (01_mapas_M.py usa q = 0-75: no se solapan).
Escribe ../nuevos/mapas.npz y relevancias.npz.  Uso: sbatch script_zd.sh 11_mapas_lym.py [--K 2 --cpu]"""
import argparse, json, os, sys, time

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
S = os.path.join(RESULTS, 'explicaciones_f9', 'M', 'nuevos')
sys.path.insert(0, os.path.join(RESULTS, 'fase10_eg', 'scripts'))
sys.path.insert(0, os.path.join(RESULTS, 'fase4', 'scripts'))
from comun10 import CAPA, cargar, cargar_refs, conductancia, mapas_canal, preprocesar, rutas_train, sorteo_eg  # noqa: E402
from comun4 import ChannelConcept, composite_final, mapa, preparar_modelo_final  # noqa: E402
from crp.attribution import CondAttribution  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=128)
ap.add_argument('--cpu', action='store_true')
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
out = S if device.type == 'cuda' else os.path.join(S, 'prueba_cpu')
os.makedirs(out, exist_ok=True)
T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)    # noqa: E731
t0 = time.perf_counter()

s = json.load(open(os.path.join(S, 'caso_lym.json')))
learn, model = cargar('convnext', device)
model = model.float().eval()
mA, compA = preparar_modelo_final(model, device), composite_final()
atA = CondAttribution(mA, no_param_grad=True)
trR = [str(p) for p in np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))['paths']]
idx0, alf0 = sorteo_eg(80, k=args.K, semilla=0)
x = preprocesar(learn, [s['ruta']] + [trR[v] for v in s['vecinos']], device)
can = list(dict.fromkeys(s['canales'] + s['canales_z']))
mapas, R_A, C_EG, fc = {}, np.zeros((4, 768), np.float32), np.zeros((4, 768), np.float32), []
for f in range(4):
    xi, c, q = x[f:f + 1], s['c'], 76 + f
    a, _ = mapa(atA, compA, 'A', xi, c, capas=[CAPA])
    R_A[f] = ChannelConcept().attribute(a.relevances[CAPA].detach(), abs_norm=False)[0].cpu().numpy()
    for k in can:
        mapas[f'A|{f}|{k}'] = mapa(atA, compA, 'A', xi, c, canales=[k])[1].astype(np.float32)
    refs = cargar_refs(learn, rutas_train(idx0[q]), device)
    C, fref = conductancia(model, xi, c, refs, T(alf0[q]))
    C_EG[f] = C.cpu().numpy()
    P = mapas_canal(model, xi, c, refs, T(alf0[q]), can).cpu().numpy().astype(np.float32)
    for k, p in zip(can, P):
        mapas[f'EG|{f}|{k}'] = p
    with torch.no_grad():
        fc.append((model(xi)[0, c].item(), fref.mean().item()))
    print(f'fila {f}: {time.perf_counter() - t0:.0f} s; ΣR/f_c {R_A[f].sum() / fc[-1][0]:.4f}, '
          f'ΣC/(f_c−E f_ref) {C_EG[f].sum() / (fc[-1][0] - fc[-1][1]):.4f}', flush=True)
np.savez_compressed(os.path.join(out, 'mapas.npz'), **mapas)
np.savez_compressed(os.path.join(out, 'relevancias.npz'), R_A=R_A, C_EG=C_EG, f_c=np.array(fc))
print(f'total {time.perf_counter() - t0:.0f} s', flush=True)
