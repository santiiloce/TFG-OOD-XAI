# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Mapas de las explicaciones en M (reglas en ../decisiones.md).
A: los 16 casos en float32. EG: se reutilizan los mapas del paso 3 de la fase 10 (192769) para los 11 casos de la fase 4
y se calculan los 5 nuevos, el suelo de ruido que falta y la aleatorización. Importa sin modificarlos comun4 y comun10.
Uso: python 01_mapas_M.py [--K 2 --cpu]   (la prueba en CPU con K=2 omite la comprobación de EG)
"""
import argparse
import ast
import json
import os
import sys
import time

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
S0 = os.path.join(RESULTS, 'explicaciones_f9', 'M')
P10 = os.path.join(RESULTS, 'fase10_eg', 'paso3')
sys.path.insert(0, os.path.join(RESULTS, 'fase10_eg', 'scripts'))
sys.path.insert(0, os.path.join(RESULTS, 'fase4', 'scripts'))
from comun10 import (CAPA, aleatorizado, cargar, cargar_refs, conductancia, mapas_canal, preprocesar, rho,  # noqa: E402
                     rutas_train, sorteo_eg)
from comun4 import ChannelConcept, composite_final, mapa, preparar_modelo_final  # noqa: E402
from crp.attribution import CondAttribution  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--K', type=int, default=128)
ap.add_argument('--cpu', action='store_true')
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
S = os.path.join(S0, 'mapas' if device.type == 'cuda' else 'prueba_cpu')
os.makedirs(S, exist_ok=True)
T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)    # noqa: E731
t_ini = time.perf_counter()
tiempos = {}

# ---------------- Casos y correspondencia con la fase 4 ----------------
casos = json.load(open(os.path.join(RESULTS, 'coherencia_f9', 'M', 'casos', 'casos_explicacion.json')))['casos']
sel4 = json.load(open(os.path.join(RESULTS, 'fase4', 'explicaciones', 'seleccion.json')))['casos']
z6 = pd.read_csv(os.path.join(RESULTS, 'fase6', 'canal102', 'presencia_normalizada.csv'))
k4 = {(s['conjunto'], s['indice']): j for j, s in enumerate(sel4)}
REUSO = {j: k4[(s['conjunto'], s['indice'])] for j, s in enumerate(casos) if s['en_fase4']}
for j, j4 in REUSO.items():
    s = casos[j]
    assert s['vecinos'] == sel4[j4]['vecinos'] and s['canales'] == sel4[j4]['canales'] and s['canales_z'] == ast.literal_eval(z6.canales_z[j4])
NUEVOS = [j for j in range(16) if j not in REUSO]
assert len(REUSO) == 11 and NUEVOS == [10, 11, 13, 14, 15]
CINCO = [0, 4, 8, 12, 14]
assert [casos[j]['grupo'] for j in CINCO] == ['bien_clasificada', 'mal_clasificada', 'id_rechazada', 'ood_aceptada', 'ood_rechazada']
q_de = lambda j, f: 4 * REUSO[j] + f if j in REUSO else 56 + 4 * NUEVOS.index(j) + f    # noqa: E731
idx0, alf0 = sorteo_eg(76, k=args.K, semilla=0)
idx1, alf1 = sorteo_eg(76, k=args.K, semilla=1)
m10 = np.load(os.path.join(P10, 'mapas.npz'))
cob10 = pd.read_csv(os.path.join(P10, 'cobertura_explicaciones.csv'))
sue10 = pd.read_csv(os.path.join(P10, 'suelo_ruido_semillas.csv'))
can_de = lambda s: list(dict.fromkeys(s['canales'] + s['canales_z']))    # noqa: E731

learn, model = cargar('convnext', device)
model = model.float().eval()
mA, compA = preparar_modelo_final(model, device), composite_final()
atA = CondAttribution(mA, no_param_grad=True)
trR = [str(p) for p in np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))['paths']]
assert trR == rutas_train(np.arange(len(trR)))


def fila_A(at, xi, c, canales, con_R=True):
    R = None
    if con_R:
        a, _ = mapa(at, compA, 'A', xi, c, capas=[CAPA])
        R = ChannelConcept().attribute(a.relevances[CAPA].detach(), abs_norm=False)[0].cpu().numpy()
    return R, {k: mapa(at, compA, 'A', xi, c, canales=[k])[1].astype(np.float32) for k in canales}


def fila_EG(m, xi, c, q, canales, semilla=0, con_C=True):
    idx, alf = (idx0, alf0) if semilla == 0 else (idx1, alf1)
    refs = cargar_refs(learn, rutas_train(idx[q]), device)
    C = conductancia(m, xi, c, refs, T(alf[q]), con_fref=False)[0].cpu().numpy() if con_C else None
    return C, dict(zip(canales, mapas_canal(m, xi, c, refs, T(alf[q]), canales).cpu().numpy().astype(np.float32)))


def comparar(h, ref):
    ref = np.asarray(ref, np.float32)
    return dict(rho=rho(h, ref), err_rel_max=float(np.abs(h - ref).max() / max(np.abs(ref).max(), 1e-30)))


def eg10(j, f, k):
    """Mapa EG reutilizado de la fase 10 (los canales están en 'original' o en 'z'; si están en los dos son el mismo cálculo)."""
    j4 = REUSO[j]
    for cr in ('original', 'z'):
        if f'{cr}|{j4}|{f}|{k}' in m10.files:
            return m10[f'{cr}|{j4}|{f}|{k}']
    raise KeyError((j, f, k))


xs = {j: preprocesar(learn, [s['ruta']] + [trR[v] for v in s['vecinos']], device) for j, s in enumerate(casos)}

# ---------------- Comprobación de EG: caso 0 recalculado frente a la fase 10 ----------------
comp = []
if args.K == 128:
    t = time.perf_counter()
    s = casos[0]
    for f in range(4):
        _, h = fila_EG(model, xs[0][f:f + 1], s['c'], q_de(0, f), can_de(s), con_C=False)
        for k in can_de(s):
            comp.append(dict(metodo='EG', ref='fase10 float32 (reutilizado)', fila=f, canal=k, **comparar(h[k], eg10(0, f, k))))
    tiempos['comprobacion_EG_s'] = time.perf_counter() - t
    ce = pd.DataFrame(comp)
    ok = bool((ce.rho >= 0.999).all() and (ce.err_rel_max <= 1e-3).all())
    print(f'COMPROBACION EG: rho min {ce.rho.min():.9f}, err rel max {ce.err_rel_max.max():.2e} -> {"pasa" if ok else "FALLA"}', flush=True)
    if not ok:
        ce.to_csv(os.path.join(S, 'comprobacion.csv'), index=False)
        print('COMPROBACION_FALLA: se para', flush=True)
        sys.exit(2)

# ---------------- Mapas ----------------
mapas, origen = {}, {}
R_A = np.full((16, 4, 768), np.nan, np.float32)
C_EG = np.full((16, 4, 768), np.nan, np.float32)        # 768 en los nuevos; solo los 8 canales mostrados en los reutilizados
cob_EG = np.full((16, 4, 2), np.nan)                    # cobertura de la fase 10 para las filas reutilizadas (orig, z)
tA = tE = 0.0
for j, s in enumerate(casos):
    can = can_de(s)
    for f in range(4):
        t = time.perf_counter()
        R, hA = fila_A(atA, xs[j][f:f + 1], s['c'], can)
        tA += time.perf_counter() - t
        R_A[j, f] = R
        for k in can:
            mapas[f'A|{j}|{f}|{k}'] = hA[k]
        if j in REUSO:
            for ci, (cr, ks) in enumerate((('original', s['canales']), ('z', s['canales_z']))):
                fila = cob10[(cob10.caso == REUSO[j]) & (cob10.fila == f) & (cob10.criterio == cr)].iloc[0]
                C_EG[j, f, ks] = ast.literal_eval(fila.C)
                cob_EG[j, f, ci] = fila.cobertura_4
            for k in can:
                mapas[f'EG|{j}|{f}|{k}'] = eg10(j, f, k)
            origen[f'{j}|{f}'] = f'fase10 paso3 (192769), caso {REUSO[j]}, q = {q_de(j, f)}'
        else:
            t = time.perf_counter()
            C, hE = fila_EG(model, xs[j][f:f + 1], s['c'], q_de(j, f), can)
            tE += time.perf_counter() - t
            C_EG[j, f] = C
            for k in can:
                mapas[f'EG|{j}|{f}|{k}'] = hE[k]
            origen[f'{j}|{f}'] = f'calculado aquí, q = {q_de(j, f)}'
        if j == 0:   # A frente a la fase 10 (float32) y la fase 4 (float16)
            for k in s['canales']:
                comp.append(dict(metodo='A', ref='fase10 float32', fila=f, canal=k, **comparar(hA[k], m10[f'A|0|{f}|{k}'])))
                comp.append(dict(metodo='A', ref='fase4 float16 (informativo)', fila=f, canal=k, **comparar(hA[k], np.load(os.path.join(RESULTS, 'fase4', 'explicaciones', 'mapas.npz'))[f'A|0|{f}|{k}'])))
        if j == 12:
            for k in s['canales']:
                comp.append(dict(metodo='A', ref='fase10 float32', fila=f, canal=k, **comparar(hA[k], m10[f'A|12|{f}|{k}'])))
    print(f'caso {j}: {time.perf_counter() - t_ini:.0f} s', flush=True)

# ---------------- Suelo de ruido (EG) y aleatorización ----------------
suelo, ale = [], []
tS = tR = 0.0
for j in CINCO:
    s, can = casos[j], can_de(casos[j])
    t = time.perf_counter()
    if j in REUSO:
        for k in can:
            fila = sue10[(sue10.caso == REUSO[j]) & (sue10.canal == k)]
            suelo.append(dict(caso=j, canal=k, rho=float(fila.rho_semillas.iloc[0]), origen='fase10'))
    else:
        _, h1 = fila_EG(model, xs[j][:1], s['c'], q_de(j, 0), can, semilla=1, con_C=False)
        for k in can:
            mapas[f'EG_semilla1|{j}|0|{k}'] = h1[k]
            suelo.append(dict(caso=j, canal=k, rho=rho(mapas[f'EG|{j}|0|{k}'], h1[k]), origen='calculado'))
    tS += time.perf_counter() - t
    t = time.perf_counter()
    for sorteo in range(5):
        torch.manual_seed(sorteo)
        ma = aleatorizado(mA, ['1', '0.model.stages.3'])
        _, hA2 = fila_A(CondAttribution(ma, no_param_grad=True), xs[j][:1], s['c'], can, con_R=False)
        del ma
        torch.manual_seed(sorteo)
        me = aleatorizado(model, ['1', '0.model.stages.3'])
        _, hE2 = fila_EG(me, xs[j][:1], s['c'], q_de(j, 0), can, con_C=False)
        del me
        for nom, h2 in (('A', hA2), ('EG', hE2)):
            for k in can:
                v = h2[k]
                ale.append(dict(metodo=nom, caso=j, canal=k, sorteo=sorteo,
                                rho=rho(mapas[f'{nom}|{j}|0|{k}'], v) if np.isfinite(v).all() and v.std() > 0 else np.nan))
    tR += time.perf_counter() - t
    print(f'controles caso {j}: {time.perf_counter() - t_ini:.0f} s', flush=True)

cp = pd.DataFrame(comp)
cp.to_csv(os.path.join(S, 'comprobacion.csv'), index=False)
resA = cp[cp.ref == 'fase10 float32']
compr = {f'{m}|{r}': dict(rho_min=float(g.rho.min()), err_rel_max=float(g.err_rel_max.max()), n=len(g)) for (m, r), g in cp.groupby(['metodo', 'ref'])}
compr['A_pasa (rho>=0.9999, err<=1e-2, frente a fase 10 float32)'] = bool((resA.rho >= 0.9999).all() and (resA.err_rel_max <= 1e-2).all())
json.dump(compr, open(os.path.join(S, 'comprobacion.json'), 'w'), indent=1)
tiempos.update(A_s=tA, EG_nuevos_s=tE, suelo_s=tS, aleatorizacion_s=tR, total_s=time.perf_counter() - t_ini)
np.savez_compressed(os.path.join(S, 'mapas.npz'), **mapas)
np.savez_compressed(os.path.join(S, 'relevancias.npz'), R_A=R_A, C_EG=C_EG, cob_EG_fase10=cob_EG)
json.dump(origen, open(os.path.join(S, 'origen_EG.json'), 'w'), indent=1, ensure_ascii=False)
pd.DataFrame(ale).to_csv(os.path.join(S, 'aleatorizacion_por_mapa.csv'), index=False)
pd.DataFrame(suelo).to_csv(os.path.join(S, 'suelo_ruido_semillas.csv'), index=False)
json.dump(dict(tiempos=tiempos, K=args.K, dispositivo=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu',
               slurm_job=os.environ.get('SLURM_JOB_ID'), nodo=os.uname().nodename), open(os.path.join(S, 'ejecucion.json'), 'w'), indent=1)
print(json.dumps(compr, indent=1), flush=True)
print(json.dumps(tiempos, indent=1), flush=True)
print('MAPAS_TERMINADOS', flush=True)
sys.stdout.flush()
os._exit(0)
