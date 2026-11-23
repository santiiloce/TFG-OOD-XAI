# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 3: explicaciones al estilo de STOOD-X con EG condicionado (φ^k) y controles de la fase 4 con EG.
Mismos 14 casos, vecinos y clase predicha que la fase 4. Características: originales (fase 4) y estandarizadas (fase 6)."""
import ast
import json
import os
import sys
import time

from comun10 import (CAPA, F10, RESULTS, aleatorizado, boot_ic, cargar, cargar_refs, conductancia, mapas_canal, preprocesar, rho,
                     rutas_train, seleccion, semilla_clase, sorteo_eg)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(RESULTS, 'fase5', 'scripts'))
sys.path.insert(0, os.path.join(RESULTS, 'fase4', 'scripts'))
from comun4 import ABREV, CLASES, DatasetRutas, cmap_rel, estilo, pinta_img, pinta_mapa   # noqa: E402
from composite_final import composite_final, preparar_modelo_final                       # noqa: E402
from crp.attribution import CondAttribution                                              # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cuda')
PRUEBA = len(sys.argv) > 1 and sys.argv[1] == 'prueba'     # 1 caso, 2 imágenes de diversidad, 1 sorteo; escribe en paso3_prueba
S = os.path.join(F10, 'paso3_prueba' if PRUEBA else 'paso3')
for d in ('figuras/original', 'figuras/z', 'figuras/A_vs_EG'):
    os.makedirs(os.path.join(S, d), exist_ok=True)
learn, model = cargar('convnext', device)
sel = json.load(open(os.path.join(RESULTS, 'fase4', 'explicaciones', 'seleccion.json')))['casos']
pz = pd.read_csv(os.path.join(RESULTS, 'fase6', 'canal102', 'presencia_normalizada.csv'))
for j, s in enumerate(sel):
    s['canales_z'] = ast.literal_eval(pz.canales_z[j])
    assert ast.literal_eval(pz.canales_fase4[j]) == s['canales']
trd = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))
trR, trL = [str(p) for p in trd['paths']], trd['label']
idx0, alf0 = sorteo_eg(56, k=128, semilla=0)
idx1, alf1 = sorteo_eg(56, k=128, semilla=1)
plt, cm = estilo(), cmap_rel()
mA, compA = preparar_modelo_final(model, device), composite_final()
atA = CondAttribution(mA, no_param_grad=True)


def mapa_A(xi, c, k):
    a = atA(xi.clone().requires_grad_(True), [{'y': [c], CAPA: [k]}], compA, init_rel=semilla_clase(c))
    return a.heatmap[0].detach().cpu().numpy()


T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)    # noqa: E731
t0 = time.perf_counter()
filas, mapas, cob = [], {}, []
for j, s in enumerate(sel[:1] if PRUEBA else sel):
    rutas = [s['ruta']] + [trR[v] for v in s['vecinos']]
    x = preprocesar(learn, rutas, device)
    dsx = DatasetRutas(learn, rutas, [0] * 4)
    can = list(dict.fromkeys(s['canales'] + s['canales_z']))
    filas_caso = []
    for f in range(4):
        q = 4 * j + f
        refs = cargar_refs(learn, rutas_train(idx0[q]), device)
        C, _ = conductancia(model, x[f:f + 1], s['c'], refs, T(alf0[q]))
        P = mapas_canal(model, x[f:f + 1], s['c'], refs, T(alf0[q]), can).cpu().numpy()
        Cn = C.cpu().numpy()
        for crit, ks in (('original', s['canales']), ('z', s['canales_z'])):
            hs = [P[can.index(k)] for k in ks]
            for k, h in zip(ks, hs):
                mapas[f'{crit}|{j}|{f}|{k}'] = h.astype(np.float32)
            cob.append(dict(caso=j, grupo=s['grupo'], fila=f, criterio=crit, cobertura_4=float(np.abs(Cn[ks]).sum() / np.abs(Cn).sum()), suma_C=float(Cn.sum()),
                            abs_rho_entre_4=float(np.mean([abs(rho(hs[p], hs[u])) for p in range(4) for u in range(p + 1, 4)])), C=[float(Cn[k]) for k in ks]))
        if f == 0:                                    # suelo de ruido: semilla 1 en la fila de la imagen
            refs1 = cargar_refs(learn, rutas_train(idx1[q]), device)
            P1 = mapas_canal(model, x[:1], s['c'], refs1, T(alf1[q]), can).cpu().numpy()
            for crit, ks in (('original', s['canales']), ('z', s['canales_z'])):
                for k in ks:
                    filas.append(dict(caso=j, criterio=crit, canal=k, rho_semillas=rho(P[can.index(k)], P1[can.index(k)])))
        filas_caso.append((Cn, P))
    # Figuras con el formato de la fase 4
    for crit in ('original', 'z'):
        ks = s['canales'] if crit == 'original' else s['canales_z']
        fig, ax = plt.subplots(4, 5, figsize=(11, 9.4))
        for f in range(4):
            Cn, P = filas_caso[f]
            pinta_img(ax[f, 0], dsx[f][0])
            cr = s['clase_real'] if f == 0 else CLASES[trL[s['vecinos'][f - 1]]]
            dd = f"d50 = {s['d50']:.3f}" if f == 0 else f"d = {s['dist_vecinos'][f - 1]:.3f}"
            ax[f, 0].set_title(f"{'imagen' if f == 0 else f'vecino {f}'}: {ABREV.get(cr, cr)}\n{dd}", fontsize=8, loc='left')
            for u, k in enumerate(ks):
                pinta_mapa(ax[f, u + 1], P[can.index(k)], cm)
                ax[f, u + 1].set_title(f'canal {k}, C = {Cn[k]:.3g}'.replace('.', ','), fontsize=8, loc='left')
        fig.suptitle(f"{s['grupo'].replace('_', ' ')} ({s['conjunto']}): clase predicha {ABREV[s['clase_pred']]}, softmax {s['softmax']:.3f}, EG condicionado (conductancia), presencia {'original' if crit == 'original' else 'estandarizada'}",
                     x=.02, ha='left', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(os.path.join(S, 'figuras', crit, f"{j:02d}_{s['grupo']}.png"), dpi=120)
        plt.close(fig)
    # Comparativa A frente a EG (casos 0 y 12; presencia original)
    if j in (0, 12):
        ks = s['canales']
        fig, ax = plt.subplots(4, 9, figsize=(6.3, 3.3))
        for f in range(4):
            pinta_img(ax[f, 0], dsx[f][0])
            for u, k in enumerate(ks):
                hA = mapa_A(x[f:f + 1], s['c'], k)
                mapas[f'A|{j}|{f}|{k}'] = hA.astype(np.float32)
                pinta_mapa(ax[f, 1 + u], hA, cm)
                pinta_mapa(ax[f, 5 + u], filas_caso[f][1][can.index(k)], cm)
                if f == 0:
                    ax[f, 1 + u].set_title(f'A, c. {k}', fontsize=7)
                    ax[f, 5 + u].set_title(f'EG, c. {k}', fontsize=7)
            ax[f, 0].set_ylabel('imagen' if f == 0 else f'vecino {f}', fontsize=7)
        fig.tight_layout(pad=0.2)
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(S, 'figuras', 'A_vs_EG', f"caso{j:02d}_{s['grupo']}.{ext}"), dpi=300)
        plt.close(fig)
    print('caso', j, round(time.perf_counter() - t0), 's', flush=True)
pd.DataFrame(cob).to_csv(os.path.join(S, 'cobertura_explicaciones.csv'), index=False)
pd.DataFrame(filas).to_csv(os.path.join(S, 'suelo_ruido_semillas.csv'), index=False)
np.savez_compressed(os.path.join(S, 'mapas.npz'), **mapas)

# ---- Controles con EG
# Diversidad y cobertura: 28 imágenes, C del paso 1 (mismas referencias), 4 canales de mayor |C|
_, rutas28, _ = seleccion(2)
x28 = preprocesar(learn, rutas28, device)
with torch.no_grad():
    pred28 = model(x28).argmax(1).cpu().numpy()
i28, a28 = sorteo_eg(28, k=128, semilla=0)
div = []
for i in range(2 if PRUEBA else 28):
    C = np.load(os.path.join(F10, 'paso1', f'C_img{i:02d}.npy'))
    top = np.argsort(-np.abs(C))[:4].tolist()
    refs = cargar_refs(learn, rutas_train(i28[i]), device)
    P = mapas_canal(model, x28[i:i + 1], int(pred28[i]), refs, T(a28[i]), top).cpu().numpy()
    rr = [rho(P[p], P[q]) for p in range(4) for q in range(p + 1, 4)]
    div.append(dict(imagen=i, abs_rho=float(np.mean(np.abs(rr))), rho=float(np.mean(rr)), frac_neg=float(np.mean(np.array(rr) < 0)), cobertura_4=float(np.abs(C[top]).sum() / np.abs(C).sum()), canales=top))
print('diversidad', round(time.perf_counter() - t0), 's', flush=True)
# Aleatorización: 5 imágenes (primera de cada grupo y las dos OOD), sus 4 características originales, 5 sorteos
vistos, cinco = set(), []
for j, s in enumerate(sel):
    if s['grupo'] not in vistos or s['grupo'] == 'ood_aceptada':
        cinco.append(j)
        vistos.add(s['grupo'])
assert len(cinco) == 5
if PRUEBA:
    cinco = cinco[:1]
ale = []
for sorteo in range(1 if PRUEBA else 5):
    torch.manual_seed(sorteo)
    ma = aleatorizado(model, ['1', '0.model.stages.3'])
    for j in cinco:
        s = sel[j]
        xi = preprocesar(learn, [s['ruta']], device)
        refs = cargar_refs(learn, rutas_train(idx0[4 * j]), device)
        P2 = mapas_canal(ma, xi, s['c'], refs, T(alf0[4 * j]), s['canales']).cpu().numpy()
        for u, k in enumerate(s['canales']):
            h = mapas[f'original|{j}|0|{k}']
            ale.append(dict(caso=j, canal=k, sorteo=sorteo, rho=rho(h, P2[u]) if np.isfinite(P2[u]).all() and P2[u].std() > 0 else np.nan))
    del ma
print('aleatorización', round(time.perf_counter() - t0), 's', flush=True)
d, e, fs, cb = pd.DataFrame(div), pd.DataFrame(ale), pd.DataFrame(filas), pd.DataFrame(cob)
d.to_csv(os.path.join(S, 'diversidad_cobertura_por_imagen.csv'), index=False)
e.to_csv(os.path.join(S, 'aleatorizacion_por_mapa.csv'), index=False)
por = e.assign(abs=e.rho.abs()).groupby('sorteo')['abs'].median()
f4 = pd.read_csv(os.path.join(RESULTS, 'fase4', 'controles', 'controles_resumen.csv'))
cob4 = pd.read_csv(os.path.join(RESULTS, 'fase4', 'explicaciones', 'cobertura.csv'))
res = [dict(metodo='EG (conductancia)', abs_rho_entre_4=d.abs_rho.median(), IC=boot_ic(d.abs_rho.values, np.median), cobertura_4=d.cobertura_4.median(), IC_cob=boot_ic(d.cobertura_4.values, np.median),
            ale_abs_rho=por.median(), ale_min=por.min(), ale_max=por.max(), ale_nan=int(e.rho.isna().sum()),
            cobertura_4_explicaciones_original=cb[cb.criterio == 'original'].cobertura_4.median(), cobertura_4_explicaciones_z=cb[cb.criterio == 'z'].cobertura_4.median(),
            abs_rho_entre_4_explicaciones_original=cb[cb.criterio == 'original'].abs_rho_entre_4.median(), abs_rho_entre_4_explicaciones_z=cb[cb.criterio == 'z'].abs_rho_entre_4.median(),
            suelo_ruido_original=fs[fs.criterio == 'original'].rho_semillas.abs().median(), suelo_ruido_z=fs[fs.criterio == 'z'].rho_semillas.abs().median(),
            IC_suelo_original=boot_ic(fs[fs.criterio == 'original'].rho_semillas.abs().values, np.median), IC_suelo_z=boot_ic(fs[fs.criterio == 'z'].rho_semillas.abs().values, np.median))]
for _, r in f4.iterrows():
    nm = 'A' if r.metodo == 'A' else 'G×E'
    c4 = cob4[cob4.metodo == r.metodo]
    res.append(dict(metodo=f'{nm} (fase 4)', abs_rho_entre_4=r.abs_rho_entre_4, IC=r.IC, cobertura_4=r.cobertura_4, IC_cob=r.IC_cob, ale_abs_rho=r.ale_abs_rho, ale_min=r.ale_min, ale_max=r.ale_max,
                    cobertura_4_explicaciones_original=c4.cobertura_4.median(), abs_rho_entre_4_explicaciones_original=c4.abs_rho_entre_4.median()))
res = pd.DataFrame(res)
res.to_csv(os.path.join(S, 'controles_resumen.csv'), index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print(res.T.to_string())
json.dump(dict(minutos=(time.perf_counter() - t0) / 60, gpu=torch.cuda.get_device_name(0)), open(os.path.join(S, 'ejecucion.json'), 'w'))
print('PASO3_TERMINADO', flush=True)
sys.stdout.flush()
os._exit(0)
