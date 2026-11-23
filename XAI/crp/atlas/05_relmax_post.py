# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4 (CPU): une los bloques de C, comprueba la matriz, elige 4 conceptos por clase (protocolo de la fase 4), las 8 referencias
RelMax por canal y las 50 candidatas de la estabilidad, y compara con A y G×E de la fase 4.
Uso: 05_relmax_post.py [parcial]   (parcial: usa solo los bloques terminados; para probar)"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

R = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
F10, F4 = os.path.join(R, 'fase10_eg'), os.path.join(R, 'fase4')
S = os.path.join(F10, 'paso4')
PARCIAL = len(sys.argv) > 1 and sys.argv[1] == 'parcial'
CLASES = ['Adenocarcinoma_ductal', 'Carcinoma_intraductal', 'Estroma', 'Glandulas_no_neoplasicas', 'Gleason_3', 'Gleason_4_cribiformes', 'Gleason_4_fusionadas', 'Gleason_4_glomeruloide',
          'Gleason_4_pobremente_formadas', 'Gleason_5_comedonecrosis', 'Gleason_5_no_glandulas', 'Inflamatorio', 'No_Cancer', 'PIN_de_alto_grado']
tr, te = np.load(os.path.join(F4, 'features', 'convnext_train.npz')), np.load(os.path.join(F4, 'features', 'convnext_test.npz'))
ntr, nte = len(tr['label']), len(te['label'])


def unir(conj, n, s=0):
    C, FC, visto = np.full((n, 768), np.nan, np.float32), np.full(n, np.nan, np.float32), np.zeros(n, int)
    fs = sorted(glob.glob(os.path.join(S, 'C', f'{conj}_*_s{s}.npz')))
    for f in fs:
        z = np.load(f)
        C[z['ids']], FC[z['ids']] = z['C'], z['f_c']
        visto[z['ids']] += 1
    return C, FC, visto, len(fs)


Ctr, FCtr, vtr, bt = unir('train', ntr)
Cte, FCte, vte, be = unir('test', nte)
comp = dict(bloques_train=bt, bloques_test=be, filas_train=int((vtr > 0).sum()), filas_test=int((vte > 0).sum()), duplicadas=int((vtr > 1).sum() + (vte > 1).sum()),
            no_finitas=int((~np.isfinite(Ctr[vtr > 0])).any(1).sum() + (~np.isfinite(Cte[vte > 0])).any(1).sum()),
            filas_cero=int((np.abs(Ctr[vtr > 0]).sum(1) == 0).sum() + (np.abs(Cte[vte > 0]).sum(1) == 0).sum()),
            completo=bool((vtr == 1).all() and (vte == 1).all()))
print(json.dumps(comp), flush=True)
if not comp['completo'] and not PARCIAL:
    sys.exit('Matriz incompleta: usa "parcial" para probar')
O = os.path.join(S, 'prueba_parcial' if PARCIAL else 'resultados')
os.makedirs(O, exist_ok=True)
json.dump(comp, open(os.path.join(O, 'comprobacion_matriz.json'), 'w'), indent=1)
mtr, mte = vtr > 0, vte > 0
# Normalización por imagen (abs_norm de CRP): C_k / Σ_k |C_k|
Ntr = np.where(mtr[:, None], Ctr / np.abs(Ctr).sum(1, keepdims=True), np.nan)
Nte = Cte[mte] / np.abs(Cte[mte]).sum(1, keepdims=True)
lte, ltr = te['label'][mte], tr['label']
if PARCIAL and len(Nte) == 0:          # solo para probar el código: sin bloques de test, los conceptos se eligen con las filas de train disponibles
    print('AVISO: prueba parcial sin test; conceptos elegidos con train', flush=True)
    Nte, lte = Ntr[mtr], ltr[mtr]
# 4 conceptos por clase: mayor media de C normalizada en el test de la clase real
media = np.full((14, 768), np.nan)
for c in range(14):
    if (lte == c).any():
        media[c] = Nte[lte == c].mean(0)
top4 = {c: np.argsort(-media[c])[:4].tolist() for c in range(14) if np.isfinite(media[c]).all()}
# Referencias RelMax: las 8 (y las 50 candidatas) de mayor C normalizada en train, por canal
Nfill = np.where(np.isfinite(Ntr), Ntr, -np.inf)
top50 = np.argsort(-Nfill, axis=0, kind='stable')[:50].T             # (768, 50) índices de train
filas_c, filas_r = [], []
c4 = pd.read_csv(os.path.join(F4, 'relmax', 'conceptos.csv'))
RM = {n: np.load(os.path.join(F4, 'relmax', n, 'RelMax_sum_normed', '0.model.stages.3.blocks.2_data.npy'))[:8] for n in ('A', 'GxE')}
for c, ks in top4.items():
    for p, k in enumerate(ks):
        refs = top50[k, :8]
        cls = ltr[refs]
        filas_c.append(dict(clase=CLASES[c], n_test=int((lte == c).sum()), puesto=p + 1, canal=int(k), media_norm=float(media[c, k]),
                            pureza_clase_concepto=float((cls == c).mean()), pureza_clase_mayoritaria=float(np.bincount(cls, minlength=14).max() / 8),
                            solape_refs_A=len(set(refs) & set(RM['A'][:, k].astype(int))), solape_refs_GxE=len(set(refs) & set(RM['GxE'][:, k].astype(int)))))
        for q, i in enumerate(refs):
            filas_r.append(dict(clase_figura=CLASES[c], puesto_concepto=p + 1, canal=int(k), puesto_ref=q + 1, idx_train=int(i), ruta=str(tr['paths'][i]), clase_ref=CLASES[ltr[i]], rel_norm=float(Ntr[i, k]), C=float(Ctr[i, k])))
dc, dr = pd.DataFrame(filas_c), pd.DataFrame(filas_r)
dc.to_csv(os.path.join(O, 'conceptos.csv'), index=False)
dr.to_csv(os.path.join(O, 'referencias.csv'), index=False)
cand = {int(k): top50[k].tolist() for k in sorted({k for ks in top4.values() for k in ks})}
json.dump(cand, open(os.path.join(O, 'candidatas50.json'), 'w'))
# Solape de conceptos por clase con la fase 4
sol = []
for c, ks in top4.items():
    a = set(c4[(c4.metodo == 'A') & (c4.clase == CLASES[c])].canal)
    g = set(c4[(c4.metodo == 'GxE') & (c4.clase == CLASES[c])].canal)
    sol.append(dict(clase=CLASES[c], canales_EG=sorted(ks), canales_A=sorted(a), canales_GxE=sorted(g), EG_y_A=len(set(ks) & a), EG_y_GxE=len(set(ks) & g), A_y_GxE=len(a & g)))
sol = pd.DataFrame(sol)
sol.to_csv(os.path.join(O, 'solape_conceptos.csv'), index=False)
# Solape de referencias en los 768 canales y repetición de conceptos entre clases
ref768 = pd.DataFrame([dict(canal=k, EG_y_A=len(set(top50[k, :8]) & set(RM['A'][:, k].astype(int))), EG_y_GxE=len(set(top50[k, :8]) & set(RM['GxE'][:, k].astype(int))),
                            A_y_GxE=len(set(RM['A'][:, k].astype(int)) & set(RM['GxE'][:, k].astype(int)))) for k in range(768)])
ref768.to_csv(os.path.join(O, 'solape_referencias_768.csv'), index=False)
todos = np.array([k for ks in top4.values() for k in ks])
u, n = np.unique(todos, return_counts=True)
rep = dict(canales_distintos=int(len(u)), puestos=int(len(todos)), max_clases_por_canal=int(n.max()), canal_mas_repetido=int(u[n.argmax()]))
pur = {}
for nom, M in (('A', RM['A']), ('GxE', RM['GxE'])):
    p4 = c4[c4.metodo == nom]
    pc, pm = [], []
    for _, r in p4.iterrows():
        cls = ltr[M[:, r.canal].astype(int)]
        pc.append((cls == CLASES.index(r.clase)).mean())
        pm.append(np.bincount(cls, minlength=14).max() / 8)
    pur[nom] = dict(pureza_clase_concepto_mediana=float(np.median(pc)), pureza_mayoritaria_mediana=float(np.median(pm)))
pur['EG'] = dict(pureza_clase_concepto_mediana=float(dc.pureza_clase_concepto.median()), pureza_mayoritaria_mediana=float(dc.pureza_clase_mayoritaria.median()))
res = dict(comprobacion=comp, repeticion_EG=rep, solape_conceptos=dict(EG_y_A_total=int(sol.EG_y_A.sum()), EG_y_GxE_total=int(sol.EG_y_GxE.sum()), A_y_GxE_total=int(sol.A_y_GxE.sum()), clases=len(sol)),
           solape_refs_768_mediana=ref768.median().drop('canal').to_dict(), pureza=pur)
json.dump(res, open(os.path.join(O, 'resumen_atlas.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
print(sol.to_string())
print('POST_TERMINADO', flush=True)
