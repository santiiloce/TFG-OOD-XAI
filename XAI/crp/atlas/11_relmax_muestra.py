# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 4 sobre la muestra estratificada (cambio del 17-11): RelMax de EG y de A sobre la misma muestra con el protocolo de la fase 4.
- Conceptos por clase: los 4 de mayor media de relevancia normalizada en el test de la clase real (EG: C de test de la fase 10;
  A: los de la fase 4, también elegidos en test).
- Referencias: las 8 imágenes de la muestra con mayor relevancia normalizada (abs_norm) del canal; candidatas de estabilidad: las 50 mayores.
- Comparación principal: A(muestra) frente a EG(muestra). Comprobación del submuestreo: A(muestra) frente a A(fase 4, train completo).
- Registra qué GPU calculó cada fila de la muestra (a partir de los logs)."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

R = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
F10, F4 = os.path.join(R, 'fase10_eg'), os.path.join(R, 'fase4')
S = os.path.join(F10, 'paso4', 'muestra')
O = os.path.join(S, 'resultados')
os.makedirs(O, exist_ok=True)
CLASES = ['Adenocarcinoma_ductal', 'Carcinoma_intraductal', 'Estroma', 'Glandulas_no_neoplasicas', 'Gleason_3', 'Gleason_4_cribiformes', 'Gleason_4_fusionadas', 'Gleason_4_glomeruloide',
          'Gleason_4_pobremente_formadas', 'Gleason_5_comedonecrosis', 'Gleason_5_no_glandulas', 'Inflamatorio', 'No_Cancer', 'PIN_de_alto_grado']
tr, te = np.load(os.path.join(F4, 'features', 'convnext_train.npz')), np.load(os.path.join(F4, 'features', 'convnext_test.npz'))
ltr, lte = tr['label'], te['label']
mu = np.load(os.path.join(S, 'muestra_idx.npy'))

# ---- Origen (GPU) de cada bloque, a partir de los logs
origen = {}
for log in glob.glob(os.path.join(F10, 'logs', 'slurm-*.out')):
    txt = open(log, errors='ignore').read()
    m = re.search(r'^(\w+)\n(.+?), [\d.]+, \d', txt, re.M)
    for f in re.findall(r'BLOQUE_TERMINADO (\S+\.npz)', txt):
        origen[os.path.basename(f)] = f'{m.group(1)} ({m.group(2)})' if m else 'desconocido'

# ---- C de EG para la muestra (bloques + faltantes) y para todo el test
C = np.full((len(ltr), 768), np.nan, np.float32)
gpu = np.array([''] * len(ltr), dtype=object)
for f in glob.glob(os.path.join(F10, 'paso4', 'C', 'train_*_s0.npz')):
    z = np.load(f)
    C[z['ids']] = z['C']
    gpu[z['ids']] = origen.get(os.path.basename(f), 'desconocido')
z = np.load(os.path.join(S, 'C_muestra_faltantes_s0.npz'))
C[z['ids']] = z['C']
log_mu = open(glob.glob(os.path.join(F10, 'logs', f"slurm-{open(os.path.join(S, 'trabajo_calculo.txt')).read().strip()}_*.out"))[0]).read().split('\n')
gpu[z['ids']] = f'{log_mu[0]} ({log_mu[1].split(",")[0]})'
Cm = C[mu]
assert np.isfinite(Cm).all() and (np.abs(Cm).sum(1) > 0).all(), 'muestra incompleta'
orig = pd.DataFrame(dict(clase=[CLASES[c] for c in ltr[mu]], gpu=gpu[mu])).value_counts().rename('filas').reset_index()
orig.to_csv(os.path.join(O, 'origen_gpu_filas_muestra.csv'), index=False)
Cte = np.full((len(lte), 768), np.nan, np.float32)
for f in glob.glob(os.path.join(F10, 'paso4', 'C', 'test_*_s0.npz')):
    z = np.load(f)
    Cte[z['ids']] = z['C']
assert np.isfinite(Cte).all()
# ---- A en la muestra
za = np.load(os.path.join(S, 'R_A_muestra.npz'))
assert (za['ids'] == mu).all()
RA = za['R'].astype(np.float64)


def norm(M):
    return M / np.abs(M).sum(1, keepdims=True)


N = {'EG': norm(Cm.astype(np.float64)), 'A': norm(RA)}
Nte = norm(Cte.astype(np.float64))
media = np.stack([Nte[lte == c].mean(0) for c in range(14)])
conc = {'EG': {c: np.argsort(-media[c])[:4].tolist() for c in range(14)}}
c4 = pd.read_csv(os.path.join(F4, 'relmax', 'conceptos.csv'))
conc['A'] = {c: c4[(c4.metodo == 'A') & (c4.clase == CLASES[c])].sort_values('puesto').canal.tolist() for c in range(14)}
conc['GxE'] = {c: c4[(c4.metodo == 'GxE') & (c4.clase == CLASES[c])].sort_values('puesto').canal.tolist() for c in range(14)}
top50 = {m: mu[np.argsort(-N[m], axis=0, kind='stable')[:50].T] for m in ('EG', 'A')}       # (768, 50) índices de train
RM4 = {n: np.load(os.path.join(F4, 'relmax', n, 'RelMax_sum_normed', '0.model.stages.3.blocks.2_data.npy'))[:8].astype(int) for n in ('A', 'GxE')}
en_muestra = np.zeros(len(ltr), bool)
en_muestra[mu] = True
filas_c, filas_r = [], []
for m in ('EG', 'A'):
    pos = {int(i): j for j, i in enumerate(mu)}
    for c in range(14):
        for p, k in enumerate(conc[m][c]):
            refs = top50[m][k, :8]
            cls = ltr[refs]
            filas_c.append(dict(metodo=f'{m} (muestra)', clase=CLASES[c], puesto=p + 1, canal=int(k), media_norm_test=float(media[c, k]) if m == 'EG' else np.nan,
                                pureza_clase_concepto=float((cls == c).mean()), pureza_clase_mayoritaria=float(np.bincount(cls, minlength=14).max() / 8),
                                clase_mayoritaria=CLASES[np.bincount(cls, minlength=14).argmax()]))
            for q, i in enumerate(refs):
                filas_r.append(dict(metodo=m, clase_figura=CLASES[c], puesto_concepto=p + 1, canal=int(k), puesto_ref=q + 1, idx_train=int(i), ruta=str(tr['paths'][i]),
                                    clase_ref=CLASES[ltr[i]], rel_norm=float(N[m][pos[int(i)], k])))
dc, dr = pd.DataFrame(filas_c), pd.DataFrame(filas_r)
dc.to_csv(os.path.join(O, 'conceptos.csv'), index=False)
dr.to_csv(os.path.join(O, 'referencias_todas.csv'), index=False)
dr[dr.metodo == 'EG'].to_csv(os.path.join(O, 'referencias.csv'), index=False)        # las que usa 06 para los mapas de EG
dr[dr.metodo == 'A'].to_csv(os.path.join(O, 'referencias_A.csv'), index=False)
cand = {int(k): top50['EG'][k].tolist() for k in sorted({k for ks in conc['EG'].values() for k in ks})}
json.dump(cand, open(os.path.join(O, 'candidatas50.json'), 'w'))

# ---- Comparaciones
sol = []
for c in range(14):
    e, a, g = set(conc['EG'][c]), set(conc['A'][c]), set(conc['GxE'][c])
    sol.append(dict(clase=CLASES[c], canales_EG=sorted(e), canales_A=sorted(a), canales_GxE=sorted(g), EG_y_A=len(e & a), EG_y_GxE=len(e & g), A_y_GxE=len(a & g)))
sol = pd.DataFrame(sol)
sol.to_csv(os.path.join(O, 'solape_conceptos.csv'), index=False)
r768 = []
for k in range(768):
    eg8, a8, a48 = set(top50['EG'][k, :8]), set(top50['A'][k, :8]), set(RM4['A'][:, k])
    r768.append(dict(canal=k, EGm_y_Am=len(eg8 & a8), Am_y_A4=len(a8 & a48), A4_en_muestra=int(en_muestra[list(a48)].sum()),
                     Am_y_A4_en_muestra=len(a8 & {i for i in a48 if en_muestra[i]}), EGm_y_GxE4=len(eg8 & set(RM4['GxE'][:, k]))))
r768 = pd.DataFrame(r768)
r768.to_csv(os.path.join(O, 'solape_referencias_768.csv'), index=False)
pur = dc.groupby('metodo')[['pureza_clase_concepto', 'pureza_clase_mayoritaria']].median().to_dict('index')
for n in ('A', 'GxE'):
    pc, pm = [], []
    for c in range(14):
        for k in conc[n][c]:
            cls = ltr[RM4[n][:, k]]
            pc.append((cls == c).mean())
            pm.append(np.bincount(cls, minlength=14).max() / 8)
    pur[f'{n} (fase 4, train completo)'] = dict(pureza_clase_concepto=float(np.median(pc)), pureza_clase_mayoritaria=float(np.median(pm)))
todos = np.array([k for ks in conc['EG'].values() for k in ks])
u, n_ = np.unique(todos, return_counts=True)
res = dict(n_muestra=len(mu), origen_filas=orig.groupby('gpu').filas.sum().to_dict(),
           repeticion_EG=dict(canales_distintos=int(len(u)), max_clases_por_canal=int(n_.max()), canal_mas_repetido=int(u[n_.argmax()]),
                              canales_en_3_o_mas_clases=[int(v) for v in u[n_ >= 3]]),
           solape_conceptos_por_clase=dict(EG_y_A_total=int(sol.EG_y_A.sum()), EG_y_GxE_total=int(sol.EG_y_GxE.sum()), A_y_GxE_total=int(sol.A_y_GxE.sum()), de=56),
           solape_refs_768_mediana=r768.drop(columns='canal').median().to_dict(),
           solape_refs_768_media=r768.drop(columns='canal').mean().to_dict(), pureza_mediana=pur)
json.dump(res, open(os.path.join(O, 'resumen_atlas.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
print(sol.to_string())
print('RELMAX_MUESTRA_TERMINADO', flush=True)
