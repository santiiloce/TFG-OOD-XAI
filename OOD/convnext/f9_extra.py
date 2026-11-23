# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9, complemento de f9_tablas.py (CPU, solo lee results_ood_convnext y results_ood):
- sensibilidad_pareada.csv: diferencia de AUROC pareada por imagen (1.000 remuestreos, semilla 0) C − B
  (KNN, MDS, RMDS y STOOD-X) y vim_256 − vim (171), en el ConvNeXt.
- conclusiones_cifras.json: cifras para comprobar las conclusiones del Swin en los dos modelos.
"""
import json, os, sys
import numpy as np, pandas as pd

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import metricas_ood as mo  # noqa: E402

R = os.path.join(RAIZ, 'results_ood_convnext')
CONJ = ['near_nct_tum_norm', 'near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
        'cov_panda', 'control_res_128', 'far_dtd']
DIST = ['knn', 'mds', 'rmds', 'vim', 'she_cosine', 'she_euclidean', 'stoodx_100']
cargar = lambda esp, m: np.load(os.path.join(R, f'espacio_{esp}', f'scores_{m}.npz'))

filas = []
pares = [('C', m, 'B', m) for m in ['knn', 'mds', 'rmds', 'stoodx_100']] + [('B', 'vim_256', 'B', 'vim')]
for ea, ma, eb, mb in pares:
    a, b = cargar(ea, ma), cargar(eb, mb)
    for c in CONJ:
        if f'{c}_conf' not in a.files:
            continue
        for k in ('id_test', c):
            assert np.array_equal(a[f'{k}_paths'], b[f'{k}_paths'])
        r = mo.bootstrap_pareado_auroc(a['id_test_conf'], a[f'{c}_conf'], b['id_test_conf'], b[f'{c}_conf'], 1000, 0)
        filas.append({'a': f'{ma}@{ea}', 'b': f'{mb}@{eb}', 'conjunto': c, 'AUROC_a': mo.auroc(a['id_test_conf'], a[f'{c}_conf']),
                      'AUROC_b': mo.auroc(b['id_test_conf'], b[f'{c}_conf']), 'dif_a_menos_b': r['diferencia'],
                      'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1]})
pd.DataFrame(filas).to_csv(os.path.join(R, 'sensibilidad_pareada.csv'), index=False)

lado = pd.read_csv(os.path.join(R, 'lado_a_lado.csv'))
out = {}
for mod in ['convnext', 'swin']:
    A = lado.pivot(index='metodo', columns='conjunto', values=f'AUROC_{mod}')
    F = lado.pivot(index='metodo', columns='conjunto', values=f'FPR@95_{mod}')
    o = {}
    o['top3_AUROC_tum_norm'] = A['near_nct_tum_norm'].sort_values(ascending=False).head(3).round(4).to_dict()
    o['rango_vim_mds_tum_norm'] = {m: int(A['near_nct_tum_norm'].rank(ascending=False)[m]) for m in ['vim', 'mds']}
    o['top3_FPR95_tum_norm'] = F['near_nct_tum_norm'].sort_values().head(3).round(4).to_dict()
    o['rmds_lym'] = {'AUROC': round(A.loc['rmds', 'afin_nct_lym'], 4), 'FPR@95': round(F.loc['rmds', 'afin_nct_lym'], 4)}
    o['mejor_AUROC_por_conjunto'] = {c: [A[c].idxmax(), round(A[c].max(), 4)] for c in CONJ}
    o['rango_stoodx'] = {c: int(A[c].rank(ascending=False)['stoodx_100']) for c in CONJ}
    o['msp_gt_mls_gt_ebo_conjuntos'] = int(sum((A.loc['msp', c] > A.loc['mls', c] > A.loc['ebo', c]) for c in CONJ))
    o['n_conjuntos_menor_que_msp'] = {m: int(sum(A.loc[m, c] < A.loc['msp', c] for c in CONJ)) for m in ['react', 'ash', 'dice']}
    o['n_conjuntos_menor_que_ebo'] = {m: int(sum(A.loc[m, c] < A.loc['ebo', c] for c in CONJ)) for m in ['react', 'ash', 'dice']}
    o['lym_menos_tum_norm'] = {m: round(A.loc[m, 'afin_nct_lym'] - A.loc[m, 'near_nct_tum_norm'], 4) for m in DIST + ['msp']}
    o['panda_menos_control'] = {m: round(A.loc[m, 'cov_panda'] - A.loc[m, 'control_res_128'], 4) for m in DIST}
    out[mod] = o
json.dump(out, open(os.path.join(R, 'conclusiones_cifras.json'), 'w'), indent=1, default=float)
print(pd.DataFrame(filas).round(4).to_string())
print(json.dumps(out, indent=1, default=float))
