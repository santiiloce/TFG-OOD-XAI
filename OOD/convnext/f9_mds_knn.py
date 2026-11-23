# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9: MDS − KNN (AUROC) pareado por imagen en cada modelo (1.000 remuestreos, semilla 0) -> mds_menos_knn.csv"""
import os, sys
import numpy as np, pandas as pd
RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import metricas_ood as mo  # noqa: E402
R = os.path.join(RAIZ, 'results_ood_convnext')
CARP = {'convnext': os.path.join(R, 'espacio_B'), 'swin': os.path.join(RAIZ, 'results_ood/faseA_2026-10-20_11-19-35')}
CONJ = ['near_nct_tum_norm', 'near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
        'cov_panda', 'control_res_128', 'far_dtd']
filas = []
for mod, carp in CARP.items():
    a, b = np.load(os.path.join(carp, 'scores_mds.npz')), np.load(os.path.join(carp, 'scores_knn.npz'))
    for c in CONJ:
        for k in ('id_test', c):
            assert np.array_equal(a[f'{k}_paths'], b[f'{k}_paths'])
        r = mo.bootstrap_pareado_auroc(a['id_test_conf'], a[f'{c}_conf'], b['id_test_conf'], b[f'{c}_conf'], 1000, 0)
        filas.append({'modelo': mod, 'conjunto': c, 'AUROC_mds': mo.auroc(a['id_test_conf'], a[f'{c}_conf']),
                      'AUROC_knn': mo.auroc(b['id_test_conf'], b[f'{c}_conf']), 'mds_menos_knn': r['diferencia'],
                      'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1]})
d = pd.DataFrame(filas); d.to_csv(os.path.join(R, 'mds_menos_knn.csv'), index=False)
print(d.round(4).to_string())
