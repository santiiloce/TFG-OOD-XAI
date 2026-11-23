# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
STOOD-X 100 % con otro K sobre las features ya guardadas de las celdas (limpia + 9 corrupciones × niveles 1/3/5,
test completo). ID = limpia, OOD = celda corrupta (misma convención que fase8_analizar.auroc_fpr).
Compara FPR@95 con K = 500 y con el mejor del resto de detectores (metricas_celda.csv).

    sbatch scripts/script_cpu_dionisio.sh results_corrupciones/scripts/fase8_stoodx_k.py --K 215
Salida: results_corrupciones/stoodx_K<K>/{puntuaciones.npz, comparacion_fpr95.csv}
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
import torch

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import corrupciones_zhang as cz        # noqa: E402
import fase8_analizar as fa            # noqa: E402
import stoodx_paper as sx              # noqa: E402
from fase8_extraer import _init_pool   # noqa: E402

SAL = os.path.join(RAIZ, 'results_corrupciones')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--K', type=int, required=True)
    K = p.parse_args().K
    out = os.path.join(SAL, f'stoodx_K{K}')
    os.makedirs(out, exist_ok=True)
    m, celdas, _ = fa.cargar(SAL, ['sub', 'resto'])
    train = dict(np.load(os.path.join(RAIZ, 'features/completo/id_train.npz')))
    clases = sx.preparar_clases(train, sx.CONFIGS['100']['quantile'], torch.device('cpu'))
    pool = mp.get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)), initializer=_init_pool)

    # Comprobación: K = 500 reproduce las puntuaciones guardadas (200 primeras imágenes limpias)
    l0 = celdas[('limpia', 0)]
    r = sx.puntuar({'feats': l0['feats'][:200].astype(np.float32), 'logits': l0['logits'][:200]},
                   clases, sx.precalcular(clases, 500, 'fija'), 500, pool)
    dif = np.max(np.abs(-r['z'] - l0['stoodx_100'][:200]))
    print(f'comprobación K=500: max |dif| = {dif:.2e}', flush=True)
    assert dif < 1e-4   # ruido float32 (~1e-6 observado)

    tablas = sx.precalcular(clases, K, 'fija')
    s = {}
    for k, d in celdas.items():
        r = sx.puntuar({'feats': d['feats'].astype(np.float32), 'logits': d['logits']}, clases, tablas, K, pool)
        s[f'{k[0]}_{k[1]}'] = -r['z']
        print(k, flush=True)
    np.savez(os.path.join(out, 'puntuaciones.npz'), idx=m['idx'], **s)

    met = pd.read_csv(os.path.join(SAL, 'metricas_celda.csv'))
    filas = []
    for c in cz.CORRUPCIONES:
        for n in cz.NIVELES:
            a, f = fa.auroc_fpr(s['limpia_0'], s[f'{c}_{n}'])
            cel = met[(met.corrupcion == c) & (met.nivel == n)].set_index('detector')
            otros = cel.drop(index=['stoodx_100', 'stoodx_875'])
            mejor = otros.FPR95.idxmin()
            filas.append(dict(corrupcion=c, nivel=n, FPR95_K500=cel.loc['stoodx_100', 'FPR95'], **{f'FPR95_K{K}': f},
                              mejor_otro=mejor, FPR95_mejor_otro=otros.FPR95.min(),
                              gana_K500=cel.loc['stoodx_100', 'FPR95'] < otros.FPR95.min(),
                              **{f'gana_K{K}': f < otros.FPR95.min()},
                              AUROC_K500=cel.loc['stoodx_100', 'AUROC'], **{f'AUROC_K{K}': a}))
    d = pd.DataFrame(filas)
    d.to_csv(os.path.join(out, 'comparacion_fpr95.csv'), index=False)
    print(d.to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    main()
