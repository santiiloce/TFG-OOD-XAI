# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 8b, complemento: IC de la variante de STOOD-X 100 % con las imágenes saturadas (U = n1·n2) como empate real
(todas con la misma puntuación), junto a la puntuación original. Reutiliza fase8b.py sin modificarlo (mismas
funciones y misma matriz de bootstrap: por imagen, 1.000 réplicas, semilla 0). No modifica ningún fichero existente.

Salidas en results_corrupciones/fase8b/:
  stoodx_variantes_auroc_err_celda.csv, stoodx_variantes_diferencias_auroc_err.csv, stoodx_variantes_aurc.csv
"""
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

import fase8b as f8b
from fase8b import CELDAS, F8, SAL, SX, COMPARADOS, B, SEMILLA, auroc_err, aurc, ic, cargar

VARIANTES = {'original': 'bruta', 'empates_reales': 'saturadas_empatadas'}
VALOR_SATURADO = -38.6910841          # el mismo que usa fase8b.py para la variante
G = {}


def _reps(args):
    clave, celda = args
    s, err, idx = G['s'][clave][celda], G['err'][celda], G['idx']
    return clave, celda, auroc_err(s, err), np.array([auroc_err(s[i], err[i]) for i in idx])


def main():
    m, celdas, _ = cargar(F8, ['sub', 'resto'])
    n = len(m['label'])
    err = {k: celdas[k]['pred'] != m['label'] for k in CELDAS}
    sat = {k: celdas[k][f'{SX}_U'] == celdas[k][f'{SX}_n1n2'] for k in CELDAS}
    idx = np.random.default_rng(SEMILLA).integers(0, n, (B, n))
    s = {('original', d): {k: celdas[k][d].astype(np.float64) for k in CELDAS} for d in [SX] + COMPARADOS}
    emp = {}
    for k in CELDAS:
        x = celdas[k][SX].astype(np.float64).copy()
        x[sat[k]] = VALOR_SATURADO
        emp[k] = x
    s[('empates_reales', SX)] = emp
    G.update({'s': s, 'err': err, 'idx': idx})

    tareas = [(c, k) for c in s for k in CELDAS]
    with mp.get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 4))) as pool:
        res = pool.map(_reps, tareas, chunksize=2)
    punto = {(c, k): p for c, k, p, _ in res}
    reps = {(c, k): r for c, k, _, r in res}

    filas = []
    for v in VARIANTES:
        for k in CELDAS:
            lo, hi = ic(reps[(v, SX), k])
            filas.append({'detector': SX, 'variante': v, 'corrupcion': k[0], 'nivel': k[1],
                          'AUROC_err': punto[(v, SX), k], 'IC_inf': lo, 'IC_sup': hi, 'n': n,
                          'n_errores': int(err[k].sum()),
                          'pct_errores_saturados': 100 * sat[k][err[k]].mean(),
                          'pct_aciertos_saturados': 100 * sat[k][~err[k]].mean()})
    auc = pd.DataFrame(filas)
    auc.to_csv(os.path.join(SAL, 'stoodx_variantes_auroc_err_celda.csv'), index=False)

    filas = []
    for v in VARIANTES:
        for o in COMPARADOS:
            for k in CELDAS:
                dr = reps[(v, SX), k] - reps[('original', o), k]
                lo, hi = ic(dr)
                filas.append({'variante': v, 'comparacion': f'{SX} - {o}', 'corrupcion': k[0], 'nivel': k[1],
                              'diferencia': punto[(v, SX), k] - punto[('original', o), k], 'IC_inf': lo, 'IC_sup': hi,
                              'resultado': 'STOOD-X mejor' if lo > 0 else ('STOOD-X peor' if hi < 0 else 'no concluyente')})
    dif = pd.DataFrame(filas)
    dif.to_csv(os.path.join(SAL, 'stoodx_variantes_diferencias_auroc_err.csv'), index=False)

    E = np.stack([err[k] for k in CELDAS], 1)
    filas = []
    for v in VARIANTES:
        S = np.stack([s[(v, SX)][k] for k in CELDAS], 1)
        p = aurc(S, E, SX)
        r = np.array([aurc(S[i], E[i], SX) for i in idx])
        lo, hi = ic(r)
        filas.append({'detector': SX, 'variante': v, 'AURC': p, 'IC_inf': lo, 'IC_sup': hi, 'n_puntos': E.size})
    au = pd.DataFrame(filas)
    au.to_csv(os.path.join(SAL, 'stoodx_variantes_aurc.csv'), index=False)

    # Comprobación: las filas "original" reproducen los CSV ya guardados de fase8b
    prev = pd.read_csv(os.path.join(SAL, 'auroc_err_celda.csv'))
    prev = prev[prev.detector == SX].set_index(['corrupcion', 'nivel'])
    o = auc[auc.variante == 'original'].set_index(['corrupcion', 'nivel'])
    print('max |dif| AUROC_err e IC original vs auroc_err_celda.csv:',
          float(np.max(np.abs(o[['AUROC_err', 'IC_inf', 'IC_sup']].values
                              - prev.loc[o.index, ['AUROC_err', 'IC_inf', 'IC_sup']].values))))
    pe = prev.loc[o.index]
    e = auc[auc.variante == 'empates_reales'].set_index(['corrupcion', 'nivel'])
    print('max |dif| AUROC_err empates_reales vs columna previa:',
          float(np.max(np.abs(e.AUROC_err.values - pe.AUROC_err_saturadas_empatadas.values))),
          '| IC:', float(np.max(np.abs(e[['IC_inf', 'IC_sup']].values
                                       - pe[['IC_inf_saturadas_empatadas', 'IC_sup_saturadas_empatadas']].values))))
    pd_ = pd.read_csv(os.path.join(SAL, 'diferencias_auroc_err.csv')).set_index(['comparacion', 'corrupcion', 'nivel'])
    od = dif[dif.variante == 'original'].set_index(['comparacion', 'corrupcion', 'nivel'])
    print('max |dif| diferencias original vs diferencias_auroc_err.csv:',
          float(np.max(np.abs(od[['diferencia', 'IC_inf', 'IC_sup']].values
                              - pd_.loc[od.index, ['diferencia', 'IC_inf', 'IC_sup']].values))))
    pa = pd.read_csv(os.path.join(SAL, 'aurc.csv'))
    pa = pa[pa.detector == SX].set_index('variante')
    for v, w in VARIANTES.items():
        x = au.set_index('variante').loc[v]
        print(f'AURC {v}: {x.AURC:.6f} [{x.IC_inf:.6f}; {x.IC_sup:.6f}]  (aurc.csv {w}: {pa.loc[w].AURC:.6f} '
              f'[{pa.loc[w].IC_inf:.6f}; {pa.loc[w].IC_sup:.6f}])')
    print(dif.groupby(['variante', 'comparacion']).apply(
        lambda x: x[x.corrupcion != 'limpia'].resultado.value_counts().to_dict()).to_string())


if __name__ == '__main__':
    main()
