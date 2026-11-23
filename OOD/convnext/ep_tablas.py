# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9b, paso 3 (CPU): tablas con M como espacio principal de los métodos de características.

Fuentes por método: logits (MSP, MaxLogit, Energy) y B (ViM 171, ReAct, ASH, DICE) de results_ood_convnext/espacio_B;
M (KNN, MDS, RMDS, SHE x3, STOOD-X) de espacio_principal/espacio_M. Mismas funciones, 1.000 remuestreos y semilla 0
que la fase 9 (f9_tablas.py y metricas_ood.py importados sin modificar).
CSV con coma decimal, IC también como texto "[a; b]" y columna `espacio`.
"""
import json, multiprocessing as mp, os, sys
import numpy as np, pandas as pd
from scipy import stats

F9 = '/mnt/homeGPU/slcerro/OpenOOD/results_ood_convnext'
EP = os.path.join(F9, 'espacio_principal')
sys.path.insert(0, os.path.join(F9, 'scripts'))
import f9_tablas as ft      # noqa: E402
mo = ft.mo
NB, SEM, CONJ, COMUNES = ft.NB, ft.SEM, ft.CONJ, ft.COMUNES
B, C, M, CSHE = (os.path.join(F9, 'espacio_B'), os.path.join(F9, 'espacio_C'), os.path.join(EP, 'espacio_M'),
                 os.path.join(EP, 'espacio_C_she'))
EN_M = ['knn', 'mds', 'rmds', 'she', 'she_cosine', 'she_euclidean', 'stoodx_100']
FUENTE = {**{m: ('logits', B) for m in ['msp', 'mls', 'ebo']}, **{m: ('B', B) for m in ['vim', 'react', 'ash', 'dice']},
          **{m: ('M', M) for m in EN_M}}
CARPETA_C = {**{m: C for m in ['knn', 'mds', 'rmds', 'stoodx_100']}, **{m: CSHE for m in ['she', 'she_cosine', 'she_euclidean']}}
DIST = ['knn', 'mds', 'rmds', 'vim', 'she_cosine', 'she_euclidean']


def conv(m):
    return ft.cargar(FUENTE[m][1], m)


def conjuntos(carpeta, m):
    f = np.load(os.path.join(carpeta, f'scores_{m}.npz')).files
    return [c for c in CONJ if f'{c}_conf' in f]


def par(a, b, c):
    for k in ('id_test', c):
        assert np.array_equal(a[f'{k}_paths'], b[f'{k}_paths']), f'orden distinto en {k}'
    r = mo.bootstrap_pareado_auroc(a['id_test_conf'], a[f'{c}_conf'], b['id_test_conf'], b[f'{c}_conf'], NB, SEM)
    return mo.auroc(a['id_test_conf'], a[f'{c}_conf']), mo.auroc(b['id_test_conf'], b[f'{c}_conf']), r


def t_tabla(x):
    carpeta, m, c, esp = x
    return ft.fila(ft.cargar(carpeta, m), m, c, esp)


def t_swin(x):
    m, c = x
    aa, ab, r = par(conv(m), ft.swin(m), c)
    return {'espacio': FUENTE[m][0], 'metodo': m, 'conjunto': c, 'AUROC_convnext': aa, 'AUROC_swin': ab,
            'dif_convnext_menos_swin': r['diferencia'], 'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1],
            'p_bootstrap': r['p_bootstrap']}


def t_sens(x):
    m, eb, cb, c = x
    aa, ab, r = par(ft.cargar(M, m), ft.cargar(cb, m), c)
    return {'espacio': 'M', 'a': f'{m}@M', 'b': f'{m}@{eb}', 'metodo': m, 'conjunto': c, 'AUROC_a': aa, 'AUROC_b': ab,
            'dif_a_menos_b': r['diferencia'], 'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1]}


def t_panda(x):
    mod, m = x
    s = conv(m) if mod == 'convnext' else ft.swin(m)
    ic = mo.bootstrap_ic(s['control_res_128_conf'], s['cov_panda_conf'], NB, SEM)['AUROC_IC']
    return {'modelo': mod, 'espacio': FUENTE[m][0] if mod == 'convnext' else '-', 'metodo': m,
            'AUROC_panda_vs_control': mo.auroc(s['control_res_128_conf'], s['cov_panda_conf']),
            'IC_inf': ic[0], 'IC_sup': ic[1]}


def t_mdsknn(x):
    mod, c = x
    a, b = (ft.cargar(M, 'mds'), ft.cargar(M, 'knn')) if mod == 'convnext' else (ft.swin('mds'), ft.swin('knn'))
    aa, ab, r = par(a, b, c)
    return {'modelo': mod, 'espacio': 'M' if mod == 'convnext' else '-', 'conjunto': c, 'AUROC_mds': aa,
            'AUROC_knn': ab, 'mds_menos_knn': r['diferencia'], 'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1]}


def guardar(df, nombre):
    """Coma decimal; cada par X_IC_inf/X_IC_sup se acompaña de X_IC = "[a; b]"."""
    df = df.copy()
    for col in [c for c in df.columns if 'IC_inf' in c]:
        sup = col.replace('IC_inf', 'IC_sup')
        df[col.replace('IC_inf', 'IC')] = [f'[{a:.4f}; {b:.4f}]'.replace('.', ',') for a, b in zip(df[col], df[sup])]
    df.to_csv(os.path.join(EP, nombre), index=False, decimal=',')
    return df


def main():
    principal = [(FUENTE[m][1], m, c, FUENTE[m][0]) for m in COMUNES for c in CONJ]
    sens = [(B, m, c, 'B') for m in EN_M for c in CONJ] + [(B, 'vim_256', c, 'B') for c in CONJ]
    sens += [(CARPETA_C[m], m, c, 'C') for m in EN_M for c in conjuntos(CARPETA_C[m], m)]
    pares = [(m, 'B', B, c) for m in EN_M for c in CONJ]
    pares += [(m, 'C', CARPETA_C[m], c) for m in EN_M for c in conjuntos(CARPETA_C[m], m)]
    with mp.get_context('fork').Pool(ft.NCPU) as pool:
        tabla = pd.DataFrame(pool.map(t_tabla, principal, chunksize=1))
        tsens = pd.DataFrame(pool.map(t_tabla, sens, chunksize=1))
        print('tablas', flush=True)
        dif = pd.DataFrame(pool.map(t_swin, [(m, c) for m in COMUNES for c in CONJ], chunksize=1))
        sp = pd.DataFrame(pool.map(t_sens, pares, chunksize=1))
        pc = pd.DataFrame(pool.map(t_panda, [(mod, m) for mod in ['convnext', 'swin'] for m in COMUNES]))
        mk = pd.DataFrame(pool.map(t_mdsknn, [(mod, c) for mod in ['convnext', 'swin'] for c in CONJ]))
        print('pareados', flush=True)
    guardar(tabla, 'tabla_principal_convnext.csv'); guardar(tsens, 'tabla_sensibilidad.csv')
    guardar(dif, 'diferencias_auroc.csv'); guardar(sp, 'sensibilidad_pareada.csv')
    guardar(pc, 'panda_vs_control.csv'); guardar(mk, 'mds_menos_knn.csv')

    sw = pd.read_csv(ft.SWIN_C)
    lado = tabla.merge(sw[['metodo', 'conjunto', 'AUROC', 'AUROC_IC_inf', 'AUROC_IC_sup', 'FPR@95', 'FPR@95_IC_inf',
                           'FPR@95_IC_sup', 'CCR@FPR5']], on=['metodo', 'conjunto'], suffixes=('_convnext', '_swin'))
    guardar(lado, 'lado_a_lado.csv')

    kd = []
    for c in CONJ:
        d = lado[lado.conjunto == c].set_index('metodo').loc[COMUNES]
        t, p = stats.kendalltau(d['AUROC_convnext'], d['AUROC_swin'])
        s = d.drop(index=['ash', 'dice'])
        t2, p2 = stats.kendalltau(s['AUROC_convnext'], s['AUROC_swin'])
        kd.append({'espacio': 'M (características), B (ViM/ReAct/ASH/DICE), logits', 'conjunto': c, 'tau': t, 'p': p,
                   'n_metodos': len(d), 'tau_sin_ash_dice': t2, 'p_sin_ash_dice': p2,
                   'orden_convnext': ' > '.join(d['AUROC_convnext'].sort_values(ascending=False).index),
                   'orden_swin': ' > '.join(d['AUROC_swin'].sort_values(ascending=False).index)})
    guardar(pd.DataFrame(kd), 'kendall.csv')

    nc = {int(k): v for k, v in json.load(open(os.path.join(B, 'config_stoodx.json')))['n_c_train_predicha'].items()}
    nc_m = {int(k): v for k, v in json.load(open(os.path.join(M, 'config_stoodx.json')))['n_c_train_predicha'].items()}
    assert nc == nc_m
    nc_sw = {int(k): v for k, v in json.load(open(os.path.join(F9, 'n_c_train.json')))['n_c_swin'].items()}
    sat = (ft.saturacion(ft.cargar(M, 'stoodx_100'), nc, 'convnext', 'M') + ft.saturacion(ft.cargar(B, 'stoodx_100'), nc, 'convnext', 'B')
           + ft.saturacion(ft.cargar(C, 'stoodx_100'), nc, 'convnext', 'C') + ft.saturacion(ft.swin('stoodx_100'), nc_sw, 'swin', '-'))
    guardar(pd.DataFrame(sat), 'saturacion_stoodx.csv')

    # Cifras para las conclusiones del Swin (mismo cálculo que f9_extra.py, ahora con M)
    out = {}
    for mod in ['convnext', 'swin']:
        A = lado.pivot(index='metodo', columns='conjunto', values=f'AUROC_{mod}')
        F = lado.pivot(index='metodo', columns='conjunto', values=f'FPR@95_{mod}')
        o = {'top5_AUROC_tum_norm': A['near_nct_tum_norm'].sort_values(ascending=False).head(5).round(4).to_dict(),
             'rango_AUROC_tum_norm': A['near_nct_tum_norm'].rank(ascending=False).astype(int).to_dict(),
             'top3_FPR95_tum_norm': F['near_nct_tum_norm'].sort_values().head(3).round(4).to_dict(),
             'rmds_lym': {'AUROC': round(A.loc['rmds', 'afin_nct_lym'], 4), 'FPR@95': round(F.loc['rmds', 'afin_nct_lym'], 4)},
             'mejor_AUROC_por_conjunto': {c: [A[c].idxmax(), round(A[c].max(), 4)] for c in CONJ},
             'top3_por_conjunto': {c: A[c].sort_values(ascending=False).head(3).round(4).to_dict() for c in CONJ},
             'rango_stoodx': {c: int(A[c].rank(ascending=False)['stoodx_100']) for c in CONJ},
             'msp_gt_mls_gt_ebo_conjuntos': int(sum(A.loc['msp', c] > A.loc['mls', c] > A.loc['ebo', c] for c in CONJ)),
             'n_menor_que_msp': {m: int(sum(A.loc[m, c] < A.loc['msp', c] for c in CONJ)) for m in ['react', 'ash', 'dice']},
             'n_menor_que_ebo': {m: int(sum(A.loc[m, c] < A.loc['ebo', c] for c in CONJ)) for m in ['react', 'ash', 'dice']},
             'lym_menos_tum_norm': {m: round(A.loc[m, 'afin_nct_lym'] - A.loc[m, 'near_nct_tum_norm'], 4) for m in DIST + ['stoodx_100', 'msp']},
             'panda_menos_control': {m: round(A.loc[m, 'cov_panda'] - A.loc[m, 'control_res_128'], 4) for m in DIST + ['stoodx_100']}}
        out[mod] = o
    fi = tabla.set_index(['metodo', 'conjunto'])
    out['convnext_FPR95_IC_tum_norm'] = {m: [round(fi.loc[(m, 'near_nct_tum_norm'), k], 4) for k in ('FPR@95', 'FPR@95_IC_inf', 'FPR@95_IC_sup')]
                                          for m in COMUNES}
    out['convnext_AUROC_IC'] = {f'{m}|{c}': [round(fi.loc[(m, c), k], 4) for k in ('AUROC', 'AUROC_IC_inf', 'AUROC_IC_sup')]
                                for m in ['knn', 'stoodx_100', 'mds', 'rmds', 'msp', 'vim'] for c in CONJ}
    json.dump(out, open(os.path.join(EP, 'conclusiones_cifras.json'), 'w'), indent=1, default=float)
    print('hecho', flush=True)


if __name__ == '__main__':
    main()
