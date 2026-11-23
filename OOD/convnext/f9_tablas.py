# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9, pasos 3-4: métricas del ConvNeXt-S y comparación con el Swin-S (CPU, solo lectura fuera de results_ood_convnext).

Métricas y bootstrap con metricas_ood.py (las mismas funciones que la fase C): IC percentil 95 %, 1.000 remuestreos,
semilla 0, remuestreo estratificado ID/OOD; diferencia ConvNeXt − Swin con bootstrap PAREADO por imagen (mismos índices
para los dos modelos; las rutas se comprueban una a una).
Salida en results_ood_convnext/: tabla_principal_convnext.csv, tabla_sensibilidad.csv, lado_a_lado.csv,
diferencias_auroc.csv, kendall.csv, panda_vs_control.csv, saturacion_stoodx.csv, conclusiones.json
"""
import json, multiprocessing as mp, os, sys
import numpy as np, pandas as pd
from scipy import stats

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import metricas_ood as mo  # noqa: E402

R = os.environ.get('F9_SALIDA', os.path.join(RAIZ, 'results_ood_convnext'))
SWIN_A = os.environ.get('F9_SWIN_A') or os.path.join(RAIZ, 'results_ood/faseA_2026-10-20_11-19-35')
SWIN_B = os.environ.get('F9_SWIN_B') or os.path.join(RAIZ, 'results_ood/faseB_paper_2026-10-20')
SWIN_C = os.environ.get('F9_SWIN_C') or os.path.join(RAIZ, 'results_ood/faseC_2026-10-20/tabla_principal.csv')
NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
NB, SEM, K = 1000, 0, 500
CONJ = ['near_nct_tum_norm', 'near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
        'cov_panda', 'control_res_128', 'far_dtd']
COMUNES = ['msp', 'mls', 'ebo', 'react', 'ash', 'dice', 'knn', 'mds', 'rmds', 'vim', 'she', 'she_cosine',
           'she_euclidean', 'stoodx_100']          # vim: 171 en ConvNeXt (principal), 256 en Swin
DISTANCIA = ['knn', 'mds', 'rmds', 'vim', 'she_cosine', 'she_euclidean']


def cargar(carpeta, m):
    return dict(np.load(os.path.join(carpeta, f'scores_{m}.npz')))


def swin(m):
    return cargar(SWIN_B if m.startswith('stoodx') else SWIN_A, m)


def fila(s, m, c, espacio):
    cid, cood = s['id_test_conf'], s[f'{c}_conf']
    met = mo.todas_las_metricas(cid, cood, s['id_test_pred'], s['id_test_label'], 14)
    ic = mo.bootstrap_ic(cid, cood, NB, SEM)
    return {'espacio': espacio, 'metodo': m, 'conjunto': c, 'AUROC': met['AUROC'], 'AUROC_IC_inf': ic['AUROC_IC'][0],
            'AUROC_IC_sup': ic['AUROC_IC'][1], 'FPR@95': met['FPR@95'], 'FPR@95_IC_inf': ic['FPR@95_IC'][0],
            'FPR@95_IC_sup': ic['FPR@95_IC'][1], 'CCR@FPR5': met['CCR@FPR5'], 'AUPR_In': met['AUPR_In'],
            'AUPR_Out': met['AUPR_Out'], 'ACC_ID': mo.exactitud(s['id_test_pred'], s['id_test_label'])}


def tarea_tabla(args):
    carpeta, m, c, espacio = args
    return fila(cargar(carpeta, m), m, c, espacio)


def tarea_par(args):
    m, c = args
    a, b = cargar(os.path.join(R, 'espacio_B'), m), swin(m)
    for k in ('id_test', c):
        assert np.array_equal(a[f'{k}_paths'], b[f'{k}_paths']), f'orden distinto: {m} {k}'
    r = mo.bootstrap_pareado_auroc(a['id_test_conf'], a[f'{c}_conf'], b['id_test_conf'], b[f'{c}_conf'], NB, SEM)
    return {'metodo': m, 'conjunto': c, 'AUROC_convnext': mo.auroc(a['id_test_conf'], a[f'{c}_conf']),
            'AUROC_swin': mo.auroc(b['id_test_conf'], b[f'{c}_conf']), 'dif_convnext_menos_swin': r['diferencia'],
            'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1], 'p_bootstrap': r['p_bootstrap']}


def tarea_panda(args):
    modelo, m = args
    s = cargar(os.path.join(R, 'espacio_B'), m) if modelo == 'convnext' else swin(m)
    # control como "negativa" y PANDA como "positiva" (AUROC > 0,5: PANDA más OOD que el control)
    a = mo.auroc(s['control_res_128_conf'], s['cov_panda_conf'])
    ic = mo.bootstrap_ic(s['control_res_128_conf'], s['cov_panda_conf'], NB, SEM)['AUROC_IC']
    return {'modelo': modelo, 'metodo': m, 'AUROC_panda_vs_control': a, 'IC_inf': ic[0], 'IC_sup': ic[1]}


def saturacion(s, n_c, modelo, espacio):
    z_max = lambda n1, n2: (n1 * n2 / 2 - 0.5) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    filas = []
    for c in ['id_test'] + CONJ:
        if f'{c}_U' not in s: continue
        pred = s[f'{c}_pred']
        n1 = np.array([min(K, n_c[p]) for p in pred]); n2 = n1 * np.array([min(K, n_c[p] - 1) for p in pred])
        sat = s[f'{c}_U'] == n1 * n2
        filas.append({'modelo': modelo, 'espacio': espacio, 'conjunto': c, 'n': len(pred),
                      'frac_saturadas': float(sat.mean()), 'min_menos_z': float(s[f'{c}_conf'].min()),
                      'menos_z_max_teorico_K500': -z_max(K, K * K),
                      'frac_pred_clase_nc_le_K': float(np.mean([n_c[p] <= K for p in pred]))})
    return filas


def main():
    B, C = os.path.join(R, 'espacio_B'), os.path.join(R, 'espacio_C')
    metodos_B = COMUNES + ['vim_256']
    trabajos = [(B, m, c, 'B') for m in metodos_B for c in CONJ]
    sens_C = [m for m in ['knn', 'mds', 'rmds', 'stoodx_100'] if os.path.exists(os.path.join(C, f'scores_{m}.npz'))]
    for m in sens_C:
        cs = [c for c in CONJ if f'{c}_conf' in np.load(os.path.join(C, f'scores_{m}.npz')).files]
        trabajos += [(C, m, c, 'C') for c in cs]
    with mp.get_context('fork').Pool(NCPU) as pool:
        tabla = pd.DataFrame(pool.map(tarea_tabla, trabajos, chunksize=1))
        tabla[(tabla.espacio == 'B') & (tabla.metodo != 'vim_256')].to_csv(os.path.join(R, 'tabla_principal_convnext.csv'), index=False)
        tabla[(tabla.espacio == 'C') | (tabla.metodo == 'vim_256')].to_csv(os.path.join(R, 'tabla_sensibilidad.csv'), index=False)
        print('tablas hechas', flush=True)
        dif = pd.DataFrame(pool.map(tarea_par, [(m, c) for m in COMUNES for c in CONJ], chunksize=1))
        dif.to_csv(os.path.join(R, 'diferencias_auroc.csv'), index=False)
        print('pareado hecho', flush=True)
        pc = pd.DataFrame(pool.map(tarea_panda, [(mod, m) for mod in ['convnext', 'swin'] for m in COMUNES]))
        pc.to_csv(os.path.join(R, 'panda_vs_control.csv'), index=False)

    # Lado a lado (Swin: tabla de la fase C, con sus IC)
    sw = pd.read_csv(SWIN_C)
    cv = tabla[(tabla.espacio == 'B') & tabla.metodo.isin(COMUNES)]
    lado = cv.merge(sw[['metodo', 'conjunto', 'AUROC', 'AUROC_IC_inf', 'AUROC_IC_sup', 'FPR@95', 'FPR@95_IC_inf',
                        'FPR@95_IC_sup', 'CCR@FPR5']], on=['metodo', 'conjunto'], suffixes=('_convnext', '_swin'))
    lado.to_csv(os.path.join(R, 'lado_a_lado.csv'), index=False)

    # Kendall entre las ordenaciones por AUROC (14 métodos comunes) en cada conjunto
    kd = []
    for c in CONJ:
        d = lado[lado.conjunto == c].set_index('metodo').loc[COMUNES]
        t, p = stats.kendalltau(d['AUROC_convnext'], d['AUROC_swin'])
        t_sin, p_sin = stats.kendalltau(d.drop(index=['ash', 'dice'])['AUROC_convnext'],
                                        d.drop(index=['ash', 'dice'])['AUROC_swin'])
        kd.append({'conjunto': c, 'tau': t, 'p': p, 'n_metodos': len(d), 'tau_sin_ash_dice': t_sin, 'p_sin_ash_dice': p_sin,
                   'orden_convnext': ' > '.join(d['AUROC_convnext'].sort_values(ascending=False).index),
                   'orden_swin': ' > '.join(d['AUROC_swin'].sort_values(ascending=False).index)})
    pd.DataFrame(kd).to_csv(os.path.join(R, 'kendall.csv'), index=False)

    # Saturación de STOOD-X
    nc_conv = {int(k): v for k, v in json.load(open(os.path.join(B, 'config_stoodx.json')))['n_c_train_predicha'].items()}
    lg = np.load(os.path.join(RAIZ, 'features/completo/id_train.npz'))['logits'].argmax(1)
    nc_swin = {c: int((lg == c).sum()) for c in range(14)}
    sat = saturacion(cargar(B, 'stoodx_100'), nc_conv, 'convnext', 'B') + saturacion(swin('stoodx_100'), nc_swin, 'swin', '-')
    if 'stoodx_100' in sens_C:
        sat += saturacion(cargar(C, 'stoodx_100'), nc_conv, 'convnext', 'C')
    pd.DataFrame(sat).to_csv(os.path.join(R, 'saturacion_stoodx.csv'), index=False)
    json.dump({'n_c_convnext': nc_conv, 'n_c_swin': nc_swin}, open(os.path.join(R, 'n_c_train.json'), 'w'), indent=1)
    print('hecho', flush=True)


if __name__ == '__main__':
    main()
