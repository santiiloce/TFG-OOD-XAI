# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 8b — ¿Rechazan los detectores las imágenes corruptas mal clasificadas? Solo sobre lo guardado en la fase 8.

Entradas (solo lectura): results_corrupciones/muestra_{sub,resto}.npz (etiquetas), celdas/{sub,resto}/*.npz
(predicción y puntuaciones por imagen y celda), results_ood/ (puntuaciones de id_val para el umbral).
Salidas: results_corrupciones/fase8b/.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
F8 = os.path.join(RAIZ, 'results_corrupciones')
SAL = os.path.join(F8, 'fase8b')
sys.path.insert(0, os.path.join(F8, 'scripts'))
from fase8_analizar import cargar, NOMBRES_DET, DETECTORES, PRINCIPALES   # noqa: E402
import corrupciones_zhang as cz                                            # noqa: E402

FASEA = os.path.join(RAIZ, 'results_ood/faseA_2026-10-20_11-19-35')
FASEB = os.path.join(RAIZ, 'results_ood/faseB_paper_2026-10-20')
B = 1000
SEMILLA = 0
N_PROC = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
CELDAS = [('limpia', 0)] + [(c, n) for c in cz.CORRUPCIONES for n in cz.NIVELES]
SX = 'stoodx_100'
COMPARADOS = ['msp', 'vim', 'mds']


def auroc_err(s, err):
    """AUROC con positivo = error y puntuación de error = -s. Rangos medios: empates exactos cuentan ½."""
    n1 = err.sum()
    n0 = len(err) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(-s)
    return (r[err].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def ic(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return np.percentile(x, [2.5, 97.5]) if len(x) else (np.nan, np.nan)


# ---------------------------------------------------------------------------------------------------------------
G = {}   # datos compartidos con los procesos (fork)


def _tarea_auroc(args):
    det, celda, variante = args
    s = G['s'][variante][det][celda]
    err = G['err'][celda]
    idx = G['idx']
    punto = auroc_err(s, err)
    reps = np.array([auroc_err(s[i], err[i]) for i in idx])
    fila = {'detector': det, 'corrupcion': celda[0], 'nivel': celda[1], 'variante': variante,
            'AUROC_err': punto, 'reps': reps}
    return fila


def main():
    os.makedirs(os.path.join(SAL, 'figuras'), exist_ok=True)
    m, celdas, _ = cargar(F8, ['sub', 'resto'])
    n = len(m['label'])
    label = m['label']
    err = {k: celdas[k]['pred'] != label for k in CELDAS}
    sat = {k: celdas[k][f'{SX}_U'] == celdas[k][f'{SX}_n1n2'] for k in CELDAS}
    sat875 = {k: celdas[k]['stoodx_875_U'] == celdas[k]['stoodx_875_n1n2'] for k in CELDAS}
    idx = np.random.default_rng(SEMILLA).integers(0, n, (B, n))

    # Variante de sensibilidad: las saturadas de STOOD-X con una puntuación común exacta (empate real, ½ en el AUROC)
    s_bruta = {d: {k: celdas[k][d].astype(np.float64) for k in CELDAS} for d in DETECTORES}
    s_emp = {d: dict(v) for d, v in s_bruta.items()}
    for d, st in [(SX, sat), ('stoodx_875', sat875)]:
        for k in CELDAS:
            x = s_emp[d][k].copy()
            x[st[k]] = -38.6910841
            s_emp[d][k] = x
    G.update({'s': {'bruta': s_bruta, 'saturadas_empatadas': s_emp}, 'err': err, 'idx': idx})

    # 0. Comprobaciones
    comp = {'n_imagenes': int(n), 'celdas': len(CELDAS), 'detectores': DETECTORES,
            'fuentes': {'etiquetas': 'results_corrupciones/muestra_{sub,resto}.npz (label)',
                        'prediccion_y_puntuaciones': 'results_corrupciones/celdas/{sub,resto}/<corrupcion>_<nivel>.npz '
                                                     '(pred, <detector>, stoodx_*_U, stoodx_*_n1n2)',
                        'id_val': 'results_ood/faseA_2026-10-20_11-19-35/scores_<m>.npz y '
                                  'results_ood/faseB_paper_2026-10-20/scores_stoodx_*.npz (id_val_conf)'},
            'finitas': bool(all(np.isfinite(s_bruta[d][k]).all() for d in DETECTORES for k in CELDAS))}
    val = {}
    for d in DETECTORES:
        f = os.path.join(FASEB if d.startswith('stoodx') else FASEA, f'scores_{d}.npz')
        val[d] = np.load(f)['id_val_conf'].astype(np.float64)
    comp['id_val_n'] = {d: int(len(v)) for d, v in val.items()}
    comp['id_val_finitas'] = bool(all(np.isfinite(v).all() for v in val.values()))
    comp['stoodx_pct_saturadas_id_val'] = {d: float(100 * np.mean(val[d] <= -38.6910)) for d in ['stoodx_100', 'stoodx_875']}
    # coherencia: exactitud limpia y errores
    comp['exactitud_limpia'] = float(1 - err[('limpia', 0)].mean())
    json.dump(comp, open(os.path.join(SAL, 'comprobaciones.json'), 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(comp, indent=1, ensure_ascii=False), flush=True)

    # 1. AUROC_err por celda (+ variante con saturadas empatadas para STOOD-X)
    tareas = [(d, k, 'bruta') for d in DETECTORES for k in CELDAS]
    tareas += [(d, k, 'saturadas_empatadas') for d in ['stoodx_100', 'stoodx_875'] for k in CELDAS]
    with mp.get_context('fork').Pool(N_PROC) as pool:
        res = pool.map(_tarea_auroc, tareas, chunksize=2)
    reps = {(r['detector'], r['corrupcion'], r['nivel'], r['variante']): r['reps'] for r in res}
    filas = []
    for r in res:
        if r['variante'] != 'bruta':
            continue
        k = (r['corrupcion'], r['nivel'])
        lo, hi = ic(r['reps'])
        fila = {'detector': r['detector'], 'corrupcion': k[0], 'nivel': k[1], 'AUROC_err': r['AUROC_err'],
                'IC_inf': lo, 'IC_sup': hi, 'n': n, 'n_errores': int(err[k].sum()),
                'pct_errores': 100 * err[k].mean()}
        if r['detector'].startswith('stoodx'):
            st = sat if r['detector'] == SX else sat875
            e = reps[(r['detector'], k[0], k[1], 'saturadas_empatadas')]
            fila.update({'pct_errores_saturados': 100 * st[k][err[k]].mean() if err[k].any() else np.nan,
                         'pct_aciertos_saturados': 100 * st[k][~err[k]].mean(),
                         'AUROC_err_saturadas_empatadas': next(x['AUROC_err'] for x in res if x['detector'] == r['detector']
                                                               and (x['corrupcion'], x['nivel']) == k
                                                               and x['variante'] == 'saturadas_empatadas'),
                         'IC_inf_saturadas_empatadas': ic(e)[0], 'IC_sup_saturadas_empatadas': ic(e)[1]})
        filas.append(fila)
    auc = pd.DataFrame(filas)
    auc.to_csv(os.path.join(SAL, 'auroc_err_celda.csv'), index=False)

    difs = []
    for o in COMPARADOS:
        for k in CELDAS:
            a = auc[(auc.detector == SX) & (auc.corrupcion == k[0]) & (auc.nivel == k[1])].AUROC_err.iloc[0]
            b = auc[(auc.detector == o) & (auc.corrupcion == k[0]) & (auc.nivel == k[1])].AUROC_err.iloc[0]
            dr = reps[(SX, k[0], k[1], 'bruta')] - reps[(o, k[0], k[1], 'bruta')]
            lo, hi = ic(dr)
            difs.append({'comparacion': f'{SX} - {o}', 'corrupcion': k[0], 'nivel': k[1], 'diferencia': a - b,
                         'IC_inf': lo, 'IC_sup': hi,
                         'resultado': 'STOOD-X mejor' if lo > 0 else ('STOOD-X peor' if hi < 0 else 'no concluyente')})
    dif = pd.DataFrame(difs)
    dif.to_csv(os.path.join(SAL, 'diferencias_auroc_err.csv'), index=False)

    # 2. Umbral fijo: percentil 5 de id_val (principal) y del test limpio (comprobación); aceptar = s >= umbral
    filas = []
    for d in DETECTORES:
        umbrales = {'id_val': float(np.quantile(val[d], 0.05)),
                    'test_limpio': float(np.quantile(s_bruta[d][('limpia', 0)], 0.05))}
        for origen, t in umbrales.items():
            for k in CELDAS:
                acc = s_bruta[d][k] >= t
                ok = ~err[k]
                A, C = acc[idx], ok[idx]
                na = A.sum(1)
                nr = n - na
                with np.errstate(invalid='ignore', divide='ignore'):
                    acep_b = na / n
                    exa_b = (A & C).sum(1) / na
                    exr_b = (~A & C).sum(1) / nr
                    gan_b = exa_b - C.mean(1)
                    cap_b = (~A & ~C).sum(1) / (~C).sum(1)
                    fila = {'detector': d, 'corrupcion': k[0], 'nivel': k[1], 'origen_umbral': origen, 'umbral': t,
                            'aceptacion': acc.mean(),
                            'exactitud_aceptadas': ok[acc].mean() if acc.any() else np.nan,
                            'exactitud_rechazadas': ok[~acc].mean() if (~acc).any() else np.nan,
                            'exactitud_celda': ok.mean(),
                            'ganancia_selectiva': (ok[acc].mean() if acc.any() else np.nan) - ok.mean(),
                            'captura_errores': (~acc & ~ok).sum() / (~ok).sum() if (~ok).any() else np.nan,
                            'n_aceptadas': int(acc.sum()), 'n_errores': int((~ok).sum())}
                for nom, bb in [('aceptacion', acep_b), ('exactitud_aceptadas', exa_b), ('exactitud_rechazadas', exr_b),
                                ('ganancia_selectiva', gan_b), ('captura_errores', cap_b)]:
                    fila[f'{nom}_IC_inf'], fila[f'{nom}_IC_sup'] = ic(bb)
                filas.append(fila)
    umb = pd.DataFrame(filas)
    umb.to_csv(os.path.join(SAL, 'umbral_fijo_celda.csv'), index=False)

    # 3. Riesgo-cobertura conjunto (limpias + 27 celdas); bootstrap por imagen original (arrastra sus 28 versiones)
    E = np.stack([err[k] for k in CELDAS], 1)                         # (n, 28)
    filas, curvas = [], {}
    rng_idx = idx
    for d in PRINCIPALES + ['oraculo', 'azar']:
        for variante in (['bruta', 'saturadas_empatadas'] if d == SX else ['bruta']):
            if d in ('oraculo', 'azar'):
                S = None
            else:
                S = np.stack([G['s'][variante][d][k] for k in CELDAS], 1)
            punto, cov, rie = aurc(S, E, d, curva=True)
            reps_a = np.array([aurc(None if S is None else S[i], E[i], d) for i in rng_idx])
            lo, hi = ic(reps_a)
            filas.append({'detector': d, 'variante': variante, 'AURC': punto, 'IC_inf': lo, 'IC_sup': hi,
                          'n_puntos': E.size, 'tasa_error_total': E.mean()})
            if variante == 'bruta':
                curvas[d] = (cov, rie)
            print(f'AURC {d} {variante}: {punto:.4f} [{lo:.4f}; {hi:.4f}]', flush=True)
    au = pd.DataFrame(filas)
    au.to_csv(os.path.join(SAL, 'aurc.csv'), index=False)
    np.savez_compressed(os.path.join(SAL, 'curvas_riesgo_cobertura.npz'),
                        **{f'{d}_cobertura': c[0] for d, c in curvas.items()},
                        **{f'{d}_riesgo': c[1] for d, c in curvas.items()})

    import fase8b_figuras
    fase8b_figuras.todas(SAL, auc, umb, curvas)


def aurc(S, E, d, curva=False):
    """AURC = media del riesgo (tasa de error entre las aceptadas) en las coberturas k/N, k = 1..N, ordenando por la
    puntuación de mayor a menor. Empates exactos: riesgo esperado con desempate al azar (errores repartidos
    uniformemente dentro de cada grupo empatado). Oráculo: errores al final. Azar: riesgo constante = tasa de error."""
    e = E.ravel().astype(np.float64)
    N = len(e)
    k = np.arange(1, N + 1)
    if d == 'azar':
        rie = np.full(N, e.mean())
    elif d == 'oraculo':
        n_ok = N - e.sum()
        rie = np.maximum(0, k - n_ok) / k
    else:
        s = S.ravel()
        o = np.argsort(-s, kind='stable')
        s, e = s[o], e[o]
        fin = np.r_[np.nonzero(np.diff(s))[0], N - 1]          # último índice de cada grupo empatado
        ini = np.r_[0, fin[:-1] + 1]
        ce = np.cumsum(e)
        err_fin = ce[fin]
        err_ini = np.r_[0, err_fin[:-1]]
        grupo = np.repeat(np.arange(len(fin)), fin - ini + 1)
        tam = (fin - ini + 1)[grupo]
        pos = k - ini[grupo]                                      # posición dentro del grupo, 1..tam
        esperado = err_ini[grupo] + pos * (err_fin - err_ini)[grupo] / tam
        rie = esperado / k
    a = rie.mean()
    if curva:
        paso = max(1, N // 2000)
        return a, (k / N)[::paso], rie[::paso]
    return a


if __name__ == '__main__':
    main()
