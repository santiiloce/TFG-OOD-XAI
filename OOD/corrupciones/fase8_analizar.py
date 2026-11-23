# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 8 — Análisis (solo CPU, sin red): comprobaciones de seguridad, métricas por celda, CEC, tablas y figuras.
Lee lo que deja fase8_extraer.py en <salida>/ (muestra.npz, celdas/*.npz) y las referencias congeladas.

    python fase8_analizar.py --modelo swin --salida results_corrupciones
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import corrupciones_zhang as cz   # noqa: E402

REFERENCIAS = {'swin': {'features': os.path.join(RAIZ, 'features/completo'),
                        'faseA': os.path.join(RAIZ, 'results_ood/faseA_2026-10-20_11-19-35'),
                        'faseB': os.path.join(RAIZ, 'results_ood/faseB_paper_2026-10-20')}}
NOMBRES_DET = {'msp': 'MSP', 'mls': 'MaxLogit', 'ebo': 'Energy', 'react': 'ReAct', 'ash': 'ASH', 'dice': 'DICE',
               'knn': 'KNN', 'mds': 'MDS', 'rmds': 'RMDS', 'vim': 'ViM', 'she': 'SHE (prod. escalar)',
               'she_cosine': 'SHE (coseno)', 'she_euclidean': 'SHE (euclídea)', 'stoodx_100': 'STOOD-X 100 %',
               'stoodx_875': 'STOOD-X 87,5 %'}
DETECTORES = list(NOMBRES_DET)
PRINCIPALES = ['msp', 'mds', 'knn', 'vim', 'rmds', 'stoodx_100']
NIVELES_CEC = [0, 1, 3, 5]
PARES = [(a, b) for i, a in enumerate(range(4)) for b in range(i + 1, 4)]   # índices de columna, a < b
B = 1000
SEMILLA = 0
NUM_CLASSES = 14
N_PROC = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def cargar(salida, bloques):
    """Une los bloques (muestra_<b>.npz, celdas/<b>/*.npz) y ordena por índice del test. Compatible con la
    disposición sin bloques de la prueba de 500 (muestra.npz, celdas/*.npz)."""
    nombres = [('limpia', 0)] + [(c, n) for c in cz.CORRUPCIONES for n in cz.NIVELES]
    if os.path.exists(os.path.join(salida, 'muestra.npz')):
        partes = [(os.path.join(salida, 'muestra.npz'), os.path.join(salida, 'celdas'))]
    else:
        partes = [(os.path.join(salida, f'muestra_{b}.npz'), os.path.join(salida, 'celdas', b)) for b in bloques]
    ms, cs, rs = [], [], []
    for fm, dc in partes:
        ms.append(dict(np.load(fm)))
        cs.append({k: dict(np.load(os.path.join(dc, f'{k[0]}_{k[1]}.npz'))) for k in nombres})
        rs.append(dict(np.load(os.path.join(dc, 'congeladas_reajuste_cpu.npz'))))
    une = lambda ds: {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
    m = une(ms)
    orden = np.argsort(m['idx'], kind='stable')
    m = {k: v[orden] for k, v in m.items()}
    celdas = {k: {kk: vv[orden] for kk, vv in une([c[k] for c in cs]).items()} for k in nombres}
    reajuste = {k: v[orden] for k, v in une(rs).items()}
    return m, celdas, reajuste


def score(celdas, c, n, det):
    return celdas[('limpia', 0) if n == 0 else (c, n)][det]


# ---------------------------------------------------------------------------
# Paso 3: comprobaciones
# ---------------------------------------------------------------------------

def comprobar_nivel0(m, celdas, reajuste, ref, salida):
    idx = m['idx']
    test = dict(np.load(os.path.join(ref['features'], 'id_test.npz')))
    assert np.array_equal(test['paths'][idx], m['paths'])
    limpia = celdas[('limpia', 0)]
    df_f = np.abs(limpia['feats'].astype(np.float64) - test['feats'][idx])
    df_l = np.abs(limpia['logits'].astype(np.float64) - test['logits'][idx])
    res = {'n': int(len(idx)),
           'feats_max_abs': float(df_f.max()), 'feats_mediana_abs': float(np.median(df_f)),
           'logits_max_abs': float(df_l.max()), 'logits_mediana_abs': float(np.median(df_l)),
           'pred_identicas_pct': float(100 * np.mean(limpia['pred'] == test['pred'][idx])),
           'n_pred_distintas': int(np.sum(limpia['pred'] != test['pred'][idx])),
           'exactitud_cpu': float(np.mean(limpia['pred'] == m['label'])),
           'exactitud_congelada_misma_muestra': float(np.mean(test['pred'][idx] == m['label'])),
           'n_pred_distintas_congelada_vs_cpu_aciertos': int(np.sum((limpia['pred'] == m['label']) != (test['pred'][idx] == m['label'])))}
    filas = []
    for det in DETECTORES:
        if det.startswith('stoodx'):
            congelada = np.load(os.path.join(ref['faseB'], f'scores_{det}.npz'))['id_test_conf'][idx]
        else:
            congelada = np.load(os.path.join(ref['faseA'], f'scores_{det}.npz'))['id_test_conf'][idx]
        filas.append({'detector': det,
                      'spearman_cpu_vs_congelada': stats.spearmanr(limpia[det], congelada)[0],
                      'max_abs_cpu_vs_congelada': float(np.max(np.abs(limpia[det] - congelada))),
                      # solo el reajuste del detector en CPU (mismas features congeladas)
                      'spearman_reajuste_vs_congelada': stats.spearmanr(reajuste[det], congelada)[0],
                      'max_abs_reajuste_vs_congelada': float(np.max(np.abs(reajuste[det] - congelada)))})
    df = pd.DataFrame(filas)
    res['spearman_min'] = float(df['spearman_cpu_vs_congelada'].min())
    res['parar'] = bool(res['pred_identicas_pct'] < 99.9 or res['spearman_min'] < 0.999)
    df.to_csv(os.path.join(salida, 'comprobacion_nivel0_puntuaciones.csv'), index=False)
    json.dump(res, open(os.path.join(salida, 'comprobacion_nivel0.json'), 'w'), indent=1)
    return res, df


def comprobar_reproducibilidad(m, celdas, salida, n=20):
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for _ in range(n):
        i = int(rng.integers(len(m['paths'])))
        c = cz.CORRUPCIONES[int(rng.integers(len(cz.CORRUPCIONES)))]
        nivel = cz.NIVELES[int(rng.integers(3))]
        import hashlib
        a = cz.corromper(str(m['paths'][i]), c, nivel)
        h = hashlib.md5(a.tobytes()).hexdigest()
        filas.append({'imagen': os.path.basename(str(m['paths'][i])), 'corrupcion': c, 'nivel': nivel,
                      'md5_guardado': str(celdas[(c, nivel)]['md5'][i]), 'md5_regenerado': h,
                      'identica': h == str(celdas[(c, nivel)]['md5'][i])})
    # motion_blur siempre, por ser la única aleatoria
    for nivel in cz.NIVELES:
        i = int(rng.integers(len(m['paths'])))
        import hashlib
        a = cz.corromper(str(m['paths'][i]), 'motion_blur', nivel)
        h = hashlib.md5(a.tobytes()).hexdigest()
        filas.append({'imagen': os.path.basename(str(m['paths'][i])), 'corrupcion': 'motion_blur', 'nivel': nivel,
                      'md5_guardado': str(celdas[('motion_blur', nivel)]['md5'][i]), 'md5_regenerado': h,
                      'identica': h == str(celdas[('motion_blur', nivel)]['md5'][i]), 'extra_motion': True})
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, 'comprobacion_reproducibilidad.csv'), index=False)
    return df


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def auroc_fpr(conf_id, conf_ood):
    """AUROC y FPR@95 con la convención de metricas_ood/OpenOOD: positiva OOD, puntuación -conf; FPR@95 = FPR en
    el primer umbral con TPR >= 0,95."""
    y = np.r_[np.zeros(len(conf_id)), np.ones(len(conf_ood))]
    fpr, tpr, _ = metrics.roc_curve(y, -np.r_[conf_id, conf_ood])
    return metrics.auc(fpr, tpr), fpr[np.argmax(tpr >= 0.95)]


def _bootstrap_det(args):
    det, c, n, s0, s1, idx = args
    a, f = auroc_fpr(s0, s1)
    reps = np.array([auroc_fpr(s0[i], s1[i]) for i in idx])
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return {'detector': det, 'corrupcion': c, 'nivel': n, 'AUROC': a, 'AUROC_ic_inf': lo[0], 'AUROC_ic_sup': hi[0],
            'FPR95': f, 'FPR95_ic_inf': lo[1], 'FPR95_ic_sup': hi[1], 'n': len(s0)}


def exactitud_balanceada(pred, label):
    rec = [np.mean(pred[label == k] == k) for k in np.unique(label)]
    return float(np.mean(rec))


def metricas_exactitud(m, celdas, idx_boot):
    label = m['label']
    ok0 = celdas[('limpia', 0)]['pred'] == label
    filas = []
    for (c, n), d in celdas.items():
        ok = d['pred'] == label
        acc_b = ok[idx_boot].mean(1)
        caida_b = ok0[idx_boot].mean(1) - acc_b
        bal_b = np.array([exactitud_balanceada(d['pred'][i], label[i]) for i in idx_boot])
        filas.append({'corrupcion': c, 'nivel': n, 'n': len(ok),
                      'exactitud': ok.mean(), 'exactitud_ic_inf': np.percentile(acc_b, 2.5),
                      'exactitud_ic_sup': np.percentile(acc_b, 97.5),
                      'exactitud_balanceada': exactitud_balanceada(d['pred'], label),
                      'exactitud_balanceada_ic_inf': np.percentile(bal_b, 2.5),
                      'exactitud_balanceada_ic_sup': np.percentile(bal_b, 97.5),
                      'caida_exactitud': ok0.mean() - ok.mean(), 'caida_ic_inf': np.percentile(caida_b, 2.5),
                      'caida_ic_sup': np.percentile(caida_b, 97.5)})
    return pd.DataFrame(filas)


def saturacion(celdas):
    filas = []
    for (c, n), d in celdas.items():
        for det in ['stoodx_100', 'stoodx_875']:
            sat = d[f'{det}_U'] == d[f'{det}_n1n2']
            k500 = d[f'{det}_n1n2'] == 500 * 500 * 500
            filas.append({'detector': det, 'corrupcion': c, 'nivel': n, 'n': len(sat),
                          'pct_saturadas': 100 * sat.mean(),
                          'pct_saturadas_clases_K500': 100 * np.mean(sat & k500),
                          'puntuacion_saturada_min_K500': float(d[det][sat & k500].min()) if np.any(sat & k500) else np.nan,
                          'puntuacion_saturada_max_K500': float(d[det][sat & k500].max()) if np.any(sat & k500) else np.nan})
    return pd.DataFrame(filas)


def secuencias(celdas, det, c):
    return np.stack([score(celdas, c, n, det) for n in NIVELES_CEC], axis=1)


def cec_imagen(S):
    """CEC por imagen (n,) y número de empates exactos (n,). S: (n, 4), "mayor = más ID"."""
    disc = np.zeros(len(S))
    emp = np.zeros(len(S))
    for a, b in PARES:
        disc += (S[:, b] > S[:, a]) + 0.5 * (S[:, b] == S[:, a])
        emp += S[:, b] == S[:, a]
    return disc / len(PARES), emp


def casi_empates(S, rel=1e-9):
    cnt = 0
    for a, b in PARES:
        d = np.abs(S[:, b] - S[:, a])
        esc = np.maximum(1.0, np.maximum(np.abs(S[:, a]), np.abs(S[:, b])))
        cnt += np.sum((d > 0) & (d <= rel * esc))
    return int(cnt)


def tabla_cec(celdas, idx_boot):
    filas, comprobacion = [], []
    rng = np.random.default_rng(SEMILLA)
    for det in DETECTORES:
        por_corr, boot_corr, emp_tot, pares_tot = {}, [], 0, 0
        for c in cz.CORRUPCIONES:
            S = secuencias(celdas, det, c)
            cec_x, emp = cec_imagen(S)
            boot = cec_x[idx_boot].mean(1)
            boot_corr.append(boot)
            fila = {'detector': det, 'corrupcion': c, 'CEC': cec_x.mean(),
                    'CEC_ic_inf': np.percentile(boot, 2.5), 'CEC_ic_sup': np.percentile(boot, 97.5),
                    'pct_pares_empatados': 100 * emp.sum() / (len(S) * len(PARES)),
                    'pct_imagenes_con_empate': 100 * np.mean(emp > 0),
                    'n_casi_empates_rel1e-9': casi_empates(S), 'n': len(S)}
            if det.startswith('stoodx'):
                # sensibilidad: pares con ambos niveles saturados (U = n1*n2) tratados como empates
                sat = np.stack([(celdas[('limpia', 0) if n == 0 else (c, n)][f'{det}_U'] ==
                                 celdas[('limpia', 0) if n == 0 else (c, n)][f'{det}_n1n2']) for n in NIVELES_CEC], 1)
                disc = np.zeros(len(S))
                ambos = 0
                for a, b in PARES:
                    t = sat[:, a] & sat[:, b]
                    ambos += t.sum()
                    disc += np.where(t, 0.5, (S[:, b] > S[:, a]) + 0.5 * (S[:, b] == S[:, a]))
                fila['pct_pares_ambos_saturados'] = 100 * ambos / (len(S) * len(PARES))
                fila['CEC_saturadas_como_empate'] = (disc / len(PARES)).mean()
            filas.append(fila)
            emp_tot += emp.sum()
            pares_tot += len(S) * len(PARES)
            # comprobación CEC = (1 - tau_a) / 2 en imágenes sin empates (hasta 300 por celda, con scipy)
            sin = np.where(emp == 0)[0]
            if len(sin):
                sel = rng.choice(sin, min(300, len(sin)), replace=False)
                ref = np.array([0, -1, -3, -5])
                tau = np.array([stats.kendalltau(S[i], ref)[0] for i in sel])
                comprobacion.append({'detector': det, 'corrupcion': c, 'n_sin_empates': len(sin),
                                     'n_comprobadas': len(sel),
                                     'max_abs_dif': float(np.max(np.abs(cec_x[sel] - (1 - tau) / 2)))})
        boot_g = np.mean(boot_corr, axis=0)
        filas.append({'detector': det, 'corrupcion': 'global',
                      'CEC': np.mean([f['CEC'] for f in filas if f['detector'] == det and f['corrupcion'] != 'global']),
                      'CEC_ic_inf': np.percentile(boot_g, 2.5), 'CEC_ic_sup': np.percentile(boot_g, 97.5),
                      'pct_pares_empatados': 100 * emp_tot / pares_tot, 'n': len(idx_boot[0])})
        if det.startswith('stoodx'):
            sub = [f for f in filas if f['detector'] == det and f['corrupcion'] != 'global']
            filas[-1]['pct_pares_ambos_saturados'] = np.mean([f['pct_pares_ambos_saturados'] for f in sub])
            filas[-1]['CEC_saturadas_como_empate'] = np.mean([f['CEC_saturadas_como_empate'] for f in sub])
    return pd.DataFrame(filas), pd.DataFrame(comprobacion)


def correlaciones(met, acc):
    caida = acc[acc.nivel > 0].set_index(['corrupcion', 'nivel'])['caida_exactitud']
    filas = []
    for det in DETECTORES:
        a = met[met.detector == det].set_index(['corrupcion', 'nivel'])['AUROC'].reindex(caida.index)
        r, p = stats.spearmanr(caida.values, a.values)
        filas.append({'detector': det, 'rho_spearman': r, 'p_valor': p, 'n_celdas': len(a)})
    return pd.DataFrame(filas)


def puntuaciones_largas(m, celdas, salida):
    imagen = np.array([os.path.basename(str(p)) for p in m['paths']])
    trozos = []
    for (c, n), d in celdas.items():
        for det in DETECTORES:
            trozos.append(pd.DataFrame({'imagen': imagen, 'clase': m['label'], 'corrupcion': c, 'nivel': n,
                                        'detector': det, 'puntuacion': d[det]}))
    df = pd.concat(trozos, ignore_index=True)
    for col in ['corrupcion', 'detector', 'imagen']:
        df[col] = df[col].astype('category')
    df.to_parquet(os.path.join(salida, 'puntuaciones.parquet'), index=False)
    return len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', default='swin', choices=list(REFERENCIAS))
    p.add_argument('--salida', required=True)
    p.add_argument('--sin_figuras', action='store_true')
    p.add_argument('--bloques', nargs='+', default=['sub', 'resto'])
    p.add_argument('--destino', default=None, help='carpeta de las tablas y figuras (por defecto, --salida)')
    args = p.parse_args()
    salida = args.salida if os.path.isabs(args.salida) else os.path.join(RAIZ, args.salida)
    ref = REFERENCIAS[args.modelo]
    m, celdas, reajuste = cargar(salida, args.bloques)
    salida = os.path.join(salida, args.destino) if args.destino else salida
    os.makedirs(salida, exist_ok=True)
    n = len(m['idx'])
    idx_boot = np.random.default_rng(SEMILLA).integers(0, n, (B, n))

    res0, df0 = comprobar_nivel0(m, celdas, reajuste, ref, salida)
    print(json.dumps(res0, indent=1))
    print(df0.to_string(float_format='%.6f'))
    rep = comprobar_reproducibilidad(m, celdas, salida)
    print(f'reproducibilidad: {rep.identica.sum()}/{len(rep)} idénticas bit a bit')
    if res0['parar']:
        print('PARADA: el nivel 0 no cumple los criterios (pred >= 99,9 % y Spearman >= 0,999).')
        return

    acc = metricas_exactitud(m, celdas, idx_boot)
    acc.to_csv(os.path.join(salida, 'exactitud_celda.csv'), index=False)
    tareas = [(det, c, nv, score(celdas, c, 0, det), celdas[(c, nv)][det], idx_boot)
              for det in DETECTORES for c in cz.CORRUPCIONES for nv in cz.NIVELES]
    with mp.get_context('fork').Pool(N_PROC) as pool:
        met = pd.DataFrame(pool.map(_bootstrap_det, tareas, chunksize=1))
    met.to_csv(os.path.join(salida, 'metricas_celda.csv'), index=False)
    sat = saturacion(celdas)
    sat.to_csv(os.path.join(salida, 'saturacion_stoodx.csv'), index=False)
    cec, comp = tabla_cec(celdas, idx_boot)
    cec.to_csv(os.path.join(salida, 'cec.csv'), index=False)
    comp.to_csv(os.path.join(salida, 'comprobacion_cec_kendall.csv'), index=False)
    cor = correlaciones(met, acc)
    cor.to_csv(os.path.join(salida, 'correlacion_exactitud_auroc.csv'), index=False)
    filas = puntuaciones_largas(m, celdas, salida)
    print(f'puntuaciones.parquet: {filas} filas; comprobación Kendall max |dif| = {comp.max_abs_dif.max():.2e}')
    if not args.sin_figuras:
        import fase8_figuras
        fase8_figuras.todas(salida, m, celdas)


if __name__ == '__main__':
    main()
