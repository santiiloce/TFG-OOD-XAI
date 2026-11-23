# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 1 — Métricas de clasificación de los 5 modelos sobre el mismo test ID (12.618 imágenes), a partir de
results_fase1/predicciones_<modelo>.npz (fase1_predicciones.py). Solo CPU.

    exactitud, F1 macro, balanced accuracy y recall por clase
    IC 95 % por bootstrap (1.000 remuestreos, semilla 0) de exactitud y F1 macro; bootstrap pareado de la
        diferencia Swin - ConvNeXt (mismos índices para ambos)
    test de McNemar Swin vs ConvNeXt (exacto binomial y chi-cuadrado con corrección de continuidad)
    sensibilidad: bootstrap por lámina (remuestreo de las 95 láminas, prefijo del nombre de fichero), porque los
        parches de test comparten láminas y no son independientes

    python fase1_metricas.py   -> results_fase1/
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
SALIDA = os.path.join(RAIZ, 'results_fase1')
MODELOS = {'swin_small_patch4_window7_224': 'Swin-S', 'convnext_small': 'ConvNeXt-S',
           'tf_efficientnetv2_s': 'EfficientNetV2-S', 'vit_base_patch16_224': 'ViT-B/16', 'resnet50': 'ResNet-50'}
A, B = 'swin_small_patch4_window7_224', 'convnext_small'
N_BOOT, SEMILLA = 1000, 0


def metricas(y, p):
    return {'exactitud': float(np.mean(y == p)), 'f1_macro': float(f1_score(y, p, average='macro')),
            'balanced_accuracy': float(balanced_accuracy_score(y, p))}


def lamina(rutas):
    return np.array([os.path.basename(r).split('_')[0] for r in rutas])


def indices_bootstrap(n, rng, grupos=None):
    if grupos is None:
        return rng.integers(0, n, n)
    unicos = np.unique(grupos)
    por_grupo = {g: np.where(grupos == g)[0] for g in unicos}
    elegidos = rng.choice(unicos, len(unicos), replace=True)
    return np.concatenate([por_grupo[g] for g in elegidos])


def bootstrap(y, preds, grupos=None):
    """Réplicas de exactitud y F1 macro para todos los modelos con los mismos índices (permite diferencias pareadas)."""
    rng = np.random.default_rng(SEMILLA)
    reps = {m: {'exactitud': [], 'f1_macro': []} for m in preds}
    for _ in range(N_BOOT):
        i = indices_bootstrap(len(y), rng, grupos)
        for m, p in preds.items():
            reps[m]['exactitud'].append(np.mean(y[i] == p[i]))
            reps[m]['f1_macro'].append(f1_score(y[i], p[i], average='macro', labels=np.arange(14), zero_division=0))
    return {m: {k: np.array(v) for k, v in r.items()} for m, r in reps.items()}


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    tabla = {'ambos_aciertan': int(np.sum(ca & cb)), 'solo_swin_acierta': int(np.sum(ca & ~cb)),
             'solo_convnext_acierta': int(np.sum(~ca & cb)), 'ambos_fallan': int(np.sum(~ca & ~cb))}
    b, c = tabla['solo_swin_acierta'], tabla['solo_convnext_acierta']
    p_exacto = stats.binomtest(b, b + c, 0.5, alternative='two-sided').pvalue
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return {**tabla, 'b_mas_c': b + c, 'p_exacto_binomial': float(p_exacto), 'chi2_continuidad': float(chi2),
            'p_chi2': float(stats.chi2.sf(chi2, 1))}


def num(x, d=3):
    return f'{x:.{d}f}'.replace('.', '{,}')


def main():
    info = json.load(open(os.path.join(RAIZ, 'features/completo/info.json')))
    clases = info['clases_id']
    datos = {m: np.load(os.path.join(SALIDA, f'predicciones_{m}.npz')) for m in MODELOS}
    y = datos[A]['label']
    rutas = datos[A]['paths']
    for m, d in datos.items():
        assert np.array_equal(d['label'], y) and np.array_equal(d['paths'], rutas), f'{m}: orden distinto'
    preds = {m: d['pred'] for m, d in datos.items()}

    # Comprobación: el Swin coincide con las predicciones de la fase A (features/completo)
    fase_a = np.load(os.path.join(RAIZ, 'features/completo/id_test.npz'))
    coincidencia_swin = float(np.mean(fase_a['pred'] == preds[A]))
    registradas = {m: v['exactitud'] for m, v in json.load(open(os.path.join(SALIDA, 'predicciones_info.json')))['modelos'].items()}

    filas, recall = [], []
    reps = bootstrap(y, preds)
    reps_lam = bootstrap(y, preds, lamina(rutas))
    for m, p in preds.items():
        met = metricas(y, p)
        fila = {'modelo': m, 'nombre': MODELOS[m], **met}
        for k in ['exactitud', 'f1_macro']:
            fila[f'{k}_IC_inf'], fila[f'{k}_IC_sup'] = np.percentile(reps[m][k], [2.5, 97.5])
            fila[f'{k}_IC_lamina_inf'], fila[f'{k}_IC_lamina_sup'] = np.percentile(reps_lam[m][k], [2.5, 97.5])
        filas.append(fila)
        r = recall_score(y, p, average=None, labels=np.arange(len(clases)), zero_division=0)
        for c, nombre in enumerate(clases):
            recall.append({'modelo': m, 'clase': nombre, 'n': int(np.sum(y == c)), 'recall': float(r[c])})
    df = pd.DataFrame(filas)
    rec = pd.DataFrame(recall)

    # Diferencia pareada Swin - ConvNeXt
    pareado = {}
    for k in ['exactitud', 'f1_macro']:
        obs = metricas(y, preds[A])[k] - metricas(y, preds[B])[k]
        for etiqueta, rr in [('imagen', reps), ('lamina', reps_lam)]:
            d = rr[A][k] - rr[B][k]
            pareado[f'{k}_{etiqueta}'] = {'diferencia': obs, 'IC_inf': float(np.percentile(d, 2.5)),
                                          'IC_sup': float(np.percentile(d, 97.5)),
                                          'p_bootstrap': float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0))))}
    mc = mcnemar(y, preds[A], preds[B])

    df.to_csv(os.path.join(SALIDA, 'metricas_modelos.csv'), index=False)
    rec.to_csv(os.path.join(SALIDA, 'recall_por_clase.csv'), index=False)
    with open(os.path.join(SALIDA, 'swin_vs_convnext.json'), 'w') as f:
        json.dump({'mcnemar': mc, 'bootstrap_pareado': pareado, 'n_boot': N_BOOT, 'semilla': SEMILLA,
                   'coincidencia_swin_con_fase_A': coincidencia_swin, 'exactitud_registrada_en_inferencia': registradas},
                  f, indent=2, ensure_ascii=False)

    # Tablas LaTeX
    orden = list(MODELOS)
    l = ['% Generado por fase1_metricas.py', '\\begin{table}[htbp]', '\\centering', '\\scriptsize',
         '\\setlength{\\tabcolsep}{3pt}',
         '\\caption{Clasificación en ID test (12.618 imágenes, 14 clases). IC del 95\\,\\% por bootstrap sobre '
         'imágenes (1.000 remuestreos, semilla 0) y, entre llaves, por lámina (remuestreo de las 95 láminas; los '
         'parches de test comparten láminas con train). Cada modelo con el preprocesado de validación de su learner '
         '(ConvNeXt-S a 256\\,px, el resto a 224\\,px).}', '\\label{tab:fase1}',
         '\\begin{tabular}{lccc}', '\\toprule',
         'Modelo & Exactitud [IC 95\\,\\%] & F1 macro [IC 95\\,\\%] & Balanced acc. \\\\', '\\midrule']
    for m in orden:
        r = df[df.modelo == m].iloc[0]
        l.append(f'{MODELOS[m]} & {num(r.exactitud, 4)} [{num(r.exactitud_IC_inf, 4)}; {num(r.exactitud_IC_sup, 4)}] '
                 f'\\{{{num(r.exactitud_IC_lamina_inf, 4)}; {num(r.exactitud_IC_lamina_sup, 4)}\\}} & '
                 f'{num(r.f1_macro, 4)} [{num(r.f1_macro_IC_inf, 4)}; {num(r.f1_macro_IC_sup, 4)}] '
                 f'\\{{{num(r.f1_macro_IC_lamina_inf, 4)}; {num(r.f1_macro_IC_lamina_sup, 4)}\\}} & '
                 f'{num(r.balanced_accuracy, 4)} \\\\')
    l += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(SALIDA, 'tabla_fase1.tex'), 'w') as f:
        f.write('\n'.join(l) + '\n')

    piv = rec.pivot(index='clase', columns='modelo', values='recall').reindex(index=clases, columns=orden)
    n = rec[rec.modelo == A].set_index('clase').n
    l = ['% Generado por fase1_metricas.py', '\\begin{table}[htbp]', '\\centering', '\\footnotesize',
         '\\setlength{\\tabcolsep}{4pt}', '\\caption{Recall por clase en ID test.}', '\\label{tab:fase1_recall}',
         '\\begin{tabular}{lr' + 'c' * len(orden) + '}', '\\toprule',
         'Clase & $n$ & ' + ' & '.join(MODELOS[m] for m in orden) + ' \\\\', '\\midrule']
    for c in clases:
        mejor = piv.loc[c].max()
        celdas = [(f'\\textbf{{{num(v)}}}' if np.isclose(v, mejor) else num(v)) for v in piv.loc[c]]
        l.append(f'{c.replace("_", " ")} & {int(n[c])} & ' + ' & '.join(celdas) + ' \\\\')
    l += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    with open(os.path.join(SALIDA, 'tabla_recall_por_clase.tex'), 'w') as f:
        f.write('\n'.join(l) + '\n')

    pd.set_option('display.width', 250)
    print('Coincidencia Swin con fase A:', coincidencia_swin)
    print(df.to_string(index=False, float_format='%.4f'))
    print(piv.round(3).to_string())
    print(json.dumps({'mcnemar': mc, 'pareado': pareado}, indent=1))


if __name__ == '__main__':
    main()
