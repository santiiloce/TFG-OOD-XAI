# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase C — Comparación conjunta de la fase A (OpenOOD) y la fase B (STOOD-X).

Solo numpy, pandas, scikit-learn y matplotlib: se ejecuta en cualquiera de los dos entornos y no
necesita GPU.

    python comparar_resultados.py --faseA results_ood/faseA_<fecha> --faseB results_ood/faseB_<fecha>

Comprueba que los conjuntos y el orden de las imágenes coinciden (mismas rutas) y genera en
results_ood/faseC_<fecha>/:
    tabla_principal.csv / .tex     métricas + IC bootstrap 95 % (AUROC, FPR@95) de todos los métodos
    bootstrap_pareado_rmds.csv     diferencia de AUROC RMDS - método en near_nct_tum_norm
    tabla_nct_por_clase.csv
    fig1..fig5 (.pdf y .png)
    resumen_numerico.json          números que respaldan analisis.md
"""

import argparse
import glob
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import metricas_ood

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'

METODOS = {
    'msp': 'MSP', 'mls': 'MaxLogit', 'ebo': 'Energy',
    'react': 'ReAct', 'ash': 'ASH', 'dice': 'DICE',
    'knn': 'KNN', 'mds': 'MDS', 'rmds': 'RMDS', 'vim': 'ViM',
    'she': 'SHE (prod. escalar)', 'she_cosine': 'SHE (coseno)', 'she_euclidean': 'SHE (euclídea)',
    'stoodx_100': 'STOOD-X (100 %)', 'stoodx_875': 'STOOD-X (87,5 %)',
    'stoodx_100_kc': 'STOOD-X K_c (exploratorio)',
    'stoodx_100_k144': 'STOOD-X (100 %, K=144)',
}
FAMILIA = {'msp': 'logits', 'mls': 'logits', 'ebo': 'logits',
           'react': 'activaciones', 'ash': 'activaciones', 'dice': 'activaciones',
           'knn': 'distancia', 'mds': 'distancia', 'rmds': 'distancia', 'vim': 'distancia',
           'she': 'distancia', 'she_cosine': 'distancia', 'she_euclidean': 'distancia',
           'stoodx_100': 'distancia', 'stoodx_875': 'distancia', 'stoodx_100_kc': 'distancia',
           'stoodx_100_k144': 'distancia'}
# Variantes exploratorias: se marcan en las tablas (columna 'exploratorio')
EXPLORATORIOS = {'stoodx_100_kc'}
CONJUNTOS = {
    'near_nct_tum_norm': 'NCT TUM+NORM', 'near_nct_tum': 'NCT TUM', 'near_nct_norm': 'NCT NORM',
    'afin_nct_str': 'NCT STR', 'afin_nct_lym': 'NCT LYM', 'afin_nct_mus': 'NCT MUS',
    'cov_panda': 'PANDA', 'control_res_128': 'Control 128 px', 'far_dtd': 'DTD',
}
PRINCIPAL = 'near_nct_tum_norm'
N_BOOT, SEMILLA = 1000, 0

COLORES = plt.get_cmap('tab20').colors
COLOR = {m: COLORES[i % len(COLORES)] for i, m in enumerate(METODOS)}
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3,
    'savefig.bbox': 'tight', 'figure.dpi': 100,
})


def guardar(fig, salida, nombre):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(salida, f'{nombre}.{ext}'), dpi=300)
    plt.close(fig)


def cargar_scores(faseA, faseB, faseB_k144=None):
    scores = {}
    for ruta in sorted(glob.glob(os.path.join(faseA, 'scores_*.npz'))) + sorted(glob.glob(os.path.join(faseB, 'scores_stoodx_*.npz'))):
        metodo = os.path.basename(ruta)[len('scores_'):-len('.npz')]
        if metodo in METODOS and os.path.exists(ruta):
            scores[metodo] = dict(np.load(ruta))
    if faseB_k144:   # K elegido solo con id_val (stoodx_paper.py --modo seleccion_k)
        scores['stoodx_100_k144'] = dict(np.load(os.path.join(faseB_k144, 'scores_stoodx_100.npz')))
    return {m: scores[m] for m in METODOS if m in scores}


def comprobar_orden(scores):
    """Mismas rutas, en el mismo orden, en todos los métodos (incluido STOOD-X)."""
    ref_metodo = next(iter(scores))
    ref = scores[ref_metodo]
    for conj in ['id_val', 'id_test'] + list(CONJUNTOS):
        for m, s in scores.items():
            assert np.array_equal(s[f'{conj}_paths'], ref[f'{conj}_paths']), f'{m}/{conj}: rutas distintas a {ref_metodo}'
            assert np.array_equal(s[f'{conj}_label'], ref[f'{conj}_label']), f'{m}/{conj}: etiquetas distintas'
    return {c: int(len(ref[f'{c}_paths'])) for c in ['id_val', 'id_test'] + list(CONJUNTOS)}


def tabla_principal(scores):
    filas = []
    for m, s in scores.items():
        cid, pid, lid = s['id_test_conf'], s['id_test_pred'], s['id_test_label']
        acc = metricas_ood.exactitud(pid, lid)
        for conj in CONJUNTOS:
            cood = s[f'{conj}_conf']
            met = metricas_ood.todas_las_metricas(cid, cood, pid, lid)
            ic = metricas_ood.bootstrap_ic(cid, cood, N_BOOT, SEMILLA)
            filas.append({'metodo': m, 'exploratorio': m in EXPLORATORIOS, 'conjunto': conj, 'AUROC': met['AUROC'],
                          'AUROC_IC_inf': ic['AUROC_IC'][0], 'AUROC_IC_sup': ic['AUROC_IC'][1],
                          'FPR@95': met['FPR@95'], 'FPR@95_IC_inf': ic['FPR@95_IC'][0],
                          'FPR@95_IC_sup': ic['FPR@95_IC'][1], 'CCR@FPR5': met['CCR@FPR5'],
                          'AUPR_In': met['AUPR_In'], 'AUPR_Out': met['AUPR_Out'], 'ACC_ID': acc})
        print(f'  bootstrap {m} hecho', flush=True)
    return pd.DataFrame(filas)


def pareado_rmds(scores):
    filas = []
    a = scores['rmds']
    for m, s in scores.items():
        if m == 'rmds':
            continue
        r = metricas_ood.bootstrap_pareado_auroc(a['id_test_conf'], a[f'{PRINCIPAL}_conf'],
                                                 s['id_test_conf'], s[f'{PRINCIPAL}_conf'], N_BOOT, SEMILLA)
        filas.append({'metodo': m, 'exploratorio': m in EXPLORATORIOS, 'dif_AUROC_rmds_menos_metodo': r['diferencia'],
                      'IC_inf': r['IC'][0], 'IC_sup': r['IC'][1], 'p_bootstrap': r['p_bootstrap'],
                      'significativa_5%': not (r['IC'][0] <= 0 <= r['IC'][1])})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_resumen(df, salida):
    metodos = [m for m in METODOS if m in df.metodo.unique()]
    conjuntos = list(CONJUNTOS)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ancho = 0.8 / len(metodos)
    x = np.arange(len(conjuntos))
    for ax, metrica, etiqueta in [(axes[0], 'AUROC', 'AUROC (↑)'), (axes[1], 'FPR@95', 'FPR@95 (↓)')]:
        for i, m in enumerate(metodos):
            d = df[df.metodo == m].set_index('conjunto').reindex(conjuntos)
            err = np.vstack([d[metrica] - d[f'{metrica}_IC_inf'], d[f'{metrica}_IC_sup'] - d[metrica]])
            ax.bar(x + (i - len(metodos) / 2 + 0.5) * ancho, d[metrica], ancho, yerr=err,
                   color=COLOR[m], label=METODOS[m], error_kw={'elinewidth': 0.5, 'capsize': 0})
        ax.set_ylabel(etiqueta)
        ax.set_ylim(0, 1)
        ax.grid(axis='x', visible=False)
    axes[0].axhline(0.5, color='k', lw=0.6, ls=':')
    axes[1].set_xticks(x, [CONJUNTOS[c] for c in conjuntos], rotation=20)
    axes[0].legend(ncol=7, loc='lower center', bbox_to_anchor=(0.5, 1.02), frameon=False)
    guardar(fig, salida, 'fig1_auroc_fpr95')


def fig_roc(scores, salida):
    conjuntos = list(CONJUNTOS)
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), sharex=True, sharey=True)
    for ax, conj in zip(axes.ravel(), conjuntos):
        for m, s in scores.items():
            conf = np.concatenate([s['id_test_conf'], s[f'{conj}_conf']])
            es_ood = np.r_[np.zeros(len(s['id_test_conf'])), np.ones(len(s[f'{conj}_conf']))]
            fpr, tpr, _ = roc_curve(es_ood, -conf)
            ax.plot(fpr, tpr, color=COLOR[m], lw=1, label=METODOS[m])
        ax.plot([0, 1], [0, 1], 'k:', lw=0.6)
        ax.set_title(CONJUNTOS[conj])
    for ax in axes[-1]:
        ax.set_xlabel('FPR (ID marcadas como OOD)')
    for ax in axes[:, 0]:
        ax.set_ylabel('TPR (OOD detectadas)')
    axes[0, 0].legend(ncol=7, loc='lower left', bbox_to_anchor=(0, 1.15), frameon=False)
    guardar(fig, salida, 'fig2_curvas_roc')


def fig_histogramas(scores, salida):
    metodos, conjuntos = list(scores), list(CONJUNTOS)
    fig, axes = plt.subplots(len(metodos), len(conjuntos), figsize=(2.0 * len(conjuntos), 1.4 * len(metodos)))
    for i, m in enumerate(metodos):
        s = scores[m]
        # STOOD-X: se dibuja z (equivalente monótono de p = norm.sf(z)) para ver las colas
        clave = 'z' if m.startswith('stoodx') else 'conf'
        nombre_eje = '-z (STOOD-X)' if clave == 'z' else 'puntuación'
        cid = s[f'id_test_{clave}']
        todos = np.concatenate([cid] + [s[f'{c}_{clave}'] for c in conjuntos])
        transf = (lambda v: -v) if clave == 'z' else (lambda v: v)   # -z: más alto = más ID, como p
        lo, hi = np.percentile(transf(todos), [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        for j, c in enumerate(conjuntos):
            ax = axes[i, j]
            ax.hist(transf(cid), bins, density=True, alpha=0.5, color='tab:blue', label='ID test')
            ax.hist(transf(s[f'{c}_{clave}']), bins, density=True, alpha=0.5, color='tab:red', label='OOD')
            ax.set_yticks([])
            ax.tick_params(labelsize=6)
            ax.grid(False)
            if i == 0:
                ax.set_title(CONJUNTOS[c], fontsize=8)
            if j == 0:
                ax.set_ylabel(METODOS[m], fontsize=7, rotation=0, ha='right', va='center')
            if i == len(metodos) - 1:
                ax.set_xlabel(nombre_eje, fontsize=6)
    axes[0, 0].legend(fontsize=6, loc='upper left', frameon=False)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    guardar(fig, salida, 'fig3_histogramas')


def fig_panda_control(df, salida):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m in METODOS:
        d = df[df.metodo == m].set_index('conjunto')
        if d.empty:
            continue
        ax.errorbar(d.loc['control_res_128', 'AUROC'], d.loc['cov_panda', 'AUROC'],
                    xerr=[[d.loc['control_res_128', 'AUROC'] - d.loc['control_res_128', 'AUROC_IC_inf']],
                          [d.loc['control_res_128', 'AUROC_IC_sup'] - d.loc['control_res_128', 'AUROC']]],
                    yerr=[[d.loc['cov_panda', 'AUROC'] - d.loc['cov_panda', 'AUROC_IC_inf']],
                          [d.loc['cov_panda', 'AUROC_IC_sup'] - d.loc['cov_panda', 'AUROC']]],
                    fmt='o', color=COLOR[m], ms=5, elinewidth=0.6, label=METODOS[m])
    ax.plot([0, 1], [0, 1], 'k:', lw=0.6)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0.3, 1)
    ax.set_xlabel('AUROC control de resolución (ID test a 128 px)')
    ax.set_ylabel('AUROC PANDA (covariate shift)')
    ax.legend(fontsize=7, loc='lower right', frameon=False)
    guardar(fig, salida, 'fig4_panda_vs_control')


def fig_control_interno(df, salida):
    clases = ['near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus']
    metodos = [m for m in METODOS if m in df.metodo.unique()]
    ncol = 5
    nfil = int(np.ceil(len(metodos) / ncol))
    fig, axes = plt.subplots(nfil, ncol, figsize=(13, 2.6 * nfil), sharey=True)
    for ax, m in zip(axes.ravel(), metodos):
        d = df[df.metodo == m].set_index('conjunto')
        vals = d.loc[clases, 'AUROC'].values
        err = np.vstack([vals - d.loc[clases, 'AUROC_IC_inf'].values, d.loc[clases, 'AUROC_IC_sup'].values - vals])
        colores = ['tab:red', 'tab:red', 'tab:gray', 'tab:gray', 'tab:gray']
        ax.bar(range(5), vals, color=colores, yerr=err, error_kw={'elinewidth': 0.6})
        ax.axhline(d.loc['cov_panda', 'AUROC'], color='tab:green', ls='--', lw=1, label='PANDA')
        ax.set_xticks(range(5), ['TUM', 'NORM', 'STR', 'LYM', 'MUS'], fontsize=7)
        ax.set_title(METODOS[m])
        ax.set_ylim(0, 1.02)
        ax.grid(axis='x', visible=False)
    for ax in axes.ravel()[len(metodos):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('AUROC')
    axes.ravel()[0].legend(fontsize=7, loc='lower left', frameon=False)
    fig.suptitle('Control interno NCT-CRC: tejido OOD (rojo) frente a tejido análogo al ID (gris); '
                 'línea verde = PANDA', fontsize=9)
    fig.tight_layout()
    guardar(fig, salida, 'fig5_control_interno_nct')


def tabla_latex(df, salida):
    piv = df.pivot(index='metodo', columns='conjunto', values='AUROC').reindex(
        index=[m for m in METODOS if m in df.metodo.unique()], columns=list(CONJUNTOS))
    piv.index = [METODOS[m] for m in piv.index]
    piv.columns = [CONJUNTOS[c] for c in piv.columns]
    with open(os.path.join(salida, 'tabla_auroc.tex'), 'w') as f:
        f.write((100 * piv).to_latex(float_format='%.1f', escape=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--faseA', required=True)
    parser.add_argument('--faseB', required=True)
    parser.add_argument('--salida', default=None)
    parser.add_argument('--faseB_k144', default=None, help='carpeta de STOOD-X 100 %% con K=144 (opcional)')
    parser.add_argument('--sensibilidad_nnk', default=None, help='carpeta de STOOD-X con NN_K=50 (opcional)')
    args = parser.parse_args()
    abs_ = lambda p: p if p is None or os.path.isabs(p) else os.path.join(RAIZ, p)   # noqa: E731
    faseA, faseB = abs_(args.faseA), abs_(args.faseB)
    salida = abs_(args.salida) or os.path.join(RAIZ, 'results_ood', f'faseC_{datetime.now():%Y-%m-%d_%H-%M-%S}')
    os.makedirs(salida, exist_ok=True)

    scores = cargar_scores(faseA, faseB, abs_(args.faseB_k144))
    print(f'Métodos: {list(scores)}')
    tamanos = comprobar_orden(scores)
    print(f'Orden de imágenes verificado en todos los métodos y conjuntos: {tamanos}')

    df = tabla_principal(scores)
    df.to_csv(os.path.join(salida, 'tabla_principal.csv'), index=False)
    par = pareado_rmds(scores)
    par.to_csv(os.path.join(salida, 'bootstrap_pareado_rmds.csv'), index=False)
    df[df.conjunto.str.contains('nct')].to_csv(os.path.join(salida, 'tabla_nct_por_clase.csv'), index=False)
    tabla_latex(df, salida)

    fig_resumen(df, salida)
    fig_roc(scores, salida)
    fig_histogramas(scores, salida)
    fig_panda_control(df, salida)
    fig_control_interno(df, salida)

    resumen = {'faseA': faseA, 'faseB': faseB, 'faseB_k144': abs_(args.faseB_k144), 'num_imagenes': tamanos, 'n_bootstrap': N_BOOT,
               'semilla_bootstrap': SEMILLA, 'tabla': df.to_dict(orient='records'),
               'pareado_rmds': par.to_dict(orient='records')}
    if args.sensibilidad_nnk:
        resumen['sensibilidad_nnk'] = pd.read_csv(os.path.join(abs_(args.sensibilidad_nnk),
                                                               'resultados_principales.csv')).to_dict(orient='records')
    with open(os.path.join(salida, 'resumen_numerico.json'), 'w') as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=float)

    pd.set_option('display.width', 250)
    for metrica in ['AUROC', 'FPR@95', 'CCR@FPR5']:
        print(f'\n===== {metrica} =====')
        print(df.pivot(index='metodo', columns='conjunto', values=metrica)
                .reindex(index=list(scores), columns=list(CONJUNTOS)).to_string(float_format='%.4f'))
    print('\n===== Bootstrap pareado AUROC(RMDS) - AUROC(método), near_nct_tum_norm =====')
    print(par.to_string(index=False, float_format='%.4f'))
    print(f'\nSalida en {salida}')


if __name__ == '__main__':
    main()
