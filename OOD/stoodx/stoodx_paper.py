# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase B — STOOD-X implementado según el texto del paper (Sevillano-García et al., 2026, Pattern
Recognition, sección 3.2 y análisis de complejidad de la pág. 7), sin el repositorio oficial.
Trabaja sobre las features exportadas en la fase A (features/<tag>/<conjunto>.npz).

Para una imagen x con clase predicha c (argmax de los logits):
    A = distancias coseno de x a sus K vecinos más cercanos de train de la clase c   (K valores)
    B = para cada uno de esos vecinos x_i, sus distancias a sus K vecinos de train de la clase c,
        excluyéndose a sí mismo (X_set \\ {x_i})                                      (K^2 valores)
    test de Mann-Whitney unilateral: scipy.stats.mannwhitneyu(A, B, alternative='greater')
    (M = K + K^2 distancias). Puntuación = p-valor (alto = ID). Se guardan también U y z.
Es nuestra interpretación del texto del paper, pendiente de confirmar con los autores (el docstring
del repositorio describe un test por vecino).

Eficiencia: las distancias de cada imagen de train a sus K vecinos de su clase solo dependen del
train, así que se precalculan una vez por clase (GPU, por bloques) y B se obtiene indexando esa tabla
con los K vecinos de x.

Configuraciones de features (método separado en todas las salidas):
    stoodx_100  -> 100 % de las features
    stoodx_875  -> 87,5 % más presente por clase (criterio de presencia del código oficial,
                   stoodx.py:221-228: suma de |f| en el train de la clase; quantile = 0.125)

Clases con n_c <= K imágenes de train: se usan todas las disponibles (A con min(K, n_c) valores y
B con min(K, n_c - 1) por vecino).

Entorno: S-STOOD-X (.venv, uv). GPU + 8 CPU: sbatch script_stoodx_paper.sh <argumentos>
    --modo validacion                                        (subconjunto, barrido de K, 100 vs 87,5, ejemplos)
    --modo completo --config 100 --conjuntos id_test cov_panda --salida results_ood/faseB_paper_X
    --modo combinar --salida results_ood/faseB_paper_X      (sin GPU)
"""

import argparse
import importlib.metadata as md
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats

import metricas_ood

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
FEATURES_ROOT = os.path.join(RAIZ, 'features')
NUM_CLASSES = 14
N_FEATURES = 768
SEMILLA = 0
N_PROCESOS = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
LOTE = 128                    # imágenes de consulta por lote en GPU
BLOQUE = 4096                 # filas por bloque en el precálculo

# quantile: criterio de presencia del código oficial (conservar las features con suma |f| >= cuantil).
# regla_k: 'fija' -> K para todas las clases (con n_c <= K se usan todas las imágenes), como el paper;
#          'mitad' -> K_c = min(K, floor(n_c / 2)) por clase predicha (variante EXPLORATORIA, propuesta tras
#                     ver la validación con datos OOD; no es el método principal).
CONFIGS = {'100': {'quantile': 0.0, 'regla_k': 'fija'},
           '875': {'quantile': 0.125, 'regla_k': 'fija'},
           '100_kc': {'quantile': 0.0, 'regla_k': 'mitad'}}

CONJUNTOS_PRINCIPALES = ['near_nct_tum_norm', 'near_nct_tum', 'near_nct_norm',
                         'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
                         'cov_panda', 'control_res_128', 'far_dtd']
CONTROL_NAME = 'control_res_128'
CONJUNTOS_NCT_POR_CLASE = ['near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus']
UNIONES = {'near_nct_tum_norm': ['near_nct_tum', 'near_nct_norm']}
A_PUNTUAR = ['id_val', 'id_test'] + [c for c in CONJUNTOS_PRINCIPALES if c not in UNIONES]

# Validación
KS = [9, 18, 36, 72, 144, 288, 500]
K_CORTE = 72                  # comprobación: tabla de K=500 cortada a K columnas == cálculo directo
VAL_OOD = ['near_nct_tum_norm', 'afin_nct_lym', 'cov_panda', 'far_dtd']
N_VAL_ID, N_VAL_OOD = 1000, 500

NOTAS = {
    'interpretacion': 'Interpretación del texto del paper (sección 3.2, paso 3, y análisis de complejidad de la '
                      'pág. 7), pendiente de confirmar con los autores: un único test de Mann-Whitney '
                      'unilateral (scipy.stats.mannwhitneyu(A, B, alternative="greater")) entre A = distancias '
                      'de x a sus K vecinos de train de la clase predicha y B = distancias de cada vecino x_i a '
                      'sus K vecinos de la misma clase excluyéndose a sí mismo (K^2 valores); M = K + K^2. El '
                      'docstring del repositorio oficial describe en cambio un test por vecino.',
    'score': 'Puntuación principal (conf) = -z, con z = (U - n1*n2/2 - 0.5) / sigma (corrección por empates y de '
             'continuidad de scipy; es la z del método asintótico de mannwhitneyu, p = norm.sf(z)). -z es '
             'monótonamente equivalente al p-valor (alto = ID; hipótesis alternativa: x está más lejos de sus '
             'vecinos que los vecinos de los suyos) y se usa para evitar el subdesbordamiento de p con K = 500 '
             '(en la validación, p = 0 exacto en el 89 % de near_nct_tum_norm, el 98 % de far_dtd y el 9 % de '
             'id_test, lo que creaba empates y bajaba el AUROC hasta 0,05). Se guardan también p y U.',
    'k_c': 'stoodx_100_kc (exploratoria): K_c = min(500, floor(n_c / 2)) por clase predicha; A = distancias de x a '
           'sus K_c vecinos de la clase; B = distancias de cada vecino a sus K_c vecinos (sin sí mismo). La regla '
           'se propuso después de ver la validación con datos OOD (con K = 500, las imágenes OOD predichas en '
           'Gleason_5_comedonecrosis, n_c = 399, se aceptaban como ID), así que es una variante exploratoria, no '
           'el método principal.',
    'distancia': 'coseno: 1 - <x/|x|, y/|y|>, calculada como producto de matrices de features normalizadas (L2) '
                 'tras aplicar la máscara de features de la clase.',
    'referencia': 'train ID de la clase predicha por el modelo (argmax de los logits), como en el paper (Fig. 4c).',
    'precalculo': 'Para cada imagen de train, sus K distancias más pequeñas a su clase (excluida ella misma por '
                  'índice), precalculadas en GPU por clase y por bloques, ordenadas; B se obtiene indexando esa '
                  'tabla con los K vecinos de x.',
    'seleccion_features': 'Adaptación: el paper selecciona las features por relevancia CRP (zennit-crp, sección '
                          '4.4), no aplicable de forma fiable a un Swin; se usa el criterio de presencia del código '
                          'oficial (stoodx.py:221-228): suma de |f| sobre el train de la clase predicha, '
                          'quantile = 0.125 -> se conservan 672 de 768 features (87,5 %) y el resto se pone a 0 en x '
                          'y en el conjunto de referencia. La máscara depende de la clase.',
    'clases_pequenas': 'Con regla_k = fija, clases con n_c <= K imágenes de train: se usan todas las disponibles '
                       '(A con min(K, n_c) valores; B con min(K, n_c - 1) por vecino).',
    'saturacion': 'Separación completa: si todas las distancias de A superan a todas las de B, U = n1*n2 y z toma su máximo (38,69 con n1 = 500, n2 = 250.000), que solo depende de n1, n2 y del término de empates de sigma. Entre imágenes saturadas, el orden lo decide ese término (rango de z < 1e-4 con K fijo), no la distancia. En stoodx_100: saturadas el 0,6 % de id_test, 66 % de near_nct_tum_norm, 32 % de afin_nct_lym, 19 % de cov_panda, 19 % de control_res_128 y 76 % de far_dtd. Cotas (todas las OOD saturadas por encima/por debajo de las ID saturadas de su mismo grupo (n1, n2)): AUROC cambia como máximo 0,0031 (near_nct_tum_norm: 0,9534 en [0,9512, 0,9553]); FPR@95 y CCR@FPR5 no cambian en ningún conjunto ni configuración (el umbral de FPR@95 queda por debajo de todo el bloque saturado porque como mucho satura el 79 % de un conjunto OOD, y el de CCR@FPR5 está en el extremo opuesto). Se mantiene -z sin desempate adicional. Detalle: empates_saturacion.csv (analisis_empates_stoodx.py).',
    'calibracion': 'Los p-valores no están calibrados en ID: en id_val (22.691), KS D = 0,80 frente a U(0,1), el 82 % tiene p < 0,05 y la distribución es bimodal (p ~ 0 o p ~ 1) en todas las clases. Solo sirven para ordenar, no como nivel de significación. Causa probable: la dependencia entre las K^2 distancias de B (vecinos compartidos y pares repetidos), que el test trata como independientes. Detalle: stoodx_validacion/calibracion_id_val.csv.',
    'k_c_resultado': 'stoodx_100_kc: variante exploratoria con resultado negativo; no corrige la degradación de K = 500 (validación: LYM 0,890 -> 0,790; completo: near_nct_tum_norm 0,943 frente a 0,953 de stoodx_100). El mecanismo de la degradación se documenta con la exclusión de clases con K/n_c alto (stoodx_validacion/auroc_por_clase_predicha.csv).',
    'test': 'scipy.stats.mannwhitneyu con method="auto" (asintótico con corrección de continuidad y por '
            'empates para estos tamaños).',
}


# ---------------------------------------------------------------------------
# Datos y referencia por clase
# ---------------------------------------------------------------------------

def cargar(tag, nombre):
    return dict(np.load(os.path.join(FEATURES_ROOT, tag, f'{nombre}.npz')))


def normalizar(F):
    return F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)


def preparar_clases(train, quantile, device):
    """Por clase predicha: índices globales en train, features enmascaradas y normalizadas, máscara."""
    pred = train['logits'].argmax(1)
    clases = {}
    for c in range(NUM_CLASSES):
        idx = np.where(pred == c)[0]
        F = torch.from_numpy(train['feats'][idx]).to(device)
        mascara = torch.ones(N_FEATURES, dtype=torch.bool, device=device)
        if quantile > 0:
            # Igual que el código oficial: n_features = #(suma|f| >= cuantil); se conservan las más presentes
            suma = F.abs().sum(0)
            n_features = int((suma >= torch.quantile(suma, quantile).item()).sum().item())
            mascara = torch.zeros_like(mascara)
            mascara[torch.argsort(suma, descending=True)[:n_features]] = True
        clases[c] = {'idx': idx, 'n': len(idx), 'mascara': mascara, 'Fn': normalizar(F * mascara)}
    return clases


def k_clase(n, K, regla):
    """(vecinos de x, vecinos de cada vecino sin sí mismo) para una clase con n imágenes de train."""
    if regla == 'mitad':
        kc = min(K, n // 2)
        return kc, kc
    return min(K, n), min(K, n - 1)


@torch.no_grad()
def precalcular(clases, K, regla='fija'):
    """Tabla por clase: (n_c, kref) distancias ascendentes a los vecinos de la clase sin uno mismo."""
    tablas = {}
    for c, d in clases.items():
        n, Fn = d['n'], d['Fn']
        kref = k_clase(n, K, regla)[1]
        vals = torch.empty((n, kref), device=Fn.device)
        inds = torch.empty((n, kref), dtype=torch.long, device=Fn.device)
        for ini in range(0, n, BLOQUE):
            fin = min(ini + BLOQUE, n)
            D = 1 - Fn[ini:fin] @ Fn.T
            filas = torch.arange(fin - ini, device=Fn.device)
            D[filas, ini + filas] = float('inf')          # excluye a la propia imagen (por índice)
            v, i = torch.topk(D, kref, dim=1, largest=False, sorted=True)
            vals[ini:fin], inds[ini:fin] = v, i
        tablas[c] = {'vals': vals, 'inds': inds}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return tablas


def cortar_tablas(tablas, K):
    return {c: {'vals': t['vals'][:, :K], 'inds': t['inds'][:, :K]} for c, t in tablas.items()}


# ---------------------------------------------------------------------------
# Test de Mann-Whitney (CPU, scipy, en paralelo)
# ---------------------------------------------------------------------------

def mann_whitney(args):
    A, B = args
    A, B = A.astype(np.float64), B.astype(np.float64)
    r = stats.mannwhitneyu(A, B, alternative='greater')
    n1, n2 = len(A), len(B)
    n = n1 + n2
    # z del método asintótico de scipy (corrección por empates y de continuidad): p = norm.sf(z)
    _, t = np.unique(np.concatenate([A, B]), return_counts=True)
    t = t.astype(np.float64)
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1))))
    z = (r.statistic - n1 * n2 / 2 - 0.5) / sigma
    return float(r.pvalue), float(r.statistic), float(z)


@torch.no_grad()
def puntuar(datos, clases, tablas, K, pool, regla='fija', n_vecinos_guardar=5):
    """Devuelve p, U, z, pred, los n_vecinos_guardar vecinos (índice global en train y distancia) y tiempos."""
    X = torch.from_numpy(datos['feats'])
    pred = datos['logits'].argmax(1)
    n = len(pred)
    p, U, z = np.empty(n), np.empty(n), np.empty(n)
    vec_idx = np.full((n, n_vecinos_guardar), -1, dtype=np.int64)
    vec_dist = np.full((n, n_vecinos_guardar), np.nan)
    t_gpu = t_cpu = 0.0
    for c in range(NUM_CLASSES):
        sel = np.where(pred == c)[0]
        if len(sel) == 0:
            continue
        d, tabla = clases[c], tablas[c]
        kx = k_clase(d['n'], K, regla)[0]
        for ini in range(0, len(sel), LOTE):
            s = sel[ini:ini + LOTE]
            t0 = time.perf_counter()
            Xn = normalizar(X[s].to(d['Fn'].device) * d['mascara'])
            D = 1 - Xn @ d['Fn'].T
            a, ai = torch.topk(D, kx, dim=1, largest=False, sorted=True)
            B = tabla['vals'][ai]                       # (lote, kx, kref)
            a_np, B_np, ai_np = a.cpu().numpy(), B.reshape(len(s), -1).cpu().numpy(), ai.cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            res = pool.map(mann_whitney, [(a_np[j], B_np[j]) for j in range(len(s))], chunksize=4)
            t_cpu += time.perf_counter() - t1
            t_gpu += t1 - t0
            p[s], U[s], z[s] = np.array(res).T
            m = min(n_vecinos_guardar, kx)
            vec_idx[s, :m] = d['idx'][ai_np[:, :m]]
            vec_dist[s, :m] = a_np[:, :m]
    return {'p': p, 'U': U, 'z': z, 'pred': pred, 'vecinos_idx': vec_idx, 'vecinos_dist': vec_dist,
            't_gpu_s': t_gpu, 't_cpu_s': t_cpu}


def versiones():
    paquetes = {}
    for nombre in ['numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'matplotlib']:
        try:
            paquetes[nombre] = md.version(nombre)
        except md.PackageNotFoundError:
            paquetes[nombre] = None
    return {'python': platform.python_version(), **paquetes,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'procesos_cpu_test': N_PROCESOS}


def clases_pequenas(clases, K):
    return [c for c, d in clases.items() if d['n'] <= K]


# ---------------------------------------------------------------------------
# Validación en un subconjunto
# ---------------------------------------------------------------------------

def construir_subconjunto(tag, datos):
    rng = np.random.default_rng(SEMILLA)
    sub = {}
    # id_test estratificado por clase real (asignación proporcional, al menos 1 por clase)
    label = datos['id_test']['label']
    clases, cuentas = np.unique(label, return_counts=True)
    cuota = np.maximum(1, np.floor(N_VAL_ID * cuentas / cuentas.sum()).astype(int))
    resto = N_VAL_ID - cuota.sum()
    frac = N_VAL_ID * cuentas / cuentas.sum() - np.floor(N_VAL_ID * cuentas / cuentas.sum())
    for i in np.argsort(-frac)[:max(resto, 0)]:
        cuota[i] += 1
    idx = np.concatenate([rng.choice(np.where(label == c)[0], q, replace=False) for c, q in zip(clases, cuota)])
    sub['id_test'] = np.sort(idx)
    for nombre in VAL_OOD:
        sub[nombre] = np.sort(rng.choice(len(datos[nombre]['label']), N_VAL_OOD, replace=False))
    return sub


def subconjunto(d, idx):
    return {k: v[idx] for k, v in d.items()}


def filas_metricas(res, sub_datos, etiqueta, clases_peq):
    """Métricas con p y con -z para cada conjunto OOD frente a id_test; también sin las clases pequeñas.
    z crece cuanto más lejos está x (p = norm.sf(z)), así que la puntuación equivalente a p es -z."""
    idt, idt_d = res['id_test'], sub_datos['id_test']
    filas = []
    peq_id = np.isin(idt['pred'], clases_peq)
    for nombre in VAL_OOD:
        ood = res[nombre]
        peq_ood = np.isin(ood['pred'], clases_peq)
        fila = {**etiqueta, 'conjunto': nombre, 'N_ID': len(idt['p']), 'N_OOD': len(ood['p'])}
        for etiqueta_score, f in [('p', lambda r: r['p']), ('menos_z', lambda r: -r['z'])]:
            m = metricas_ood.todas_las_metricas(f(idt), f(ood), idt['pred'], idt_d['label'], NUM_CLASSES)
            fila.update({f'AUROC_{etiqueta_score}': m['AUROC'], f'FPR@95_{etiqueta_score}': m['FPR@95'],
                         f'CCR@FPR5_{etiqueta_score}': m['CCR@FPR5']})
        fila['max_dif_p_vs_menos_z'] = max(abs(fila[f'{k}_p'] - fila[f'{k}_menos_z'])
                                           for k in ['AUROC', 'FPR@95', 'CCR@FPR5'])
        fila.update({'prop_p1_ood': float(np.mean(ood['p'] == 1)), 'prop_p0_ood': float(np.mean(ood['p'] == 0)),
                     'prop_p1_id_test': float(np.mean(idt['p'] == 1)), 'prop_p0_id_test': float(np.mean(idt['p'] == 0))})
        fila['clases_pequenas'] = ','.join(map(str, clases_peq))
        fila['N_ID_en_clases_pequenas'] = int(peq_id.sum())
        fila['N_OOD_en_clases_pequenas'] = int(peq_ood.sum())
        # (a) sin las imágenes de id_test predichas en clases con n_c <= K; (b) sin las de id_test ni las OOD
        for sufijo, keep_id, keep_ood in [('sin_clases_pequenas_id', ~peq_id, np.ones(len(peq_ood), bool)),
                                          ('sin_clases_pequenas_id_y_ood', ~peq_id, ~peq_ood)]:
            if peq_id.any() or (peq_ood.any() and sufijo.endswith('ood')):
                for etiqueta_score, f in [('p', lambda r: r['p']), ('menos_z', lambda r: -r['z'])]:
                    m = metricas_ood.todas_las_metricas(f(idt)[keep_id], f(ood)[keep_ood], idt['pred'][keep_id],
                                                        idt_d['label'][keep_id], NUM_CLASSES)
                    fila.update({f'AUROC_{etiqueta_score}_{sufijo}': m['AUROC'],
                                 f'FPR@95_{etiqueta_score}_{sufijo}': m['FPR@95'],
                                 f'CCR@FPR5_{etiqueta_score}_{sufijo}': m['CCR@FPR5']})
        filas.append(fila)
    return filas


def reanalizar_validacion(args):
    """Recalcula las tablas y la figura de la validación a partir de las puntuaciones guardadas (sin GPU)."""
    salida = args.salida
    cfg = json.load(open(os.path.join(salida, 'config.json')))
    tam = list(cfg['tamano_clases_train_pred'].values())
    filas_previas = pd.read_csv(os.path.join(salida, 'barrido_K_y_configuraciones.csv'))
    ruta_t = os.path.join(salida, 'tiempos_validacion.json')
    tiempos = json.load(open(ruta_t)) if os.path.exists(ruta_t) else {}
    filas = []
    combinaciones = [('100', K) for K in KS] + [('875', 500)]
    if os.path.exists(os.path.join(salida, 'scores_validacion_100_kc_K500.npz')):
        combinaciones.append(('100_kc', 500))
    for config, K in combinaciones:
        d = dict(np.load(os.path.join(salida, f'scores_validacion_{config}_K{K}.npz')))
        res = {n: {'p': d[f'{n}_p'], 'z': d[f'{n}_z'], 'pred': d[f'{n}_pred']} for n in ['id_test'] + VAL_OOD}
        sub_datos = {n: {'label': d[f'{n}_label']} for n in ['id_test'] + VAL_OOD}
        clave = f'{config}_K{K}'
        if clave in tiempos:
            t = tiempos[clave]
        else:
            previa = filas_previas[(filas_previas.config.astype(str) == config) & (filas_previas.K == K)].iloc[0]
            t = {k: float(previa[k]) for k in ['t_precalculo_s', 'ms_por_imagen_gpu', 'ms_por_imagen_cpu']}
            tiempos[clave] = t
        etiqueta = {'config': config, 'K': K, **t}
        # Con la regla K_c (<= n_c / 2) ninguna clase usa la clase entera
        peq = [] if CONFIGS[config]['regla_k'] == 'mitad' else [c for c, n in enumerate(tam) if n <= K]
        filas.extend(filas_metricas(res, sub_datos, etiqueta, peq))
    json.dump(tiempos, open(ruta_t, 'w'), indent=1)
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, 'barrido_K_y_configuraciones.csv'), index=False)
    df[df.config == '100'].to_csv(os.path.join(salida, 'barrido_K.csv'), index=False)
    df[df.K == 500].to_csv(os.path.join(salida, 'comparacion_100_vs_875.csv'), index=False)
    figura_K(df, salida)
    print(f'Tablas de validación recalculadas: {sorted(set(zip(df.config.astype(str), df.K)))}')
    return df


def figura_K(df, salida):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = df[df.config == '100']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, nombre in enumerate(VAL_OOD):
        dd = d[d.conjunto == nombre].sort_values('K')
        color = f'C{i}'
        ax1.plot(dd.K, dd.AUROC_p, 'o-', color=color, label=f'{nombre} (p)')
        ax1.plot(dd.K, dd.AUROC_menos_z, 'x--', color=color, alpha=0.7, label=f'{nombre} (-z)')
    ax1.set_xscale('log')
    ax1.set_xticks(KS, KS)
    ax1.set_xlabel('K')
    ax1.set_ylabel('AUROC')
    ax1.legend(fontsize=6.5, frameon=False, ncol=2)
    ax1.grid(alpha=0.3)
    t = d.drop_duplicates('K').sort_values('K')
    ax2.plot(t.K, t.ms_por_imagen_gpu, 'o-', label='GPU (A y B)')
    ax2.plot(t.K, t.ms_por_imagen_cpu, 's-', label=f'CPU (test, {N_PROCESOS} procesos)')
    ax2.plot(t.K, t.ms_por_imagen_gpu + t.ms_por_imagen_cpu, 'k--', label='total')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks(KS, KS)
    ax2.set_xlabel('K')
    ax2.set_ylabel('ms por imagen')
    ax2.legend(fontsize=8, frameon=False)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(salida, f'fig_barrido_K.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)


def figuras_ejemplos(res, sub_datos, train_paths, nombres_clases, salida, sufijo='', titulo='100 %, K=500'):
    """Explicación por ejemplos (etapa 2 del paper): imagen + 5 vecinos de train de la clase predicha."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rng = np.random.default_rng(SEMILLA)

    def elegir(p, cola, n=5):
        orden = np.argsort(p, kind='stable')
        diez = max(n, int(np.ceil(0.10 * len(p))))
        candidatos = orden[:diez] if cola == 'baja' else orden[-diez:]
        return np.sort(rng.choice(candidatos, n, replace=False))

    # Orden por la puntuación principal -z (p tiene empates en 0 con K = 500)
    categorias = [
        ('id_alta', 'id_test', elegir(-res['id_test']['z'], 'alta')),
        ('id_baja_falsas_alarmas', 'id_test', elegir(-res['id_test']['z'], 'baja')),
        ('nct_tum_norm_baja_detectadas', 'near_nct_tum_norm', elegir(-res['near_nct_tum_norm']['z'], 'baja')),
        ('nct_tum_norm_alta_aceptadas', 'near_nct_tum_norm', elegir(-res['near_nct_tum_norm']['z'], 'alta')),
        ('nct_lym_azar', 'afin_nct_lym', np.sort(rng.choice(len(res['afin_nct_lym']['p']), 5, replace=False))),
        ('dtd_azar', 'far_dtd', np.sort(rng.choice(len(res['far_dtd']['p']), 5, replace=False))),
    ]
    seleccion = {}
    for nombre_fig, conj, idx in categorias:
        r, d = res[conj], sub_datos[conj]
        fig, axes = plt.subplots(5, 6, figsize=(12, 11))
        filas = []
        for fila, i in enumerate(idx):
            ax = axes[fila, 0]
            ax.imshow(Image.open(str(d['paths'][i])).convert('RGB'))
            real = nombres_clases[d['label'][i]] if d['label'][i] >= 0 else 'OOD'
            ax.set_title(f'{conj}\nreal: {real[:22]}\npred: {nombres_clases[r["pred"][i]][:22]}\n'
                         f'p={r["p"][i]:.3g}  z={r["z"][i]:.2f}', fontsize=6.5)
            for j in range(5):
                ax = axes[fila, j + 1]
                g = r['vecinos_idx'][i, j]
                if g >= 0:
                    ax.imshow(Image.open(str(train_paths[g])).convert('RGB'))
                    ax.set_title(f'vecino {j + 1}\nd={r["vecinos_dist"][i, j]:.4f}', fontsize=6.5)
            filas.append({'ruta': str(d['paths'][i]), 'p': float(r['p'][i]), 'z': float(r['z'][i]),
                          'pred': nombres_clases[r['pred'][i]],
                          'vecinos': [str(train_paths[g]) for g in r['vecinos_idx'][i] if g >= 0],
                          'distancias': [float(x) for x in r['vecinos_dist'][i]]})
        for ax in axes.ravel():
            ax.axis('off')
        fig.suptitle(f'STOOD-X (paper, {titulo}): {nombre_fig}, orden por -z. Columna 1: imagen; 2-6: vecinos de '
                     f'train de la clase predicha', fontsize=9)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(salida, f'ejemplos{sufijo}_{nombre_fig}.{ext}'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        seleccion[nombre_fig] = filas
    with open(os.path.join(salida, f'ejemplos{sufijo}_seleccion.json'), 'w') as f:
        json.dump(seleccion, f, indent=2, ensure_ascii=False)


def validacion(args, pool):
    device = torch.device('cuda')
    salida = args.salida
    os.makedirs(salida, exist_ok=True)
    tag = args.features_tag
    datos = {n: cargar(tag, n) for n in ['id_test'] + VAL_OOD}
    train = cargar(tag, 'id_train')
    info = json.load(open(os.path.join(FEATURES_ROOT, tag, 'info.json')))
    nombres_clases = info['clases_id']

    ruta_sub = os.path.join(salida, 'subconjunto_validacion.json')
    if os.path.exists(ruta_sub):
        sub = {k: np.array(v['indices']) for k, v in json.load(open(ruta_sub)).items()}
    else:
        sub = construir_subconjunto(tag, datos)
        json.dump({k: {'indices': v.tolist(), 'rutas': [str(x) for x in datos[k]['paths'][v]]} for k, v in sub.items()},
                  open(ruta_sub, 'w'), indent=1, ensure_ascii=False)
    sub_datos = {k: subconjunto(datos[k], v) for k, v in sub.items()}
    print('Subconjunto:', {k: len(v) for k, v in sub.items()}, flush=True)

    filas, registro = [], {'tiempos_precalculo_s': {}, 'tamano_clases_train_pred': None}
    resultados = {}

    def evaluar(config, K, tablas, clases, t_pre, etiqueta_extra=None):
        res = {n: puntuar(sub_datos[n], clases, tablas, K, pool) for n in sub_datos}
        n_img = sum(len(r['p']) for r in res.values())
        etiqueta = {'config': config, 'K': K, 't_precalculo_s': t_pre,
                    'ms_por_imagen_gpu': 1000 * sum(r['t_gpu_s'] for r in res.values()) / n_img,
                    'ms_por_imagen_cpu': 1000 * sum(r['t_cpu_s'] for r in res.values()) / n_img,
                    **(etiqueta_extra or {})}
        filas.extend(filas_metricas(res, sub_datos, etiqueta, clases_pequenas(clases, K)))
        # Comprobación p = norm.sf(z) (z es la del método asintótico de scipy)
        dif = max(float(np.max(np.abs(stats.norm.sf(r['z']) - r['p']))) for r in res.values())
        print(f'  config={config} K={K}: precálculo {t_pre:.1f} s, {etiqueta["ms_por_imagen_gpu"]:.2f} ms/img GPU, '
              f'{etiqueta["ms_por_imagen_cpu"]:.2f} ms/img CPU; max|sf(z)-p|={dif:.2e}', flush=True)
        registro.setdefault('max_dif_sf_z_vs_p', {})[f'{config}_K{K}'] = dif
        return res

    # 4a. Barrido de K con el 100 % de las features
    clases = preparar_clases(train, CONFIGS['100']['quantile'], device)
    registro['tamano_clases_train_pred'] = {nombres_clases[c]: d['n'] for c, d in clases.items()}
    tablas_corte = None
    for K in KS:
        t0 = time.perf_counter()
        tablas = precalcular(clases, K)
        t_pre = time.perf_counter() - t0
        resultados[('100', K)] = evaluar('100', K, tablas, clases, t_pre)
        if K == max(KS):
            tablas_corte = tablas
        del tablas
        torch.cuda.empty_cache()

    # Comprobación: tabla de K=500 cortada a K_CORTE columnas == cálculo directo con K_CORTE
    directo = precalcular(clases, K_CORTE)
    cortado = cortar_tablas(tablas_corte, K_CORTE)
    tabla_igual = all(torch.equal(directo[c]['vals'], cortado[c]['vals']) for c in clases)
    indices_iguales = all(torch.equal(directo[c]['inds'], cortado[c]['inds']) for c in clases)
    res_cortado = {n: puntuar(sub_datos[n], clases, cortado, K_CORTE, pool) for n in sub_datos}
    p_iguales = all(np.array_equal(res_cortado[n]['p'], resultados[('100', K_CORTE)][n]['p']) for n in sub_datos)
    registro['comprobacion_corte'] = {'K': K_CORTE, 'distancias_identicas': tabla_igual,
                                      'indices_identicos': indices_iguales, 'p_identicos': p_iguales}
    print(f'Corte K=500 -> K={K_CORTE}: distancias idénticas={tabla_igual}, índices idénticos={indices_iguales}, '
          f'p idénticos={p_iguales}', flush=True)
    del directo, cortado, tablas_corte
    torch.cuda.empty_cache()

    # 4b. 87,5 % con K = 500
    clases875 = preparar_clases(train, CONFIGS['875']['quantile'], device)
    registro['features_conservadas_875_por_clase'] = {nombres_clases[c]: int(d['mascara'].sum()) for c, d in clases875.items()}
    t0 = time.perf_counter()
    tablas = precalcular(clases875, 500)
    resultados[('875', 500)] = evaluar('875', 500, tablas, clases875, time.perf_counter() - t0)
    del tablas
    torch.cuda.empty_cache()

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, 'barrido_K_y_configuraciones.csv'), index=False)
    df[df.config == '100'].to_csv(os.path.join(salida, 'barrido_K.csv'), index=False)
    df[df.K == 500].to_csv(os.path.join(salida, 'comparacion_100_vs_875.csv'), index=False)
    figura_K(df, salida)

    # Puntuaciones del subconjunto (por si hacen falta más análisis)
    for (config, K), res in resultados.items():
        np.savez(os.path.join(salida, f'scores_validacion_{config}_K{K}.npz'),
                 **{f'{n}_{k}': v for n, r in res.items() for k, v in r.items() if isinstance(v, np.ndarray)},
                 **{f'{n}_paths': sub_datos[n]['paths'] for n in sub_datos},
                 **{f'{n}_label': sub_datos[n]['label'] for n in sub_datos})

    # 4c. Explicaciones por ejemplos (100 %, K = 500)
    figuras_ejemplos(resultados[('100', 500)], sub_datos, train['paths'], nombres_clases, salida)

    # Imágenes completas por conjunto que caen en clases pequeñas (con K = 500)
    peq = clases_pequenas(clases, 500)
    cuenta = {}
    for n in ['id_val', 'id_test'] + [c for c in CONJUNTOS_PRINCIPALES]:
        pr = cargar(tag, n)['logits'].argmax(1)
        cuenta[n] = {nombres_clases[c]: int(np.sum(pr == c)) for c in peq}
    registro['imagenes_en_clases_pequenas_K500'] = cuenta

    config = {'fase': 'B (validación)', 'metodo': 'stoodx_paper', 'entorno': sys.prefix, 'versiones': versiones(),
              'carpeta_features': os.path.join(FEATURES_ROOT, tag), 'semilla': SEMILLA,
              'subconjunto': {k: int(len(v)) for k, v in sub.items()}, 'Ks': KS, 'configs_quantile': CONFIGS,
              'notas': NOTAS, **registro}
    with open(os.path.join(salida, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=float)

    pd.set_option('display.width', 250)
    cols = ['config', 'K', 'conjunto', 'AUROC_p', 'AUROC_menos_z', 'FPR@95_p', 'FPR@95_menos_z', 'CCR@FPR5_p',
            'CCR@FPR5_menos_z', 'max_dif_p_vs_menos_z', 'prop_p1_id_test', 'prop_p0_id_test', 'prop_p1_ood', 'prop_p0_ood',
            'ms_por_imagen_gpu', 'ms_por_imagen_cpu', 't_precalculo_s']
    print(df[cols].to_string(index=False, float_format='%.4f'))
    print(f'Salida en {salida}')


def cargar_subconjunto(salida, datos):
    sub = {k: np.array(v['indices']) for k, v in json.load(open(os.path.join(salida, 'subconjunto_validacion.json'))).items()}
    return sub, {k: subconjunto(datos[k], v) for k, v in sub.items()}


def validacion_extra(args, pool):
    """GPU. (3a) stoodx_100_kc en el subconjunto; (3c) id_test completo con cada K y con K_c (para el AUROC por
    clase predicha); (3b) p-valores de id_val completo con K = 500 y con K_c (calibración, solo datos ID)."""
    device = torch.device('cuda')
    salida, tag = args.salida, args.features_tag
    datos = {n: cargar(tag, n) for n in ['id_test', 'id_val'] + VAL_OOD}
    train = cargar(tag, 'id_train')
    sub, sub_datos = cargar_subconjunto(salida, datos)
    clases = preparar_clases(train, CONFIGS['100']['quantile'], device)
    ruta_t = os.path.join(salida, 'tiempos_validacion.json')
    tiempos = json.load(open(ruta_t)) if os.path.exists(ruta_t) else {}

    # (3a) K_c en el subconjunto
    t0 = time.perf_counter()
    tablas_kc = precalcular(clases, 500, 'mitad')
    t_pre = time.perf_counter() - t0
    res = {n: puntuar(sub_datos[n], clases, tablas_kc, 500, pool, 'mitad') for n in sub_datos}
    n_img = sum(len(r['p']) for r in res.values())
    tiempos['100_kc_K500'] = {'t_precalculo_s': t_pre,
                              'ms_por_imagen_gpu': 1000 * sum(r['t_gpu_s'] for r in res.values()) / n_img,
                              'ms_por_imagen_cpu': 1000 * sum(r['t_cpu_s'] for r in res.values()) / n_img}
    json.dump(tiempos, open(ruta_t, 'w'), indent=1)
    np.savez(os.path.join(salida, 'scores_validacion_100_kc_K500.npz'),
             **{f'{n}_{k}': v for n, r in res.items() for k, v in r.items() if isinstance(v, np.ndarray)},
             **{f'{n}_paths': sub_datos[n]['paths'] for n in sub_datos},
             **{f'{n}_label': sub_datos[n]['label'] for n in sub_datos})
    print(f'K_c en el subconjunto: {tiempos["100_kc_K500"]}', flush=True)

    # (3c) id_test completo con cada K (regla fija) y con K_c
    completo_test = {'pred': datos['id_test']['logits'].argmax(1), 'label': datos['id_test']['label']}
    for K in KS:
        tablas = precalcular(clases, K)
        r = puntuar(datos['id_test'], clases, tablas, K, pool)
        completo_test[f'K{K}_p'], completo_test[f'K{K}_z'] = r['p'], r['z']
        print(f'id_test completo, K={K}: {1000 * (r["t_gpu_s"] + r["t_cpu_s"]) / len(r["p"]):.2f} ms/img', flush=True)
        del tablas
        torch.cuda.empty_cache()
    r = puntuar(datos['id_test'], clases, tablas_kc, 500, pool, 'mitad')
    completo_test['kc_p'], completo_test['kc_z'] = r['p'], r['z']
    np.savez(os.path.join(salida, 'id_test_completo_scores.npz'), **completo_test)

    # (3b) id_val completo con K = 500 (fija) y con K_c
    val = {'pred': datos['id_val']['logits'].argmax(1), 'label': datos['id_val']['label']}
    tablas = precalcular(clases, 500)
    for clave, tb, regla in [('K500', tablas, 'fija'), ('kc', tablas_kc, 'mitad')]:
        r = puntuar(datos['id_val'], clases, tb, 500, pool, regla)
        val[f'{clave}_p'], val[f'{clave}_z'] = r['p'], r['z']
        print(f'id_val completo, {clave}: {1000 * (r["t_gpu_s"] + r["t_cpu_s"]) / len(r["p"]):.2f} ms/img', flush=True)
    np.savez(os.path.join(salida, 'id_val_calibracion.npz'), **val)
    print('validacion_extra terminada', flush=True)


def analisis_validacion(args):
    """CPU. Tablas de la validación (con stoodx_100_kc), calibración en id_val (3b), análisis por clase predicha
    (3c) y ejemplos ordenados por -z (4)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    salida, tag = args.salida, args.features_tag
    info = json.load(open(os.path.join(FEATURES_ROOT, tag, 'info.json')))
    nombres = info['clases_id']
    cfg = json.load(open(os.path.join(salida, 'config.json')))
    n_c = np.array(list(cfg['tamano_clases_train_pred'].values()))
    k_c = n_c // 2
    reanalizar_validacion(args)

    # ---- 3b. Calibración en id_val (solo datos ID): ¿p aproximadamente uniforme por clase predicha?
    val = dict(np.load(os.path.join(salida, 'id_val_calibracion.npz')))
    filas = []
    for clave, etiqueta in [('K500', 'K=500'), ('kc', 'K_c')]:
        for c in list(range(NUM_CLASSES)) + ['todas']:
            sel = np.ones(len(val['pred']), bool) if c == 'todas' else val['pred'] == c
            p = val[f'{clave}_p'][sel]
            fila = {'config': etiqueta, 'clase_pred': 'todas' if c == 'todas' else nombres[c],
                    'n_c_train': int(n_c.sum()) if c == 'todas' else int(n_c[c]),
                    'K_usado': '' if c == 'todas' else (min(500, n_c[c]) if clave == 'K500' else int(k_c[c])),
                    'n_val': int(sel.sum())}
            if sel.sum() >= 20:
                ks = stats.kstest(p, 'uniform')
                fila.update({'KS_D': ks.statistic, 'KS_p': ks.pvalue, 'prop_p_lt_0.05': float(np.mean(p < 0.05)),
                             'prop_p_0': float(np.mean(p == 0)), 'prop_p_1': float(np.mean(p == 1)),
                             'mediana_p': float(np.median(p))})
            filas.append(fila)
    cal = pd.DataFrame(filas)
    cal.to_csv(os.path.join(salida, 'calibracion_id_val.csv'), index=False)
    fig, axes = plt.subplots(3, 5, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), list(range(NUM_CLASSES)) + ['todas']):
        sel = np.ones(len(val['pred']), bool) if c == 'todas' else val['pred'] == c
        for clave, color, etiqueta in [('K500', 'tab:blue', 'K=500'), ('kc', 'tab:orange', 'K_c')]:
            ax.hist(val[f'{clave}_p'][sel], bins=20, range=(0, 1), density=True, alpha=0.5, color=color, label=etiqueta)
        ax.axhline(1, color='k', lw=0.6, ls=':')
        titulo = 'todas' if c == 'todas' else f'{nombres[c][:24]}\nn_c={n_c[c]}, n_val={sel.sum()}'
        ax.set_title(titulo, fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes.ravel()[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Calibración en id_val: histograma de p por clase predicha (uniforme = línea de puntos)', fontsize=10)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(salida, f'fig_calibracion_id_val.{ext}'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---- 3c. Clases predichas en los conjuntos OOD, K/n_c y AUROC (-z) por clase predicha
    sub = json.load(open(os.path.join(salida, 'subconjunto_validacion.json')))
    distrib = []
    for nombre in VAL_OOD + ['id_test']:
        pred_full = cargar(tag, nombre)['logits'].argmax(1)
        pred_sub = pred_full[np.array(sub[nombre]['indices'])]
        for c in range(NUM_CLASSES):
            distrib.append({'conjunto': nombre, 'clase_pred': nombres[c], 'n_c_train': int(n_c[c]),
                            **{f'K{K}/n_c': min(K, n_c[c]) / n_c[c] for K in KS}, 'K_c/n_c': k_c[c] / n_c[c],
                            'n_subconjunto': int(np.sum(pred_sub == c)),
                            'frac_subconjunto': float(np.mean(pred_sub == c)),
                            'n_completo': int(np.sum(pred_full == c)), 'frac_completo': float(np.mean(pred_full == c))})
    distrib = pd.DataFrame(distrib)
    distrib.to_csv(os.path.join(salida, 'clases_predichas_por_conjunto.csv'), index=False)

    test = dict(np.load(os.path.join(salida, 'id_test_completo_scores.npz')))
    filas = []
    configs = [(f'K{K}', K) for K in KS] + [('kc', None)]
    for nombre in VAL_OOD:
        datos_ood = {}
        for clave, K in configs:
            ruta = (os.path.join(salida, f'scores_validacion_100_K{K}.npz') if K
                    else os.path.join(salida, 'scores_validacion_100_kc_K500.npz'))
            d = np.load(ruta)
            datos_ood[clave] = (d[f'{nombre}_z'], d[f'{nombre}_pred'])
        for c in range(NUM_CLASSES):
            n_ood = int(np.sum(datos_ood['K9'][1] == c))
            n_id = int(np.sum(test['pred'] == c))
            fila = {'conjunto': nombre, 'clase_pred': nombres[c], 'n_c_train': int(n_c[c]), 'n_ood': n_ood,
                    'n_id_test': n_id}
            for clave, K in configs:
                z_ood, pred_ood = datos_ood[clave]
                fila[f'K/n_c_{clave}'] = (min(K, n_c[c]) if K else k_c[c]) / n_c[c]
                if n_ood >= 20 and n_id >= 20:
                    fila[f'AUROC_{clave}'] = metricas_ood.auroc(-test[f'{clave}_z'][test['pred'] == c],
                                                                -z_ood[pred_ood == c])
                # Detectabilidad frente a todo id_test (sin exigir ID de la misma clase)
                if n_ood >= 20:
                    fila[f'AUROC_vs_todo_id_{clave}'] = metricas_ood.auroc(-test[f'{clave}_z'], -z_ood[pred_ood == c])
            filas.append(fila)
    por_clase = pd.DataFrame(filas)
    por_clase.to_csv(os.path.join(salida, 'auroc_por_clase_predicha.csv'), index=False)

    # ---- 4. Ejemplos ordenados por -z, con K = 500 y con K_c
    train_paths = cargar(tag, 'id_train')['paths']
    datos = {n: cargar(tag, n) for n in ['id_test'] + VAL_OOD}
    _, sub_datos = cargar_subconjunto(salida, datos)
    for archivo, sufijo, titulo in [('scores_validacion_100_K500.npz', '_K500', '100 %, K=500'),
                                    ('scores_validacion_100_kc_K500.npz', '_kc', '100 %, K_c=min(500, n_c/2)')]:
        d = np.load(os.path.join(salida, archivo))
        res = {n: {k: d[f'{n}_{k}'] for k in ['p', 'z', 'pred', 'vecinos_idx', 'vecinos_dist']} for n in sub_datos}
        figuras_ejemplos(res, sub_datos, train_paths, nombres, salida, sufijo, titulo)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(cal.to_string(index=False, float_format='%.3f'))
    cols = ['conjunto', 'clase_pred', 'n_c_train', 'n_ood', 'n_id_test'] + \
           [f'AUROC_{k}' for k, _ in configs] + ['AUROC_vs_todo_id_K288', 'AUROC_vs_todo_id_K500', 'AUROC_vs_todo_id_kc']
    print(por_clase[por_clase.n_ood >= 20][[c for c in cols if c in por_clase]].to_string(index=False, float_format='%.3f'))


# ---------------------------------------------------------------------------
# Selección de K solo con datos ID (sin data snooping)
# ---------------------------------------------------------------------------

def seleccion_k(args, pool):
    """GPU. Leave-one-class-out en id_val: para cada clase h, se quitan de la referencia las imágenes de train con
    etiqueta h, se predice con el logit de h anulado y las imágenes de id_val de clase h hacen de pseudo-OOD frente
    al resto de id_val. Regla fijada ANTES de ver resultados: K = argmax del AUROC medio (-z) sobre las 14 clases;
    a igualdad (redondeo a 3 decimales), el K más pequeño. No se usa ningún dato de test ni OOD."""
    device = torch.device('cuda')
    salida, tag = args.salida, args.features_tag
    os.makedirs(salida, exist_ok=True)
    train, val = cargar(tag, 'id_train'), cargar(tag, 'id_val')
    nombres = json.load(open(os.path.join(FEATURES_ROOT, tag, 'info.json')))['clases_id']
    filas = []
    for h in range(NUM_CLASSES):
        keep = train['label'] != h
        tr = {k: v[keep] for k, v in train.items()}
        tr['logits'] = tr['logits'].copy()
        tr['logits'][:, h] = -np.inf
        clases = preparar_clases(tr, 0.0, device)
        del clases[h]                                   # sin referencia para la clase retenida
        va = {'feats': val['feats'], 'logits': val['logits'].copy()}
        va['logits'][:, h] = -np.inf
        es_ood = val['label'] == h
        tablas = precalcular(clases, max(KS))
        for K in KS:
            r = puntuar(va, clases, cortar_tablas(tablas, K), K, pool)   # corte == cálculo directo (K_CORTE)
            conf = -r['z']
            filas.append({'clase_retenida': nombres[h], 'K': K, 'N_ID': int((~es_ood).sum()), 'N_OOD': int(es_ood.sum()),
                          'AUROC': metricas_ood.auroc(conf[~es_ood], conf[es_ood]),
                          'FPR@95': metricas_ood.fpr_at_tpr(conf[~es_ood], conf[es_ood])})
            print(f'  h={nombres[h]:<28} K={K:<4} AUROC={filas[-1]["AUROC"]:.4f} FPR@95={filas[-1]["FPR@95"]:.4f}', flush=True)
        del tablas, clases
        torch.cuda.empty_cache()
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(salida, 'seleccion_k_loco_por_clase.csv'), index=False)
    res = df.groupby('K')[['AUROC', 'FPR@95']].mean().reset_index()
    mejor = res.AUROC.round(3).max()
    k_elegido = int(res[res.AUROC.round(3) == mejor].K.min())
    res.to_csv(os.path.join(salida, 'seleccion_k_loco.csv'), index=False)
    json.dump({'K_elegido': k_elegido, 'criterio': seleccion_k.__doc__, 'Ks': KS,
               'AUROC_medio_por_K': dict(zip(res.K.astype(int).astype(str), res.AUROC))},
              open(os.path.join(salida, 'K_elegido.json'), 'w'), indent=2, ensure_ascii=False)
    print(res.to_string(index=False, float_format='%.4f'))
    print(f'K elegido (solo ID): {k_elegido}')


# ---------------------------------------------------------------------------
# Ejecución completa
# ---------------------------------------------------------------------------

def completo(args, pool):
    device = torch.device('cuda')
    partes = os.path.join(args.salida, f'partes_{args.config}')
    os.makedirs(partes, exist_ok=True)
    train = cargar(args.features_tag, 'id_train')
    t0 = time.perf_counter()
    cfg = CONFIGS[args.config]
    clases = preparar_clases(train, cfg['quantile'], device)
    tablas = precalcular(clases, args.k, cfg['regla_k'])
    t_pre = time.perf_counter() - t0
    print(f'stoodx_{args.config}: precálculo K={args.k} (regla {cfg["regla_k"]}) en {t_pre:.1f} s', flush=True)
    for nombre in args.conjuntos:
        destino = os.path.join(partes, f'{nombre}.npz')
        if os.path.exists(destino):
            print(f'  {nombre}: ya calculado, se salta')
            continue
        datos = cargar(args.features_tag, nombre)
        r = puntuar(datos, clases, tablas, args.k, pool, cfg['regla_k'])
        n = len(r['p'])
        print(f'  {nombre}: {n} imágenes, {1000 * r["t_gpu_s"] / n:.2f} ms/img GPU + '
              f'{1000 * r["t_cpu_s"] / n:.2f} ms/img CPU; p==1: {np.mean(r["p"] == 1):.3f}, '
              f'p==0: {np.mean(r["p"] == 0):.3f}', flush=True)
        # conf = -z (puntuación principal, más alto = más ID); p y U se guardan también
        np.savez(destino, conf=-r['z'], p=r['p'], U=r['U'], z=r['z'], pred=r['pred'], label=datos['label'],
                 paths=datos['paths'], vecinos_idx=r['vecinos_idx'], vecinos_dist=r['vecinos_dist'],
                 t_gpu_s=r['t_gpu_s'], t_cpu_s=r['t_cpu_s'], t_precalculo_s=t_pre,
                 gpu=torch.cuda.get_device_name(0))


def combinar(args):
    info = json.load(open(os.path.join(FEATURES_ROOT, args.features_tag, 'info.json')))
    nombres_clases = info['clases_id']
    filas_p, filas_a, filas_n, registro = [], [], [], {}
    for config in CONFIGS:
        metodo = f'stoodx_{config}'
        partes = os.path.join(args.salida, f'partes_{config}')
        punt, tiempos = {}, {}
        faltan = [n for n in A_PUNTUAR if not os.path.exists(os.path.join(partes, f'{n}.npz'))]
        if faltan:
            print(f'{metodo}: faltan {faltan}; se omite')
            continue
        for n in A_PUNTUAR:
            d = dict(np.load(os.path.join(partes, f'{n}.npz')))
            tiempos[n] = {'n': int(len(d['conf'])), 't_gpu_s': float(d.pop('t_gpu_s')),
                          't_cpu_s': float(d.pop('t_cpu_s')), 't_precalculo_s': float(d.pop('t_precalculo_s')),
                          'gpu': str(d.pop('gpu'))}
            punt[n] = d
        for n, comp in UNIONES.items():
            juntos = {k: np.concatenate([punt[c][k] for c in comp]) for k in punt[comp[0]]}
            orden = np.argsort(juntos['paths'], kind='stable')
            punt[n] = {k: v[orden] for k, v in juntos.items()}
        for n, d in punt.items():
            ref = np.load(os.path.join(FEATURES_ROOT, args.features_tag, f'{n}.npz'))['paths']
            assert np.array_equal(ref, d['paths']), f'El orden de {n} no coincide con las features de la fase A'
        np.savez(os.path.join(args.salida, f'scores_{metodo}.npz'),
                 **{f'{n}_{k}': v for n, d in punt.items() for k, v in d.items()})

        idt = punt['id_test']
        acc_id = metricas_ood.exactitud(idt['pred'], idt['label'])
        for nombre in CONJUNTOS_PRINCIPALES:
            ood = punt[nombre]
            m = metricas_ood.todas_las_metricas(idt['conf'], ood['conf'], idt['pred'], idt['label'], NUM_CLASSES)
            mp = metricas_ood.todas_las_metricas(idt['p'], ood['p'], idt['pred'], idt['label'], NUM_CLASSES)   # p (secundaria)
            acc_c = metricas_ood.exactitud(ood['pred'], ood['label']) if nombre == CONTROL_NAME else np.nan
            base = {'metodo': metodo, 'conjunto': nombre, 'N_ID': len(idt['conf']), 'N_OOD': len(ood['conf'])}
            fila = {**base, 'AUROC': m['AUROC'], 'FPR@95': m['FPR@95'], 'CCR@FPR5': m['CCR@FPR5'], 'ACC_ID': acc_id}
            filas_p.append({**fila, 'ACC_control_res_128': acc_c})
            filas_a.append({**fila, 'AUPR_In': m['AUPR_In'], 'AUPR_Out': m['AUPR_Out'],
                            'FPR_real_CCR': m['FPR_real_CCR'], 'tau_CCR': m['tau_CCR'],
                            'AUROC_p': mp['AUROC'], 'FPR@95_p': mp['FPR@95'], 'CCR@FPR5_p': mp['CCR@FPR5'],
                            'prop_p1_ood': float(np.mean(ood['p'] == 1)), 'prop_p0_ood': float(np.mean(ood['p'] == 0)),
                            **{f'CCR@FPR5_{c}': v for c, v in zip(nombres_clases, m['CCR@FPR5_por_clase'])}})
            if nombre in CONJUNTOS_NCT_POR_CLASE:
                filas_n.append({**base, 'clase_nct': nombre.split('_')[-1].upper(), 'papel': nombre.split('_')[0],
                                'AUROC': m['AUROC'], 'FPR@95': m['FPR@95'], 'CCR@FPR5': m['CCR@FPR5'],
                                'AUPR_In': m['AUPR_In'], 'AUPR_Out': m['AUPR_Out']})
            print(f'{metodo} - {nombre:<18} AUROC={m["AUROC"]:.4f}  FPR@95={m["FPR@95"]:.4f}  CCR@FPR5={m["CCR@FPR5"]:.4f}')
        registro[metodo] = {'quantile': CONFIGS[config]['quantile'], 'K': args.k, 'tiempos_por_conjunto': tiempos,
                            'acc_id_test': acc_id,
                            'regla_k': CONFIGS[config]['regla_k'], 'exploratorio': config == '100_kc',
                            'prop_p1_id_test': float(np.mean(idt['p'] == 1)),
                            'prop_p0_id_test': float(np.mean(idt['p'] == 0)),
                            'num_imagenes': {n: int(len(d['conf'])) for n, d in punt.items()}}
    pd.DataFrame(filas_p).to_csv(os.path.join(args.salida, 'resultados_principales.csv'), index=False)
    pd.DataFrame(filas_a).to_csv(os.path.join(args.salida, 'resultados_anexo.csv'), index=False)
    pd.DataFrame(filas_n).to_csv(os.path.join(args.salida, 'resultados_nct_por_clase.csv'), index=False)
    config = {'fase': 'B', 'metodo': 'stoodx_paper (stoodx_100, stoodx_875; stoodx_100_kc exploratorio)', 'entorno': sys.prefix,
              'activacion': 'source /mnt/homeGPU/slcerro/.uv/env.sh && source /mnt/homeGPU/slcerro/S-STOOD-X/.venv/bin/activate',
              'versiones': {**versiones(), 'gpu': sorted({t['gpu'] for r in registro.values()
                                                         for t in r['tiempos_por_conjunto'].values()})},
              'carpeta_features': os.path.join(FEATURES_ROOT, args.features_tag), 'semilla': SEMILLA,
              'hiperparametros': {'distancia': 'coseno', 'K': args.k, 'referencia': 'train ID de la clase predicha',
                                  'test': 'scipy.stats.mannwhitneyu(A, B, alternative="greater")',
                                  'puntuacion_principal': '-z (columna conf), monótonamente equivalente a p = norm.sf(z); '
                                                          'se guardan también p, U y z'},
              'notas': NOTAS, **registro}
    with open(os.path.join(args.salida, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=float)
    print(f'Guardado en {args.salida}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modo', choices=['validacion', 'validacion_extra', 'seleccion_k', 'reanalizar_validacion', 'analisis_validacion', 'completo', 'combinar'], required=True)
    parser.add_argument('--features_tag', default='completo')
    parser.add_argument('--salida', default=None)
    parser.add_argument('--config', choices=list(CONFIGS), default='100')
    parser.add_argument('--k', type=int, default=500)
    parser.add_argument('--conjuntos', nargs='+', default=A_PUNTUAR)
    args = parser.parse_args()
    if args.salida is None:
        args.salida = ('results_ood/stoodx_validacion' if 'validacion' in args.modo or args.modo == 'seleccion_k'
                       else f'results_ood/faseB_paper_{datetime.now():%Y-%m-%d}')
    args.salida = args.salida if os.path.isabs(args.salida) else os.path.join(RAIZ, args.salida)

    if args.modo == 'combinar':
        combinar(args)
        return
    if args.modo == 'reanalizar_validacion':
        reanalizar_validacion(args)
        return
    if args.modo == 'analisis_validacion':
        analisis_validacion(args)
        return
    if not torch.cuda.is_available():
        sys.exit('No hay GPU disponible: lanza este script con SLURM (sbatch script_stoodx_paper.sh)')
    torch.manual_seed(SEMILLA)
    print(f'GPU: {torch.cuda.get_device_name(0)}; {N_PROCESOS} procesos para el test', flush=True)
    with mp.get_context('fork').Pool(N_PROCESOS) as pool:
        if args.modo == 'validacion':
            validacion(args, pool)
        elif args.modo == 'validacion_extra':
            validacion_extra(args, pool)
        elif args.modo == 'seleccion_k':
            seleccion_k(args, pool)
        else:
            completo(args, pool)


if __name__ == '__main__':
    main()
