# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Métricas de detección OOD comunes a las fases A (OpenOOD), B (STOOD-X) y C (comparación).

Solo depende de numpy y scikit-learn, para usarse igual en fastai_env_ood y en el entorno
de S-STOOD-X.

Convención (Yang et al., 2024, sección 2.1 y nota al pie; OpenOOD, metrics.py:41-48):
    - la clase POSITIVA es OOD;
    - las puntuaciones `conf` son "más alto = más ID", así que para la curva ROC se usa -conf.

Métricas principales: AUROC, FPR@95 (FPR sobre ID cuando TPR sobre OOD = 0,95) y CCR@FPR5.
Anexo: AUPR-In (ID positiva) y AUPR-Out (OOD positiva).
"""

import numpy as np
from sklearn import metrics


# ---------------------------------------------------------------------------
# Métricas de detección
# ---------------------------------------------------------------------------

def _preparar(conf_id, conf_ood):
    conf_id = np.asarray(conf_id, dtype=np.float64).ravel()
    conf_ood = np.asarray(conf_ood, dtype=np.float64).ravel()
    conf = np.concatenate([conf_id, conf_ood])
    es_ood = np.concatenate([np.zeros(len(conf_id), dtype=int), np.ones(len(conf_ood), dtype=int)])
    return conf, es_ood


def auroc(conf_id, conf_ood):
    conf, es_ood = _preparar(conf_id, conf_ood)
    return metrics.roc_auc_score(es_ood, -conf)


def fpr_at_tpr(conf_id, conf_ood, tpr_objetivo=0.95):
    """Proporción de ID marcada como OOD en el primer umbral que detecta >= tpr_objetivo de las OOD
    (mismo criterio que openood.evaluators.metrics.auc_and_fpr_recall)."""
    conf, es_ood = _preparar(conf_id, conf_ood)
    fpr, tpr, _ = metrics.roc_curve(es_ood, -conf)
    return fpr[np.argmax(tpr >= tpr_objetivo)]


def aupr_in(conf_id, conf_ood):
    conf, es_ood = _preparar(conf_id, conf_ood)
    precision, recall, _ = metrics.precision_recall_curve(1 - es_ood, conf)
    return metrics.auc(recall, precision)


def aupr_out(conf_id, conf_ood):
    conf, es_ood = _preparar(conf_id, conf_ood)
    precision, recall, _ = metrics.precision_recall_curve(es_ood, -conf)
    return metrics.auc(recall, precision)


def umbral_fpr(conf_ood, fpr_objetivo=0.05):
    """Umbral tau tal que se acepta como ID (conf >= tau) el fpr_objetivo de las OOD:
    tau es la ceil(fpr_objetivo * N_OOD)-ésima puntuación OOD más alta. Con empates en tau
    puede aceptarse algo más; ccr_fpr devuelve la FPR real."""
    conf_ood = np.sort(np.asarray(conf_ood, dtype=np.float64).ravel())[::-1]
    n_aceptadas = max(1, int(np.ceil(fpr_objetivo * len(conf_ood))))
    return conf_ood[n_aceptadas - 1]


def ccr_fpr(conf_id, conf_ood, pred_id, label_id, fpr_objetivo=0.05, num_clases=None):
    """CCR@FPRx (Dhamija et al., 2018):
        tau: el fpr_objetivo de las OOD tiene conf >= tau
        CCR = #(ID con conf >= tau y bien clasificada) / #ID
    Devuelve (ccr, fpr_real, tau, ccr_por_clase). ccr_por_clase[c] es el recall de la clase c
    con el rechazo aplicado (NaN si la clase no aparece)."""
    conf_id = np.asarray(conf_id, dtype=np.float64).ravel()
    pred_id = np.asarray(pred_id).ravel()
    label_id = np.asarray(label_id).ravel()
    tau = umbral_fpr(conf_ood, fpr_objetivo)
    fpr_real = np.mean(np.asarray(conf_ood, dtype=np.float64) >= tau)
    aciertos = (conf_id >= tau) & (pred_id == label_id)
    ccr = aciertos.mean()

    if num_clases is None:
        num_clases = int(label_id.max()) + 1
    por_clase = np.full(num_clases, np.nan)
    for c in range(num_clases):
        mascara = label_id == c
        if mascara.any():
            por_clase[c] = aciertos[mascara].mean()
    return ccr, fpr_real, tau, por_clase


def exactitud(pred, label):
    return float(np.mean(np.asarray(pred) == np.asarray(label)))


def todas_las_metricas(conf_id, conf_ood, pred_id, label_id, num_clases=None):
    ccr, fpr_real, tau, por_clase = ccr_fpr(conf_id, conf_ood, pred_id, label_id, 0.05, num_clases)
    return {
        'AUROC': auroc(conf_id, conf_ood),
        'FPR@95': fpr_at_tpr(conf_id, conf_ood, 0.95),
        'CCR@FPR5': ccr,
        'FPR_real_CCR': fpr_real,
        'tau_CCR': tau,
        'AUPR_In': aupr_in(conf_id, conf_ood),
        'AUPR_Out': aupr_out(conf_id, conf_ood),
        'CCR@FPR5_por_clase': por_clase,
    }


# ---------------------------------------------------------------------------
# Bootstrap (fase C). Remuestreo estratificado: ID y OOD por separado, para que cada réplica
# conserve los tamaños de ambos conjuntos.
# ---------------------------------------------------------------------------

def _auroc_rapido(conf_id, conf_ood):
    """AUROC por rangos (Mann-Whitney, empates con rango medio); equivale a roc_auc_score."""
    n_id, n_ood = len(conf_id), len(conf_ood)
    # Rangos medios en empates (equivalente a scipy.stats.rankdata, solo con numpy)
    _, inverso, cuentas = np.unique(np.concatenate([conf_id, conf_ood]),
                                    return_inverse=True, return_counts=True)
    rangos = (np.cumsum(cuentas) - (cuentas - 1) / 2)[inverso]
    # OOD positiva con -conf: P(conf_ood < conf_id) + 0,5 P(empate)
    suma_id = rangos[:n_id].sum()
    return (suma_id - n_id * (n_id + 1) / 2) / (n_id * n_ood)


def indices_bootstrap(n_id, n_ood, n_remuestreos=1000, semilla=0):
    rng = np.random.default_rng(semilla)
    for _ in range(n_remuestreos):
        yield rng.integers(0, n_id, n_id), rng.integers(0, n_ood, n_ood)


def bootstrap_ic(conf_id, conf_ood, n_remuestreos=1000, semilla=0, alfa=0.05):
    """IC percentil al (1-alfa) de AUROC y FPR@95. Devuelve dict con las réplicas y los extremos."""
    conf_id = np.asarray(conf_id, dtype=np.float64)
    conf_ood = np.asarray(conf_ood, dtype=np.float64)
    aurocs, fprs = [], []
    for i_id, i_ood in indices_bootstrap(len(conf_id), len(conf_ood), n_remuestreos, semilla):
        aurocs.append(_auroc_rapido(conf_id[i_id], conf_ood[i_ood]))
        fprs.append(fpr_at_tpr(conf_id[i_id], conf_ood[i_ood]))
    aurocs, fprs = np.array(aurocs), np.array(fprs)
    q = [100 * alfa / 2, 100 * (1 - alfa / 2)]
    return {'AUROC_IC': np.percentile(aurocs, q), 'FPR@95_IC': np.percentile(fprs, q),
            'AUROC_reps': aurocs, 'FPR@95_reps': fprs}


def bootstrap_pareado_auroc(conf_id_a, conf_ood_a, conf_id_b, conf_ood_b,
                            n_remuestreos=1000, semilla=0, alfa=0.05):
    """Diferencia AUROC(a) - AUROC(b) con los MISMOS índices para ambos métodos (las puntuaciones
    deben estar en el mismo orden de imágenes). Devuelve diferencia observada, IC y p-valor
    bilateral bootstrap (proporción de réplicas al otro lado del 0, por 2)."""
    a_id, a_ood = np.asarray(conf_id_a, np.float64), np.asarray(conf_ood_a, np.float64)
    b_id, b_ood = np.asarray(conf_id_b, np.float64), np.asarray(conf_ood_b, np.float64)
    assert len(a_id) == len(b_id) and len(a_ood) == len(b_ood)
    observada = _auroc_rapido(a_id, a_ood) - _auroc_rapido(b_id, b_ood)
    difs = np.array([
        _auroc_rapido(a_id[i], a_ood[j]) - _auroc_rapido(b_id[i], b_ood[j])
        for i, j in indices_bootstrap(len(a_id), len(a_ood), n_remuestreos, semilla)
    ])
    ic = np.percentile(difs, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    p = min(1.0, 2 * min(np.mean(difs <= 0), np.mean(difs >= 0)))
    return {'diferencia': observada, 'IC': ic, 'p_bootstrap': p, 'reps': difs}
