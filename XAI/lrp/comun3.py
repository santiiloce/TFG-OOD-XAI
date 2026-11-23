# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Utilidades de la fase 3. Importa, sin modificarlos, los módulos de results_crp/scripts y results_crp/fase2/scripts."""
import os
import sys

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
FASE3 = os.path.join(RESULTS, 'fase3')
sys.path.insert(0, os.path.join(RESULTS, 'fase2', 'scripts'))

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
from comun2 import BASE_A, cargar, preprocesar, seleccion, semilla_clase  # noqa: E402,F401


def fusionar_bn(m):
    """Fusiona cada BatchNorm1d de la cabeza en la Linear siguiente (en evaluación BN es afín y Dropout es la identidad):
    W' = W diag(gamma/sigma), b' = b + W (beta - gamma mu / sigma), sigma = sqrt(var + eps). Modifica m: pasar una copia."""
    cabeza = m[1]
    for i_bn, i_lin in [(2, 4), (6, 8)]:
        bn, lin = cabeza[i_bn], cabeza[i_lin]
        assert isinstance(bn, torch.nn.BatchNorm1d) and isinstance(lin, torch.nn.Linear)
        assert not bn.training
        with torch.no_grad():
            sigma = (bn.running_var + bn.eps).sqrt()
            escala = bn.weight / sigma
            nueva = torch.nn.Linear(lin.in_features, lin.out_features, bias=True).to(lin.weight.device)
            nueva.weight.copy_(lin.weight * escala[None, :])
            desplaz = bn.bias - bn.weight * bn.running_mean / sigma
            nueva.bias.copy_((lin.bias if lin.bias is not None else 0) + lin.weight @ desplaz)
        nueva.eval()
        cabeza[i_lin] = nueva
        cabeza[i_bn] = torch.nn.Identity()
    return m


def boot_ic(v, estadistico=np.mean, n=10000, semilla=0):
    """IC 95 % percentil por bootstrap de imágenes."""
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    i = np.random.default_rng(semilla).integers(0, len(v), (n, len(v)))
    r = estadistico(v[i], axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))
