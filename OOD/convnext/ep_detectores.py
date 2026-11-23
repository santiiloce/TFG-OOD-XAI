# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9b, paso 2 (CPU, zeus): detectores de características en M y SHE en C (sensibilidad).

Usa results_ood_convnext/scripts/f9_detectores.py sin modificarlo (mismos .yml, APS_mode=False, ajuste solo con ID).
M va por la rama genérica de f9_detectores (CachedC: [M | logits guardados]); SHE toma la clase predicha de los logits.
Salida: espacio_principal/espacio_M/ (knn, mds, rmds, she, she_cosine, she_euclidean, stoodx_100)
        espacio_principal/espacio_C_she/ (she, she_cosine, she_euclidean en C)
"""
import os, sys, time
from argparse import Namespace

F9 = '/mnt/homeGPU/slcerro/OpenOOD/results_ood_convnext'
EP = os.path.join(F9, 'espacio_principal')
os.environ['F9_FEATS'] = os.path.join(EP, 'features_M')
sys.path.insert(0, os.path.join(F9, 'scripts'))
import numpy as np, torch   # noqa: E402
import f9_detectores as fd  # noqa: E402

SHE = ['she', 'she_cosine', 'she_euclidean']
torch.set_num_threads(fd.NCPU); torch.manual_seed(0); np.random.seed(0)

t = time.time()
sal = os.path.join(EP, 'espacio_M'); os.makedirs(sal, exist_ok=True)
args = Namespace(espacio='M', metodos=['knn', 'mds', 'rmds'] + SHE, stoodx_conjuntos=fd.STOODX_DISCO)
fd.detectores(args, sal)
print(f'==== M, OpenOOD: {(time.time() - t) / 3600:.2f} h', flush=True); t = time.time()
fd.stoodx(args, sal)
print(f'==== M, STOOD-X: {(time.time() - t) / 3600:.2f} h', flush=True); t = time.time()

fd.FEATS = os.path.join(F9, 'features')           # C de la fase 9 (solo lectura)
sal = os.path.join(EP, 'espacio_C_she'); os.makedirs(sal, exist_ok=True)
fd.detectores(Namespace(espacio='C', metodos=SHE, stoodx_conjuntos=[]), sal)
print(f'==== C, SHE: {(time.time() - t) / 3600:.2f} h', flush=True)
