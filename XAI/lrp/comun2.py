# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Utilidades de la fase 2. Usa zennit y zennit-crp del entorno crp_env (no results_crp/pydeps).
lrp_convnext_v2.py se importa de results_crp/scripts, sin modificarlo."""
import os
import sys
from pathlib import Path

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
FASE2 = os.path.join(RESULTS, 'fase2')
sys.path.insert(0, os.path.join(RESULTS, 'scripts'))

import numpy as np
import torch
from fastai.vision.all import load_learner

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
EJEC = '/mnt/homeGPU/slcerro/tfg/executions/def1'
PKL = {'convnext': f'{EJEC}/param2_2026-03-24_16-19-23/convnext_small_model.pkl',
       'resnet50': f'{EJEC}/param1_2026-03-24_16-19-19/resnet50_model.pkl'}
# Variante A de results_crp (ConvNeXt)
BASE_A = dict(ln='pass', canonizer=True, gelu_pass=True, bn_pass=True, sin_sesgo=True, mlp='zplus', suma='abs')


def cargar(nombre, device):
    learn = load_learner(PKL[nombre], cpu=True)     # cpu=True: en fp32
    return learn, learn.model.eval().to(device)


def seleccion(n_por_clase=2, semilla=0):
    d = np.load(os.path.join(RAIZ, 'features/completo/id_test.npz'))
    rng = np.random.default_rng(semilla)
    idx = []
    for c in range(14):
        idx.extend(sorted(rng.choice(np.where(d['label'] == c)[0], n_por_clase, replace=False).tolist()))
    idx = np.array(idx)
    return idx, [str(p) for p in d['paths'][idx]], d['label'][idx]


def preprocesar(learn, rutas, device):
    dl = learn.dls.test_dl([Path(r) for r in rutas], bs=len(rutas), num_workers=0)
    return dl.one_batch()[0].as_subclass(torch.Tensor).to(device)


def semilla_clase(c):
    def seed(p):
        s = torch.zeros_like(p)
        s[:, c] = p[:, c]
        return s
    return seed
