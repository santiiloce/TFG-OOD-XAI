# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Configuración final de la fase 5, reutilizable en la fase 4.

Es `A` (H1, γ = (∞, ∞, ∞)): la regla de adopción no permitió sustituirla (results_crp/fase5/resumen.md).

Uso:
    from composite_final import preparar_modelo_final, composite_final
    m = preparar_modelo_final(model, device)     # copia; no toca `model`
    comp = composite_final()
    a = CondAttribution(m, no_param_grad=True)(x, [{'y': [c]}], comp, init_rel=semilla_clase(c), record_layer=[...])
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lrp_fase5 import INF, composite, preparar_modelo, semilla_clase   # noqa: E402,F401

H_FINAL = 'H1'
GAMMA_FINAL = (INF, INF, INF)    # G1 = stages.0 y stages.1, G2 = stages.2, G3 = stages.3; ∞ = z⁺


def preparar_modelo_final(model, device=None):
    return preparar_modelo(model, H_FINAL, device)


def composite_final():
    return composite(*GAMMA_FINAL, h=H_FINAL)
