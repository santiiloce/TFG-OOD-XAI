# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Utilidades de la fase 4. `A` viene de fase5/scripts/composite_final.py; G×E es CRP sin composite, con relevancia = activación × gradiente."""
import os
import sys
from pathlib import Path

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
FASE4 = os.path.join(RESULTS, 'fase4')
sys.path.insert(0, os.path.join(RESULTS, 'fase5', 'scripts'))
sys.path.insert(0, os.path.join(RESULTS, 'fase2', 'scripts'))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from composite_final import composite_final, preparar_modelo_final    # noqa: E402,F401
from crp.attribution import CondAttribution                           # noqa: E402
from crp.concepts import ChannelConcept                               # noqa: E402,F401
from crp.visualization import FeatureVisualization                    # noqa: E402
from fastai.vision.all import PILImage                                # noqa: E402
from lrp_fase5 import FASE5, aleatorizado, boot_ic, cargar, fusionar_bn, preprocesar, rho, seleccion, semilla_clase   # noqa: E402,F401

CAPA = '0.model.stages.3.blocks.2'
RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
CLASES = ['Adenocarcinoma_ductal', 'Carcinoma_intraductal', 'Estroma', 'Glandulas_no_neoplasicas', 'Gleason_3', 'Gleason_4_cribiformes', 'Gleason_4_fusionadas', 'Gleason_4_glomeruloide',
          'Gleason_4_pobremente_formadas', 'Gleason_5_comedonecrosis', 'Gleason_5_no_glandulas', 'Inflamatorio', 'No_Cancer', 'PIN_de_alto_grado']


def semilla_uno(p):
    """Semilla 1 en toda la salida; CRP la enmascara a la clase condicionada: el mapa es ∂f_c/∂x · x, como en las fases 3 y 5."""
    return torch.ones_like(p)


class CondGxE(CondAttribution):
    """CRP sin composite. Mapa: suma por color de gradiente × entrada (CRP por defecto devolvería solo el gradiente). Relevancia en una capa: activación × gradiente."""

    def heatmap_modifier(self, data, on_device=None):
        h = (data.grad.detach() * data.detach()).sum(1)
        return h.to(on_device) if on_device else h

    def _collect_hook_activation_relevance(self, layer_out, on_device=None):
        a, r = super()._collect_hook_activation_relevance(layer_out, on_device)
        return a, {k: r[k] * a[k] for k in r}


class FVGxE(FeatureVisualization):
    """FeatureVisualization para G×E: el gancho de CRP entrega el gradiente; aquí se multiplica por la activación de la misma capa. ActMax no se guarda: es el de `A`."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._act = {}

    def analyze_activation(self, act, layer_name, concept, data_indices, targets):
        self._act[layer_name] = act

    def analyze_relevance(self, rel, layer_name, concept, data_indices, targets):
        super().analyze_relevance(rel * self._act[layer_name], layer_name, concept, data_indices, targets)


class DatasetRutas(torch.utils.data.Dataset):
    """Devuelve (imagen uint8 tras los after_item de fastai, etiqueta). preprocesa() aplica la conversión y normalización de after_batch."""

    def __init__(self, learn, rutas, etiquetas):
        self.rutas, self.etiquetas, self.ai = list(rutas), np.asarray(etiquetas), learn.dls.after_item

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        t = self.ai(PILImage.create(Path(self.rutas[i])))
        return torch.as_tensor(np.asarray(t)).as_subclass(torch.Tensor), int(self.etiquetas[i])


def hacer_preprocesa(learn, device):
    nrm = [t for t in learn.dls.after_batch.fs if type(t).__name__ == 'Normalize'][0]
    m, s = nrm.mean.as_subclass(torch.Tensor).to(device), nrm.std.as_subclass(torch.Tensor).to(device)
    return lambda b: (b.float() / 255.0 - m) / s


def metodo(nombre, model, device):
    """Devuelve (modelo, atribuidor, composite, kw de semilla) de 'A' o 'GxE'."""
    if nombre == 'A':
        m = preparar_modelo_final(model, device)
        return m, CondAttribution(m, no_param_grad=True), composite_final()
    import copy
    m = copy.deepcopy(model).to(device).eval()
    return m, CondGxE(m, no_param_grad=True), None


def mapa(at, comp, nombre, xi, c, canales=None, capas=()):
    """Mapa y resultado CRP. Condición: clase c (+ canales de CAPA). Semilla: logit de c en `A`; 1 en G×E."""
    cond = {'y': [c]}
    if canales is not None:
        cond[CAPA] = list(canales)
    ini = semilla_clase(c) if nombre == 'A' else semilla_uno
    a = at(xi.clone().requires_grad_(True), [cond], comp, init_rel=ini, record_layer=list(capas))
    return a, a.heatmap[0].detach().cpu().numpy()


# ---- Figuras
def cmap_rel():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('rel', ['#2a78d6', '#f0efec', '#e34948'])


def estilo():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'text.color': '#0b0b0b', 'axes.labelcolor': '#52514e', 'axes.edgecolor': '#52514e', 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb'})
    return plt


def pinta_mapa(ax, h, cm):
    v = float(np.percentile(np.abs(h), 99.5)) or 1.0
    ax.imshow(h, cmap=cm, vmin=-v, vmax=v, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def pinta_img(ax, t):
    ax.imshow(np.asarray(t).transpose(1, 2, 0))
    ax.set_xticks([])
    ax.set_yticks([])


ABREV = {c: c.replace('Adenocarcinoma_ductal', 'Adeno. ductal').replace('Carcinoma_intraductal', 'Ca. intraductal').replace('Glandulas_no_neoplasicas', 'Gl. no neoplás.').replace('Gleason_4_cribiformes', 'G4 cribif.').replace('Gleason_4_fusionadas', 'G4 fusion.')
         .replace('Gleason_4_glomeruloide', 'G4 glomer.').replace('Gleason_4_pobremente_formadas', 'G4 pobre.').replace('Gleason_5_comedonecrosis', 'G5 comedo.').replace('Gleason_5_no_glandulas', 'G5 no gl.').replace('Gleason_3', 'G3').replace('PIN_de_alto_grado', 'PIN alto').replace('No_Cancer', 'No cáncer') for c in CLASES}
