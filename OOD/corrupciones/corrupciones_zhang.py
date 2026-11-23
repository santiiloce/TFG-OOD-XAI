# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Corrupciones de Zhang et al. (MICCAI 2022) llamando a su código sin modificarlo
(/mnt/homeGPU/slcerro/externos/robustness_benchmark, utils.Distortions; solo lectura).

Adaptaciones (ver decisiones.md):
- utils.py importa visdom y wandb (solo para registro); se sustituyen por módulos vacíos para poder importarlo.
- Distortions.__init__ lee blots.npy y bubbles.npy con ruta relativa; se cargan con ruta absoluta.
- marking_blur y bubble_blur usan siempre los recursos de 96 px (clave '96'). Nuestras imágenes son de 224 px y el
  repositorio trae recursos de 224 px (clave '224'): se colocan bajo la clave '96' para que el código original use
  los de 224 sin cambiar una línea.
- La única corrupción aleatoria es motion_blur (ángulo np.random.uniform(-45, 45)); antes de cada llamada se fija
  np.random.seed(semilla(nombre de fichero, corrupción, nivel)).
"""
import hashlib
import os
import sys
import types

import numpy as np
from PIL import Image

ZHANG = '/mnt/homeGPU/slcerro/externos/robustness_benchmark'
CORRUPCIONES = ['brightness', 'saturate', 'hue', 'marking_blur', 'bubble_blur',
                'jpeg_compression', 'pixelate', 'defocus_blur', 'motion_blur']
NOMBRES = {'brightness': 'brillo', 'saturate': 'saturación', 'hue': 'tono', 'marking_blur': 'marcas',
           'bubble_blur': 'burbujas', 'jpeg_compression': 'JPEG', 'pixelate': 'pixelado',
           'defocus_blur': 'desenfoque', 'motion_blur': 'movimiento'}
NIVELES = [1, 3, 5]
ALEATORIAS = {'motion_blur'}

for _m in ('visdom', 'wandb'):
    if _m not in sys.modules:
        stub = types.ModuleType(_m)
        stub.Visdom = object
        sys.modules[_m] = stub
sys.path.insert(0, ZHANG)
import utils as _zhang  # noqa: E402
sys.path.pop(0)


def _distorsiones():
    d = _zhang.Distortions.__new__(_zhang.Distortions)
    blots = np.load(os.path.join(ZHANG, 'blots.npy'), allow_pickle=True)
    bubbles = np.load(os.path.join(ZHANG, 'bubbles.npy'), allow_pickle=True)
    d.blots = [{'96': blots[0]['224']}]
    d.bubbles = [{'96': bubbles[0]['224']}]
    return d


DIST = _distorsiones()


def semilla(ruta, corrupcion, nivel):
    clave = f'{os.path.basename(ruta)}|{corrupcion}|{nivel}'.encode()
    return int.from_bytes(hashlib.sha256(clave).digest()[:4], 'little')


def corromper(ruta, corrupcion, nivel, img=None):
    """Imagen original (uint8, tamaño nativo) -> imagen corrupta como array uint8 (H, W, 3). `img`: la imagen ya
    leída (PIL RGB) para no releerla; la semilla depende solo del nombre de `ruta`."""
    img = Image.open(ruta).convert('RGB') if img is None else img.copy()
    if nivel == 0:
        return np.asarray(img, dtype=np.uint8)
    np.random.seed(semilla(ruta, corrupcion, nivel))
    out = getattr(DIST, corrupcion)(img, nivel)
    out = np.asarray(out.convert('RGB') if isinstance(out, Image.Image) else out)
    assert out.shape == (img.size[1], img.size[0], 3), (corrupcion, nivel, out.shape)
    return out.astype(np.uint8)
