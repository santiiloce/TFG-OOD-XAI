# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Escribe results_corrupciones/parametros_corrupciones.csv leyendo los parámetros de Distortions directamente del
utils.py de Zhang et al. (commit 2b288c1) con ast, y los tamaños de las manchas de blots.npy. Nada copiado a mano."""
import ast
import csv
import os
import subprocess

import numpy as np

ZHANG = '/mnt/homeGPU/slcerro/externos/robustness_benchmark'
COMMIT = '2b288c11f944192bea7ffae912687d27b3ad220f'
SALIDA = '/mnt/homeGPU/slcerro/OpenOOD/results_corrupciones/parametros_corrupciones.csv'
NIVELES = [1, 3, 5]

assert subprocess.check_output(['git', '-C', ZHANG, 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
fuente = subprocess.check_output(['git', '-C', ZHANG, 'show', f'{COMMIT}:utils.py'], text=True)
clase = next(n for n in ast.parse(fuente).body if isinstance(n, ast.ClassDef) and n.name == 'Distortions')
metodos = {f.name: f for f in clase.body if isinstance(f, ast.FunctionDef)}


def lista(metodo, var, rama=None):
    """Valor de la asignación `var = [...][severity - 1]` (o `var = [...]`) en el método. Con `rama`, la que está
    dentro del `if x.size == rama`."""
    nodos = metodos[metodo].body
    if rama is not None:
        nodos = []
        for n in ast.walk(metodos[metodo]):
            if isinstance(n, ast.If):
                cmp = n.test
                if (isinstance(cmp, ast.Compare) and isinstance(cmp.comparators[0], ast.Tuple)
                        and ast.literal_eval(cmp.comparators[0]) == rama):
                    nodos = n.body
        assert nodos, (metodo, rama)
    for n in nodos:
        if isinstance(n, ast.Assign) and n.targets[0].id == var:
            v = n.value.value if isinstance(n.value, ast.Subscript) else n.value
            return ast.literal_eval(v)
    raise KeyError((metodo, var))


blots = np.load(os.path.join(ZHANG, 'blots.npy'), allow_pickle=True)[0]['224']
tam_mancha = [f"{dict(b)['blur_h']}x{dict(b)['blur_w']} px en ({dict(b)['positions'][0][0]}, {dict(b)['positions'][0][1]})"
              for b in blots]
hue_comentario = '# this implementation is unreliable' in fuente
assert hue_comentario

REC224 = ('el código publicado usa siempre los recursos de 96 px (clave "96"); se usan los de 224 px del mismo '
          'repositorio (clave "224"), sin modificar el código')
filas = [
    ('brillo', 'brightness', 'suma a V (HSV, [0,1])', lista('brightness', 'c'), ''),
    ('saturación', 'saturate', 'suma a S (HSV, [0,1])', lista('saturate', 'c'), ''),
    ('tono', 'hue', 'suma a H (HSV, con vuelta)', lista('hue', 'c'),
     'el código la marca como "this implementation is unreliable and needs debug"; se usa tal cual'),
    ('marcas', 'marking_blur', 'mancha fija de blots.npy[0]["224"] (alto x ancho, posición x, y de la esquina)', tam_mancha,
     REC224 + '; no aleatoria (misma mancha y posición en todas las imágenes)'),
    ('burbujas', 'bubble_blur', 'proporción de mezcla con la burbuja fija de bubbles.npy[0]["224"]',
     lista('bubble_blur', 'ratios'), REC224 + '; no aleatoria'),
    ('JPEG', 'jpeg_compression', 'calidad JPEG', lista('jpeg_compression', 'c'), ''),
    ('pixelado', 'pixelate', 'factor de reducción (BOX, ida y vuelta)', lista('pixelate', 'c'), ''),
    ('desenfoque', 'defocus_blur', '(radio del disco, sigma del antialias), rama 224 px',
     lista('defocus_blur', 'c', (224, 224)), ''),
    ('movimiento', 'motion_blur', '(radio, sigma) de ImageMagick, rama 224 px; ángulo U(-45, 45)',
     lista('motion_blur', 'c', (224, 224)),
     'única aleatoria: np.random.seed(sha256(fichero|corrupción|nivel)) antes de cada llamada'),
]
fmt = lambda v: f'({v[0]}; {v[1]})' if isinstance(v, tuple) else str(v)
with open(SALIDA, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['corrupcion', 'codigo', 'parametro', 'nivel_1', 'nivel_3', 'nivel_5', 'nota'])
    for nombre, cod, par, vals, nota in filas:
        assert len(vals) == 5, (cod, vals)
        w.writerow([nombre, cod, par] + [fmt(vals[n - 1]) for n in NIVELES] + [nota])
print(open(SALIDA).read())
