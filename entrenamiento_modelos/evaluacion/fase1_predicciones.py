# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 1 — Predicciones de los 5 clasificadores sobre el mismo test ID (12.618 imágenes).

Cada modelo se carga desde su learner exportado (learn.export -> *_model.pkl) y se evalúa con
learn.dls.test_dl(...), es decir, con el preprocesado de validación de su propio learner: Resize con
method='squish' a su tamaño de entrenamiento (224, salvo convnext_small, entrenado a 256) y
Normalize(imagenet_stats), sin aumentos (aug_transforms solo actúa en entrenamiento). Es el mismo camino que usó
clasificador.py para medir la exactitud de test.

Entorno: fastai_env_ood. GPU: sbatch script_fase1.sh
Salida: results_fase1/predicciones_<modelo>.npz (rutas, etiqueta, predicción, probabilidades), en el orden de
features/completo/id_test.npz (rutas ordenadas, igual que en la fase A).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastai.vision.all import load_learner

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
EJECUCIONES = '/mnt/homeGPU/slcerro/tfg/executions/def1'
SALIDA = os.path.join(RAIZ, 'results_fase1')
sys.path.insert(0, EJECUCIONES)          # por si el learner exportado referencia módulos del entrenamiento

MODELOS = {
    'swin_small_patch4_window7_224': ('param4_2026-03-24_21-55-59', 224),
    'convnext_small': ('param2_2026-03-24_16-19-23', 256),
    'tf_efficientnetv2_s': ('param3_2026-03-24_16-19-23', 224),
    'vit_base_patch16_224': ('param0_2026-03-24_16-18-54', 224),
    'resnet50': ('param1_2026-03-24_16-19-19', 224),
}


def main():
    if not torch.cuda.is_available():
        sys.exit('No hay GPU disponible: lanza este script con SLURM (sbatch script_fase1.sh)')
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    os.makedirs(SALIDA, exist_ok=True)
    ref = np.load(os.path.join(RAIZ, 'features/completo/id_test.npz'))
    rutas = [str(p) for p in ref['paths']]
    clases = json.load(open(os.path.join(RAIZ, 'features/completo/info.json')))['clases_id']
    registro = {}
    for modelo, (carpeta, tam) in MODELOS.items():
        inicio = time.perf_counter()
        learn = load_learner(os.path.join(EJECUCIONES, carpeta, f'{modelo}_model.pkl'), cpu=False)
        vocab = list(learn.dls.vocab)
        assert vocab == clases, f'{modelo}: el vocabulario no coincide con las clases ID'
        # Preprocesado de validación del learner (comprobación del tamaño)
        tfms = [repr(t) for t in learn.dls.after_item.fs]
        # Como en clasificador.py: objetos Path (con str, el pipeline de etiquetas no aplica parent_label)
        dl = learn.dls.test_dl([Path(r) for r in rutas], with_labels=True, bs=64, num_workers=8)
        probs, targs = learn.get_preds(dl=dl)
        probs, targs = probs.float().numpy(), targs.numpy()
        assert np.array_equal(targs, ref['label']), f'{modelo}: etiquetas distintas a las de la fase A'
        pred = probs.argmax(1)
        acc = float(np.mean(pred == targs))
        np.savez(os.path.join(SALIDA, f'predicciones_{modelo}.npz'), paths=np.array(rutas, dtype=str),
                 label=targs.astype(np.int64), pred=pred.astype(np.int64), probs=probs.astype(np.float32))
        registro[modelo] = {'carpeta': carpeta, 'tamano_entrenamiento': tam, 'transformaciones_item': tfms,
                            'exactitud': acc, 'tiempo_s': time.perf_counter() - inicio}
        print(f'{modelo}: exactitud {acc:.4f}  ({tfms})  {time.perf_counter() - inicio:.0f} s', flush=True)
        del learn
        torch.cuda.empty_cache()
    with open(os.path.join(SALIDA, 'predicciones_info.json'), 'w') as f:
        json.dump({'gpu': torch.cuda.get_device_name(0), 'modelos': registro,
                   'preprocesado': 'learn.dls.test_dl: Resize(method=squish) al tamaño de entrenamiento de cada '
                                   'modelo + Normalize(imagenet_stats), sin aumentos; convnext_small a 256 y el '
                                   'resto a 224.'}, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
