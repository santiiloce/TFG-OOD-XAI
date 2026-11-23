# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 3.1: vector de características de ConvNeXt (768, media espacial de CAPA), logits y predicción. Val y NCT solo sirven para fijar y aplicar el umbral de aceptación."""
import os
import time
from pathlib import Path

from comun4 import CAPA, FASE4, RAIZ, cargar, np, torch

device = torch.device('cuda')
learn, model = cargar('convnext', device)
S = os.path.join(FASE4, 'features')
capa = dict(model.named_modules())[CAPA]
g = {}
capa.register_forward_hook(lambda _m, _a, s: g.__setitem__('s', s))
for nom, ori in [('train', 'id_train'), ('test', 'id_test'), ('val', 'id_val'), ('nct_tum', 'near_nct_tum'), ('nct_norm', 'near_nct_norm')]:
    f = os.path.join(S, f'convnext_{nom}.npz')
    if os.path.exists(f):
        continue
    d = np.load(os.path.join(RAIZ, f'features/completo/{ori}.npz'))
    rutas = [str(p) for p in d['paths']]
    dl = learn.dls.test_dl([Path(r) for r in rutas], bs=128, num_workers=4)     # conserva el orden
    t, F, L = time.perf_counter(), [], []
    with torch.no_grad():
        for b in dl:
            y = model(b[0].as_subclass(torch.Tensor).to(device))
            F.append(g['s'].mean((2, 3)).cpu().numpy())
            L.append(y.cpu().numpy())
    F, L = np.concatenate(F), np.concatenate(L)
    assert len(F) == len(rutas)
    np.savez(f, feats=F.astype(np.float32), logits=L.astype(np.float32), pred=L.argmax(1), label=d['label'], paths=d['paths'])
    print(nom, F.shape, 'acierto', float((L.argmax(1) == d['label']).mean()), round(time.perf_counter() - t), 's', flush=True)
print('CARACTERISTICAS_TERMINADAS', flush=True)
