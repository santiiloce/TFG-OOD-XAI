# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Paso 2 (salidas): 4 conceptos por clase (media de la relevancia condicional normalizada, en todo el test de la clase real), 8 referencias RelMax, figuras, RelMax frente a ActMax y solapamiento."""
import os
import time
from pathlib import Path

from comun4 import ABREV, CAPA, CLASES, FASE4, ChannelConcept, DatasetRutas, cargar, cmap_rel, estilo, mapa, metodo, np, pinta_img, pinta_mapa, preprocesar, torch

import pandas as pd

device = torch.device('cuda')
learn, model = cargar('convnext', device)
S = os.path.join(FASE4, 'relmax')
te, tr = np.load(os.path.join(FASE4, 'features', 'convnext_test.npz')), np.load(os.path.join(FASE4, 'features', 'convnext_train.npz'))
rte, lte = [str(p) for p in te['paths']], te['label']
rtr, ltr = [str(p) for p in tr['paths']], tr['label']
ds = DatasetRutas(learn, rtr, ltr)
plt = estilo()
cm = cmap_rel()
filas, refs, conceptos = [], [], {}
RM = {n: np.load(os.path.join(S, n, 'RelMax_sum_normed', f'{CAPA}_data.npy')) for n in ('A', 'GxE')}
RR = {n: np.load(os.path.join(S, n, 'RelMax_sum_normed', f'{CAPA}_rel.npy')) for n in ('A', 'GxE')}
AM = np.load(os.path.join(S, 'A', 'ActMax_sum_normed', f'{CAPA}_data.npy'))
print('formas', RM['A'].shape, AM.shape, flush=True)
modelos = {}
for nom, lote in [('A', 16), ('GxE', 32)]:
    m, at, comp = metodo(nom, model, device)
    modelos[nom] = (m, at, comp)
    t = time.perf_counter()
    crudo, norm, npos = np.zeros((14, 768)), np.zeros((14, 768)), np.zeros((14, 768))
    n_c = np.bincount(lte, minlength=14)
    dl = learn.dls.test_dl([Path(r) for r in rte], bs=lote, num_workers=4)
    j = 0
    for b in dl:
        x = b[0].as_subclass(torch.Tensor).to(device).clone().requires_grad_(True)
        ys = lte[j:j + len(x)]
        a = at(x, [{'y': [int(y)]} for y in ys], comp, record_layer=[CAPA])       # sin init_rel: R = f_c(x), como en RelMax
        r = ChannelConcept().attribute(a.relevances[CAPA].detach(), abs_norm=False).cpu().numpy()
        rn = r / (np.abs(r).sum(1, keepdims=True) + 1e-10)
        np.add.at(crudo, ys, r)
        np.add.at(norm, ys, rn)
        np.add.at(npos, ys, r > 0)
        j += len(x)
    assert j == len(rte)
    crudo, norm, npos = crudo / n_c[:, None], norm / n_c[:, None], npos / n_c[:, None]
    np.savez(os.path.join(S, f'relevancia_test_{nom}.npz'), crudo=crudo, norm=norm, frac_pos=npos, n=n_c)
    print(nom, 'test', round(time.perf_counter() - t), 's', flush=True)
    top = np.argsort(-norm, 1)[:, :4]
    conceptos[nom] = top
    for c in range(14):
        for p, k in enumerate(top[c]):
            filas.append(dict(metodo=nom, clase=CLASES[c], n_test=int(n_c[c]), puesto=p + 1, canal=int(k), relevancia_media_norm=norm[c, k], relevancia_media_cruda=crudo[c, k], frac_imagenes_positiva=npos[c, k]))
    pd.DataFrame(filas).to_csv(os.path.join(S, 'conceptos.csv'), index=False)
    # Figuras concepto × referencias
    for c in range(14):
        fig, ax = plt.subplots(8, 8, figsize=(13, 13.6))
        for f, k in enumerate(top[c]):
            for q, i in enumerate(RM[nom][:8, k]):
                i = int(i)
                xi = preprocesar(learn, [rtr[i]], device)
                h = mapa(at, comp, nom, xi, int(ltr[i]), canales=[int(k)])[1]
                pinta_img(ax[2 * f, q], ds[i][0])
                pinta_mapa(ax[2 * f + 1, q], h, cm)
                ax[2 * f, q].set_title(f'{ABREV[CLASES[ltr[i]]]}  {RR[nom][q, k]:.3f}', fontsize=7, loc='left')
                refs.append(dict(metodo=nom, clase_figura=CLASES[c], puesto_concepto=f + 1, canal=int(k), puesto_ref=q + 1, idx_train=i, ruta=rtr[i], clase_ref=CLASES[ltr[i]], rel_norm=float(RR[nom][q, k])))
            ax[2 * f, 0].set_ylabel(f'canal {k}\nmedia {norm[c, k]:.3f}', fontsize=8)
        fig.suptitle(f'{CLASES[c].replace("_", " ")}: {"A" if nom == "A" else "G×E"}, 4 conceptos más relevantes en test y 8 referencias RelMax (arriba imagen, abajo mapa del canal)', x=.02, ha='left', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .975))
        os.makedirs(os.path.join(S, 'figuras', nom), exist_ok=True)
        fig.savefig(os.path.join(S, 'figuras', nom, f'{c:02d}_{CLASES[c]}.png'), dpi=110)
        plt.close(fig)
    pd.DataFrame(refs).to_csv(os.path.join(S, 'referencias.csv'), index=False)
    print(nom, 'figuras', round(time.perf_counter() - t), 's', flush=True)
sol = pd.DataFrame([dict(clase=CLASES[c], canales_A=sorted(conceptos['A'][c].tolist()), canales_GxE=sorted(conceptos['GxE'][c].tolist()), coinciden=len(set(conceptos['A'][c]) & set(conceptos['GxE'][c]))) for c in range(14)])
sol.to_csv(os.path.join(S, 'solapamiento.csv'), index=False)
print(sol.to_string())
# Repetición de conceptos entre clases
rep = {n: dict(canales_distintos=int(len(np.unique(conceptos[n]))), max_clases_por_canal=int(np.unique(conceptos[n], return_counts=True)[1].max())) for n in conceptos}
print(rep)
# RelMax frente a ActMax: 3 canales más repetidos en el conjunto de los dos métodos (empates, índice menor)
u, f = np.unique(np.concatenate([conceptos['A'].ravel(), conceptos['GxE'].ravel()]), return_counts=True)
tres = u[np.argsort(-f, kind='stable')[:3]].tolist()
fig, ax = plt.subplots(9, 8, figsize=(12, 13.4))
filas_cmp = []
for g, k in enumerate(tres):
    for f_, (et, M) in enumerate([('RelMax A', RM['A']), ('RelMax G×E', RM['GxE']), ('ActMax', AM)]):
        for q, i in enumerate(M[:8, k]):
            pinta_img(ax[3 * g + f_, q], ds[int(i)][0])
            ax[3 * g + f_, q].set_title(ABREV[CLASES[ltr[int(i)]]], fontsize=7, loc='left')
        ax[3 * g + f_, 0].set_ylabel(f'canal {k}\n{et}', fontsize=8)
    filas_cmp.append(dict(canal=k, A_y_ActMax=len(set(RM['A'][:8, k]) & set(AM[:8, k])), GxE_y_ActMax=len(set(RM['GxE'][:8, k]) & set(AM[:8, k])), A_y_GxE=len(set(RM['A'][:8, k]) & set(RM['GxE'][:8, k]))))
fig.tight_layout()
fig.savefig(os.path.join(S, 'figuras', 'relmax_vs_actmax.png'), dpi=110)
plt.close(fig)
pd.DataFrame(filas_cmp).to_csv(os.path.join(S, 'relmax_vs_actmax.csv'), index=False)
print(pd.DataFrame(filas_cmp).to_string())
tot = [dict(A_y_ActMax=len(set(RM['A'][:8, k]) & set(AM[:8, k])), GxE_y_ActMax=len(set(RM['GxE'][:8, k]) & set(AM[:8, k])), A_y_GxE=len(set(RM['A'][:8, k]) & set(RM['GxE'][:8, k]))) for k in range(768)]
print('Coincidencias en las 8 referencias, mediana en los 768 canales', pd.DataFrame(tot).median().to_dict(), flush=True)
pd.DataFrame(tot).to_csv(os.path.join(S, 'coincidencia_referencias_768.csv'), index=False)
print('CONCEPTOS_TERMINADOS', flush=True)
