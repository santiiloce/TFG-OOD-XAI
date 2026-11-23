# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Comprobaciones finales (CPU, con lo ya calculado): (1) pureza del atlas de EG con la semilla 1; (2) mapas de EG por parcelas de 16 px."""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
F10 = os.path.join(R, 'fase10_eg')
O = os.path.join(F10, 'paso4', 'muestra', 'resultados')
S = os.path.join(F10, 'comprobaciones_finales')
os.makedirs(S, exist_ok=True)
ltr = np.load(os.path.join(R, 'fase4', 'features', 'convnext_train.npz'))['label']
rng = np.random.default_rng(0)

# ---- (1) Pureza con la semilla 1
cand = {int(k): np.array(v) for k, v in json.load(open(os.path.join(O, 'candidatas50.json'))).items()}
z = np.load(os.path.join(O, 'C_semilla1_parte0.npz'))
c1 = {int(i): c for i, c in zip(z['ids'], z['C'])}
con = pd.read_csv(os.path.join(O, 'conceptos.csv'))
ref = pd.read_csv(os.path.join(O, 'referencias_todas.csv'))
CL = sorted(con.clase.unique())
filas = []
for r in con.itertuples():
    c = CL.index(r.clase)
    if r.metodo.startswith('EG'):
        ids = cand[r.canal]
        n1 = np.array([c1[i][r.canal] / np.abs(c1[i]).sum() for i in ids])
        top1 = ids[np.argsort(-n1, kind='stable')[:8]]
        top0 = ref[(ref.metodo == 'EG') & (ref.clase_figura == r.clase) & (ref.canal == r.canal)].idx_train.values
        assert set(top0) <= set(ids) and len(top0) == 8
        conj = {'EG semilla 0': top0, 'EG semilla 1': top1}
    else:
        conj = {'A': ref[(ref.metodo == 'A') & (ref.clase_figura == r.clase) & (ref.canal == r.canal)].idx_train.values}
    for m, t in conj.items():
        cls = ltr[t]
        assert len(t) == 8
        filas.append(dict(metodo=m, clase=r.clase, canal=r.canal, pureza_concepto=(cls == c).mean(), pureza_mayoritaria=np.bincount(cls, minlength=14).max() / 8))
p = pd.DataFrame(filas)
p.to_csv(os.path.join(S, 'pureza_semillas_por_concepto.csv'), index=False)
pc = p.groupby(['metodo', 'clase'])[['pureza_concepto', 'pureza_mayoritaria']].mean()


def dif(a, b, col, B=10_000):
    """Diferencia de medias (a − b) con IC bootstrap remuestreando las 14 clases."""
    x, y = pc.loc[a][col].reindex(CL).values, pc.loc[b][col].reindex(CL).values
    ii = rng.integers(0, 14, (B, 14))
    d = (x[ii] - y[ii]).mean(1)
    return float((x - y).mean()), float(np.quantile(d, .025)), float(np.quantile(d, .975))


t1 = []
for col in ('pureza_concepto', 'pureza_mayoritaria'):
    for m in ('EG semilla 0', 'EG semilla 1', 'A'):
        v = p[p.metodo == m][col]
        t1.append(dict(medida=col, fila=m, mediana=v.median(), media=v.mean(), dif=np.nan, IC_inf=np.nan, IC_sup=np.nan))
    for a, b in (('EG semilla 1', 'A'), ('EG semilla 0', 'A'), ('EG semilla 1', 'EG semilla 0')):
        d, lo, hi = dif(a, b, col)
        t1.append(dict(medida=col, fila=f'{a} − {b}', mediana=np.nan, media=np.nan, dif=d, IC_inf=lo, IC_sup=hi))
t1 = pd.DataFrame(t1)
print(t1.to_string())


# ---- (2) Parcelas de 16 px
def parcela(h):
    return h.reshape(16, 16, 16, 16).sum((1, 3))


def sp(a, b):
    return float(spearmanr(a.ravel(), b.ravel()).correlation)


mp = np.load(os.path.join(F10, 'paso3', 'mapas.npz'))
grupos = {}
for k in mp.files:
    crit, j, f, can = k.split('|')
    if crit in ('original', 'z'):
        grupos.setdefault((crit, int(j), int(f)), []).append(mp[k])
div = []
for (crit, j, f), hs in grupos.items():
    assert len(hs) == 4
    pares = [(u, v) for u in range(4) for v in range(u + 1, 4)]
    div.append(dict(criterio=crit, caso=j, fila=f, pixel=np.mean([abs(sp(hs[u], hs[v])) for u, v in pares]),
                    parcela=np.mean([abs(sp(parcela(hs[u]), parcela(hs[v]))) for u, v in pares])))
div = pd.DataFrame(div)
e0, e1 = np.load(os.path.join(F10, 'paso0', 'eg_semilla0.npz'))['mapas'], np.load(os.path.join(F10, 'paso0', 'eg_semilla1.npz'))['mapas']
assert e0.dtype == e1.dtype == np.float32
sue = pd.DataFrame([dict(imagen=i, pixel=abs(sp(e0[i], e1[i])), parcela=abs(sp(parcela(e0[i]), parcela(e1[i])))) for i in range(len(e0))])
div.to_csv(os.path.join(S, 'diversidad_parcelas.csv'), index=False)
sue.to_csv(os.path.join(S, 'suelo_semillas_EG_sin_condicionar.csv'), index=False)


def bmed(v, B=10_000):
    v = np.asarray(v)
    m = np.median(v[rng.integers(0, len(v), (B, len(v)))], 1)
    return float(np.median(v)), float(np.quantile(m, .025)), float(np.quantile(m, .975))


t2 = []
for crit in ('original', 'z'):
    d = div[div.criterio == crit]
    for esc in ('pixel', 'parcela'):
        t2.append(dict(medida=f'Diversidad |ρ| entre los 4 conceptos (explicaciones, {crit})', escala=esc, n=len(d), **dict(zip(('mediana', 'IC_inf', 'IC_sup'), bmed(d[esc])))))
for esc in ('pixel', 'parcela'):
    t2.append(dict(medida='Referencia: |ρ| entre semillas 0 y 1, EG sin condicionar (paso 0)', escala=esc, n=len(sue), **dict(zip(('mediana', 'IC_inf', 'IC_sup'), bmed(sue[esc])))))
sp3 = pd.read_csv(os.path.join(F10, 'paso3', 'suelo_ruido_semillas.csv'))
for crit in ('original', 'z'):
    t2.append(dict(medida=f'Suelo |ρ| entre semillas, EG condicionado ({crit})', escala='pixel', n=int((sp3.criterio == crit).sum()),
                   **dict(zip(('mediana', 'IC_inf', 'IC_sup'), bmed(sp3[sp3.criterio == crit].rho_semillas.abs())))))
    t2.append(dict(medida=f'Suelo |ρ| entre semillas, EG condicionado ({crit})', escala='parcela', n=0, mediana=np.nan, IC_inf=np.nan, IC_sup=np.nan))
t2.append(dict(medida='Aleatorización |ρ|, EG condicionado', escala='parcela', n=0, mediana=np.nan, IC_inf=np.nan, IC_sup=np.nan))
t2 = pd.DataFrame(t2)
print(t2.to_string())
t1.to_csv(os.path.join(S, 'pureza_semillas.csv'), index=False)
t2.to_csv(os.path.join(S, 'parcelas.csv'), index=False)
# ---- Tablas (mismo formato que 08_figuras_tablas.py)
TAB = os.path.join(F10, 'tabla')
n3 = lambda v, d=3: '--' if not np.isfinite(v) else f'{v:.{d}f}'.replace('.', '{,}')   # noqa: E731


def tex(nombre, cab, filas, alin):
    with open(os.path.join(TAB, nombre + '.tex'), 'w') as o:
        o.write(f'\\begin{{tabular}}{{{alin}}}\n\\toprule\n' + ' & '.join(cab) + ' \\\\\n\\midrule\n')
        for f in filas:
            o.write(' & '.join(f) + ' \\\\\n')
        o.write('\\bottomrule\n\\end{tabular}\n')


t2.to_csv(os.path.join(TAB, 't8_eg_parcelas.csv'), index=False)
tex('t8_eg_parcelas', ['Medida', 'Escala', '$n$', 'Mediana', 'IC 95 \\%'],
    [[r.medida.replace('|ρ|', '$|\\rho|$'), 'píxel' if r.escala == 'pixel' else 'parcela 16 px', str(r.n), n3(r.mediana),
      f'({n3(r.IC_inf)}; {n3(r.IC_sup)})' if r.n else 'no guardado'] for r in t2.itertuples()], 'p{7cm}lrrr')
t1.to_csv(os.path.join(TAB, 't7_atlas_pureza_semilla1.csv'), index=False)
nom = {'pureza_concepto': 'clase del concepto', 'pureza_mayoritaria': 'clase mayoritaria'}
tex('t7_atlas_pureza_semilla1', ['Pureza', 'Fila', 'Mediana', 'Media', 'Diferencia de medias (IC 95 \\%)'],
    [[nom[r.medida], r.fila.replace('A', '\\texttt{A}') if r.fila == 'A' or r.fila.endswith(' A') else r.fila, n3(r.mediana), n3(r.media),
      '' if not np.isfinite(r.dif) else f'{n3(r.dif)} ({n3(r.IC_inf)}; {n3(r.IC_sup)})'] for r in t1.itertuples()], 'llrrr')
print('COMPROBACIONES_TERMINADAS')
