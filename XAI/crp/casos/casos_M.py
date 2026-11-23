# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Casos, vecinos y canales de las explicaciones definidos con el STOOD-X de la fase 9b en el espacio M
(media espacial de los 768 canales de stages.3.blocks.2). Adaptación de ../../scripts/casos.py (versión con B).
Se mantienen los casos de la versión con B que cumplen la definición en M. Solo CPU; escribe en coherencia_f9/M/casos/.
Uso: /mnt/homeGPU/slcerro/fastai_env_ood/bin/python results_crp/coherencia_f9/M/scripts/casos_M.py
"""
import json, os
import numpy as np
import pandas as pd

R = '/mnt/homeGPU/slcerro/OpenOOD'
F9M = f'{R}/results_ood_convnext/espacio_principal'
F4 = f'{R}/results_crp/fase4'
CB = f'{R}/results_crp/coherencia_f9/casos/casos_explicacion.json'
S = f'{R}/results_crp/coherencia_f9/M/casos'
os.makedirs(S, exist_ok=True)
N9 = {'test': 'id_test', 'nct_tum': 'near_nct_tum', 'nct_norm': 'near_nct_norm'}
CLASES = pd.read_csv(f'{F4}/explicaciones/caracteristicas_por_clase.csv')['clase'].tolist()
SUELO = -38.6911

def fM(n, k): return np.load(f'{F9M}/features_M/{n}_{k}.npy', allow_pickle=True)
f4 = {n: np.load(f'{F4}/features/convnext_{n}.npz') for n in ('train', 'test', 'nct_tum', 'nct_norm')}
st = {n: dict(np.load(f'{F9M}/espacio_M/partes_stoodx_100/{N9[n]}.npz')) for n in N9}
M = {n: fM(N9[n], 'M') for n in N9}
for n in N9:
    assert (st[n]['paths'] == f4[n]['paths']).all() and (st[n]['pred'] == f4[n]['pred']).all()
    assert (fM(N9[n], 'paths') == f4[n]['paths']).all()
thr = float(np.percentile(st['test']['conf'], 5))
acc = {n: st[n]['conf'] >= thr for n in N9}
def soft(n):
    L = f4[n]['logits'].astype(np.float64); e = np.exp(L - L.max(1, keepdims=True)); return (e / e.sum(1, keepdims=True)).max(1)
sm = {n: soft(n) for n in N9}
te = f4['test']; bien = te['pred'] == te['label']
selB = json.load(open(CB))['casos']
sel4 = json.load(open(f'{F4}/explicaciones/seleccion.json'))['casos']
def casosB(g): return [(s['conjunto'], s['indice']) for s in selB if s['grupo'] == g]
pred = {n: f4[n]['pred'] for n in N9}

casos, motivos = [], {}
def add(g, n, i, m): casos.append((g, n, int(i))); motivos[(g, n, int(i))] = m
def ngrupo(g): return sum(c[0] == g for c in casos)

# --- ID bien clasificada
g = 'bien_clasificada'; clases = set()
for n, i in casosB(g):
    if bien[i] and acc['test'][i] and pred[n][i] not in clases: add(g, n, i, 'se mantiene (versión B)'); clases.add(pred[n][i])
for i in np.argsort(-sm['test'], kind='stable'):
    if ngrupo(g) == 4: break
    if bien[i] and acc['test'][i] and te['pred'][i] not in clases: add(g, 'test', i, 'nueva: mayor softmax'); clases.add(te['pred'][i])

# --- ID mal clasificada
g = 'mal_clasificada'
for n, i in casosB(g):
    assert not bien[i]; add(g, n, i, 'se mantiene (versión B)')

# --- ID rechazada (falsa alarma)
g = 'id_rechazada'; clases = set()
for n, i in casosB(g):
    if bien[i] and not acc['test'][i] and pred[n][i] not in clases: add(g, n, i, 'se mantiene (versión B)'); clases.add(pred[n][i])
eleg = np.where(bien & ~acc['test'] & ~st['test']['saturada'])[0]
for i in np.random.default_rng(0).permutation(eleg):
    if ngrupo(g) == 4: break
    if te['pred'][i] not in clases: add(g, 'test', i, 'nueva: sorteo semilla 0 (no saturada)'); clases.add(te['pred'][i])
n_rech = dict(bien_y_rechazadas=int((bien & ~acc['test']).sum()), no_saturadas=int(len(eleg)))

# --- OOD aceptada / rechazada
ood = {}
def mas_cercana(idx, conf, med, desde_arriba, excluir, n):
    for excl in (True, False):
        ok = idx[pred[n][idx] != excluir] if (excl and excluir is not None) else idx
        if desde_arriba: ok = ok[conf[ok] >= med]
        if len(ok): return int(ok[np.argmin(np.abs(conf[ok] - med))]), excl or excluir is None
    raise RuntimeError
for g in ('ood_aceptada', 'ood_rechazada'):
    clase_tum = None
    for n in ('nct_tum', 'nct_norm'):
        conf = st[n]['conf']
        pool = np.where(acc[n])[0] if g == 'ood_aceptada' else np.where(~acc[n] & ~st[n]['saturada'])[0]
        assert len(pool), (g, n)
        med = float(np.median(conf[pool]))
        ood[f'{g}/{n}'] = dict(n_pool=int(len(pool)), frac_aceptadas=float(acc[n].mean()), n_rechazadas=int((~acc[n]).sum()), mediana_pool=med)
        previo = [i for m, i in casosB(g) if m == n][0]
        cumple = (acc[n][previo] and conf[previo] >= med) if g == 'ood_aceptada' else (not acc[n][previo] and not st[n]['saturada'][previo])
        if cumple and pred[n][previo] != clase_tum:
            add(g, n, previo, 'se mantiene (versión B)'); i = previo
        else:
            i, distinta = mas_cercana(pool, conf, med, g == 'ood_aceptada', clase_tum, n)
            add(g, n, i, ('nueva: aceptada más cercana a la mediana desde arriba' if g == 'ood_aceptada'
                          else 'nueva: rechazada no saturada más cercana a la mediana') + ('' if distinta else ' (no hay de otra clase)'))
        if n == 'nct_tum': clase_tum = pred[n][i]

# --- Vecinos (M) y canales (M)
def nrm(X): X = np.asarray(X, np.float64); return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
Mtr = fM('id_train', 'M'); p9tr = fM('id_train', 'pred')
assert (p9tr == f4['train']['pred']).all()
mu, sd = Mtr.mean(0), Mtr.std(0)
def presencia(Mx, z):
    P = np.abs((Mx - mu) / sd if z else Mx).sum(0); top = np.argsort(-P)[:4]; return top.tolist(), P[top].tolist()
def d50(n, i):
    c = pred[n][i]; s = nrm(f4['train']['feats'][f4['train']['pred'] == c]) @ nrm(f4[n]['feats'][i][None])[0]
    return float(1 - np.sort(s)[::-1][min(50, len(s)) - 1])
f6 = pd.read_csv(f'{R}/results_crp/fase6/canal102/presencia_normalizada.csv')
p4 = {(s['conjunto'], s['indice']): (j, s) for j, s in enumerate(sel4)}
pB = {(s['conjunto'], s['indice']): s for s in selB}
GR_4 = {'id_rechazada': 'baja_pertenencia'}

filas = []
for g, n, i in casos:
    c = int(pred[n][i])
    idx = np.where(p9tr == c)[0]
    d = 1 - nrm(Mtr[idx]) @ nrm(M[n][i][None])[0]
    o = np.argsort(d, kind='stable')[:5]
    assert (idx[o] == st[n]['vecinos_idx'][i]).all(), (g, n, i)          # = vecinos que usa la puntuación de la fase 9b
    vec, dv = idx[o[:3]].tolist(), d[o[:3]].tolist()
    Mx = np.vstack([M[n][i][None], Mtr[vec]])
    can, pres = presencia(Mx, False); canz, presz = presencia(Mx, True)
    lab = int(f4[n]['label'][i])
    r = dict(grupo=g, conjunto=n, indice=i, ruta=str(f4[n]['paths'][i]), clase_real=CLASES[lab] if lab >= 0 else 'OOD',
             clase_pred=CLASES[c], c=c, softmax=float(sm[n][i]), d50=d50(n, i), vecinos=vec, dist_vecinos=dv,
             canales=can, presencia=pres, canales_z=canz, presencia_z=presz,
             stoodx_menos_z=float(st[n]['conf'][i]), stoodx_p=float(st[n]['p'][i]), saturada=bool(st[n]['saturada'][i]),
             percentil_id_test=float((st['test']['conf'] <= st[n]['conf'][i]).mean() * 100),
             decision='ID (aceptada)' if acc[n][i] else 'OOD (rechazada)', motivo=motivos[(g, n, i)])
    q = p4.get((n, i))                                     # misma semántica de claves que la versión B: comparación con la fase 4
    if q is not None and (q[1]['grupo'] == g or GR_4.get(g) == q[1]['grupo']):
        j4, s4 = q; z6 = json.loads(f6.canales_z[j4])
        r.update(en_fase4=True, vecinos_fase4=s4['vecinos'], solape_vecinos=len(set(vec) & set(s4['vecinos'])),
                 canales_fase4=s4['canales'], solape_canales=len(set(can) & set(s4['canales'])),
                 canales_z_fase6=z6, solape_canales_z=len(set(canz) & set(z6)))
    else:
        r['en_fase4'] = False
    b = pB.get((n, i))                                      # comparación con la versión B (claves añadidas)
    if b is not None and b['grupo'] == g:
        r.update(en_B=True, vecinos_B=b['vecinos'], solape_vecinos_B=len(set(vec) & set(b['vecinos'])),
                 canales_B=b['canales'], solape_canales_B=len(set(can) & set(b['canales'])),
                 canales_z_B=b['canales_z'], solape_canales_z_B=len(set(canz) & set(b['canales_z'])),
                 menos_z_B=b['stoodx_menos_z'], percentil_B=b['percentil_id_test'], decision_B=b['decision'])
    else:
        r['en_B'] = False
    filas.append(r)

# Casos de la versión B que no se mantienen: su situación en M (para explicar el cambio)
fuera = []
for b in selB:
    n, i = b['conjunto'], b['indice']
    if (b['grupo'], n, i) not in casos:
        fuera.append(dict(grupo=b['grupo'], conjunto=n, indice=i, menos_z_B=b['stoodx_menos_z'], decision_B=b['decision'],
                          menos_z_M=float(st[n]['conf'][i]), percentil_M=float((st['test']['conf'] <= st[n]['conf'][i]).mean() * 100),
                          decision_M='ID (aceptada)' if acc[n][i] else 'OOD (rechazada)', saturada_M=bool(st[n]['saturada'][i]),
                          mediana_pool_M=ood.get(f"{b['grupo']}/{n}", {}).get('mediana_pool')))
info = dict(espacio='M', umbral_menos_z=thr, retenido_id_test=float(acc['test'].mean()), suelo_saturacion=SUELO, id_rechazada=n_rech,
            ood=ood, casos_B_sustituidos=fuera)
json.dump(dict(info=info, casos=filas), open(f'{S}/casos_explicacion.json', 'w'), indent=1, ensure_ascii=False)
df = pd.DataFrame(filas); df.to_csv(f'{S}/casos_explicacion.csv', index=False)
pd.set_option('display.width', 320); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 36)
print(json.dumps(info, indent=1, default=str))
print(df[['grupo', 'conjunto', 'indice', 'clase_real', 'clase_pred', 'softmax', 'stoodx_menos_z', 'percentil_id_test', 'saturada', 'decision', 'motivo']].to_string())
print(df[['grupo', 'indice', 'vecinos', 'dist_vecinos', 'canales', 'canales_z', 'solape_vecinos', 'solape_canales', 'solape_canales_z',
          'solape_vecinos_B', 'solape_canales_B', 'solape_canales_z_B']].to_string())
