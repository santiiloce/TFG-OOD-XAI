# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Pasos 3 y 4 (GPU). Todos los métodos comparados (A, G×E, T4, EG, IG) se calculan en este mismo trabajo.
    python 02_eg_ig_fidelidad.py [--limite 2]     # --limite: solo mide tiempo con 2 imágenes, no escribe resultados
Detalles fijados en ../decisiones.md."""
import argparse
from pathlib import Path
import json
import os
import sys
import time

from comun_t4 import (T4, aleatorizado, aopc, boot_ic, cargar, composite_final, composite_t4, copia_para_lrp, curva, eg,
                      fusionar_ln, gxe, ig, ordenes, preprocesar, rho, rutas_train, seleccion, semilla_clase, sorteo_eg)

import numpy as np
import pandas as pd
import torch
from crp.attribution import CondAttribution
from torchvision.transforms.functional import gaussian_blur

ap = argparse.ArgumentParser()
ap.add_argument('--limite', type=int, default=0)
ap.add_argument('--prueba', type=int, default=0, help='N imágenes de test y de sensibilidad, 1 sorteo; escribe en t4_ig/prueba')
args = ap.parse_args()
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
entorno = dict(dispositivo=str(device), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               nodo=os.uname().nodename, python=sys.executable, torch=torch.__version__)
print(json.dumps(entorno), flush=True)
SALIDA = os.path.join(T4, 'prueba' if args.prueba else 'final')
N_SORTEOS = 1 if args.prueba else 5
os.makedirs(SALIDA, exist_ok=True)
METODOS = ['A', 'G×E', 'T4', 'EG', 'IG']
ALEAT = ['1', '0.model.stages.3']
K = 128                  # muestras de EG y pasos de IG (cambio previo a los pasos 3-4: antes 32; ver decisiones.md)
OBJ_COMPLETITUD = 0.05   # objetivo: error relativo mediano < 5 %
ABPC_A_FASE5 = (0.170, 0.039, 0.307)   # ABPC desenfoque de A en la fase 5 (media, IC 95 %), mismas 112 imágenes
T0 = time.perf_counter()

learn, model = cargar('convnext', device)
base_lrp = copia_para_lrp(model).to(device)
mT, _ = fusionar_ln(copia_para_lrp(model).to(device))
compA, compT = composite_final(), composite_t4()


def lrp(m, comp, xi, c):
    a = CondAttribution(m, no_param_grad=True)(xi.clone().requires_grad_(True), [{'y': [c]}], comp, init_rel=semilla_clase(c))
    return a.heatmap[0].detach().cpu().numpy()


_cache_refs = {}
HILOS = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))


def cargar_refs(rutas_r):
    """Como preprocesar (comun2), pero en lotes de 16 con HILOS procesos: mismo preprocesado, mismo orden."""
    dl = learn.dls.test_dl([Path(r) for r in rutas_r], bs=16, num_workers=HILOS)
    return torch.cat([b[0].as_subclass(torch.Tensor) for b in dl]).to(device)


def por_parcela(h, P=16):
    """Suma del mapa en las parcelas de P px de la prueba de fidelidad (para |ρ| a nivel de parcela)."""
    return torch.nn.functional.avg_pool2d(torch.as_tensor(h)[None, None], P)[0, 0].numpy() * P * P


def refs_de(indices, cache=True):
    if not cache:      # en test cada imagen tiene sus referencias: no se guardan (≈ 2,8 GB)
        return cargar_refs(rutas_train(indices))
    k = tuple(int(i) for i in indices)
    if k not in _cache_refs:
        _cache_refs[k] = cargar_refs(rutas_train(indices)).cpu()
    return _cache_refs[k].to(device)


class Explicador:
    """Mapas de los cinco métodos para un modelo (original o aleatorizado) con las mismas referencias de EG."""

    def __init__(self, m_orig, m_A, m_T4):
        self.m, self.mA, self.mT = m_orig, m_A, m_T4

    def mapas(self, xi, c, refs, alfas, metodos=METODOS):
        out, info = {}, {}
        for met in metodos:
            if met == 'A':
                out[met] = lrp(self.mA, compA, xi, c)
            elif met == 'T4':
                out[met] = lrp(self.mT, compT, xi, c)
            elif met == 'G×E':
                out[met] = gxe(self.m, xi, c)
            elif met == 'EG':
                out[met], s, fr = eg(self.m, xi, c, refs, alfas)
                info['EG'] = (s, fr)
            elif met == 'IG':
                base = gaussian_blur(xi, kernel_size=97, sigma=16.0)
                out[met], s, fr = ig(self.m, xi, c, base, pasos=K)
                info['IG'] = (s, fr)
        return out, info


# ---------------- Paso 3 y fidelidad: 112 imágenes de test
_, rutas, labels = seleccion(8)
if args.limite or args.prueba:
    rutas, labels = rutas[:args.limite or args.prueba], labels[:args.limite or args.prueba]
x = preprocesar(learn, rutas, device)
N = len(x)
with torch.no_grad():
    logits = torch.cat([model(x[i:i + 16]) for i in range(0, N, 16)])
pred = logits.argmax(1).cpu().numpy()
idx_eg, alf_eg = sorteo_eg(len(seleccion(8)[1]), k=K)
exp = Explicador(model, base_lrp, mT)
mapas = {k: [] for k in METODOS}
comp_filas, t_met = [], {k: 0.0 for k in METODOS + ['carga_refs']}
for i in range(N):
    c, xi = int(pred[i]), x[i:i + 1]
    t = time.perf_counter()
    refs = refs_de(idx_eg[i], cache=False)
    t_met['carga_refs'] += time.perf_counter() - t
    alfas = torch.as_tensor(alf_eg[i], dtype=x.dtype, device=device)
    for met in METODOS:
        t = time.perf_counter()
        h, info = exp.mapas(xi, c, refs, alfas, [met])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_met[met] += time.perf_counter() - t
        mapas[met].append(h[met])
        if met in info:
            s, fr = info[met]
            fx = float(logits[i, c])
            comp_filas.append(dict(metodo=met, imagen=i, f_x=fx, f_ref=fr, suma_atrib=s, delta_f=fx - fr,
                                   err_rel=abs(s - (fx - fr)) / abs(fx - fr)))
    if i % 16 == 15:
        print('mapas', i + 1, '/', N, round(time.perf_counter() - T0), 's', flush=True)
print('tiempo por imagen (s):', json.dumps({k: round(v / N, 3) for k, v in t_met.items()}), flush=True)
if args.limite:
    # Fidelidad: 2 órdenes x 128 parcelas por método; se mide con la 1.ª imagen y A
    fondo = gaussian_blur(x, kernel_size=97, sigma=16.0)
    t = time.perf_counter()
    o, n = ordenes(mapas['A'][0])
    for k in ('MoRF', 'LeRF'):
        curva(model, x[:1], fondo[:1], int(pred[0]), o[k], n * n // 2, n)
    tf = time.perf_counter() - t
    print('fidelidad, s por imagen y método:', round(tf, 2), flush=True)
    t_map, t_carga = sum(v for k, v in t_met.items() if k != 'carga_refs') / N, t_met['carga_refs'] / N
    est = 112 * (t_map + t_carga) + 112 * 5 * tf + 28 * (7 * t_map + t_met['EG'] / N) + 28 * 2 * t_carga
    print(f'ESTIMACION_TOTAL: {est / 60:.0f} min', flush=True)
    sys.exit(0)
np.savez_compressed(os.path.join(SALIDA, 'mapas_test.npz'), **{k: np.stack(v).astype(np.float32) for k, v in mapas.items()}, pred=pred)
cmp = pd.DataFrame(comp_filas)
cmp.to_csv(os.path.join(SALIDA, 'completitud_por_imagen.csv'), index=False)
cres = pd.DataFrame([dict(metodo=k, n=len(g), K=K, err_rel_mediana=g.err_rel.median(), p5=g.err_rel.quantile(.05), p95=g.err_rel.quantile(.95),
                          err_rel_max=g.err_rel.max(), delta_f_mediana=g.delta_f.median(),
                          cumple_objetivo_5pc=bool(g.err_rel.median() < OBJ_COMPLETITUD)) for k, g in cmp.groupby('metodo')])
cres.to_csv(os.path.join(SALIDA, 'completitud_resumen.csv'), index=False)
print(cres.round(4).to_string(index=False), flush=True)

fondo = gaussian_blur(x, kernel_size=97, sigma=16.0)
ok = pred == labels
res, filas, npz = {}, [], {}
for met in METODOS:
    r = {'MoRF': [], 'LeRF': []}
    for i in range(N):
        if not np.isfinite(mapas[met][i]).all():
            for k in r:
                r[k].append(np.full(129, np.nan))
            continue
        o, n = ordenes(mapas[met][i])
        for k in r:
            r[k].append(curva(model, x[i:i + 1], fondo[i:i + 1], int(pred[i]), o[k], n * n // 2, n))
    cur = {k: np.stack(v) for k, v in r.items()}
    for k, v in cur.items():
        npz[f'desenfoque|{met}|{k}'] = v.astype(np.float32)
    a = {k: aopc(v) for k, v in cur.items()}
    res[met] = a
    ab = a['MoRF'] - a['LeRF']
    val = np.isfinite(ab)
    ic = boot_ic(ab[val])
    filas.append(dict(metodo=met, n_validos=int(val.sum()), MoRF=a['MoRF'][val].mean(), LeRF=a['LeRF'][val].mean(), ABPC=ab[val].mean(),
                      ABPC_IC_inf=ic[0], ABPC_IC_sup=ic[1], ABPC_bien_clasificadas=ab[val & ok].mean(),
                      nota='referencia = desenfoque, mismo fondo que la prueba: puede salir favorecido' if met == 'IG' else ''))
    print('fidelidad', met, 'ABPC', round(filas[-1]['ABPC'], 3), ic, round(time.perf_counter() - T0), 's', flush=True)
pd.DataFrame(filas).to_csv(os.path.join(SALIDA, 'fidelidad.csv'), index=False)
fA = next(f for f in filas if f['metodo'] == 'A')
cordura = dict(ABPC_A=float(fA['ABPC']), IC_A=[float(fA['ABPC_IC_inf']), float(fA['ABPC_IC_sup'])], ABPC_A_fase5=ABPC_A_FASE5[0], IC_A_fase5=list(ABPC_A_FASE5[1:]),
               dif_con_fase5=float(fA['ABPC']) - ABPC_A_FASE5[0], dentro_IC_fase5=bool(ABPC_A_FASE5[1] <= fA['ABPC'] <= ABPC_A_FASE5[2]))
print('CORDURA A frente a la fase 5:', json.dumps(cordura), flush=True)
np.savez_compressed(os.path.join(SALIDA, 'curvas.npz'), **npz)
pares = []
for met in METODOS:
    for ref in ('A', 'G×E'):
        if met == ref:
            continue
        d = (res[met]['MoRF'] - res[met]['LeRF']) - (res[ref]['MoRF'] - res[ref]['LeRF'])
        d = d[np.isfinite(d)]
        ic = boot_ic(d)
        pares.append(dict(metodo=met, referencia=ref, medida='ABPC desenfoque', n=len(d), dif_media=d.mean(), IC_inf=ic[0], IC_sup=ic[1],
                          frac_pos=float((d > 0).mean())))
pd.DataFrame(pares).to_csv(os.path.join(SALIDA, 'diferencias_emparejadas.csv'), index=False)
print(pd.DataFrame(pares).round(3).to_string(index=False), flush=True)

# ---------------- Sensibilidad: 28 imágenes de la fase 5
_, rutas_s, _ = seleccion(2)
if args.prueba:
    rutas_s = rutas_s[:args.prueba]
xs = preprocesar(learn, rutas_s, device)
with torch.no_grad():
    lg = model(xs)
idx_s, alf_s = sorteo_eg(len(xs), k=K, semilla=0)
idx_s1, alf_s1 = sorteo_eg(len(xs), k=K, semilla=1)       # control: suelo de ruido de EG
sens, base_map = [], {}
for i in range(len(xs)):
    o = torch.argsort(lg[i], descending=True)
    c, c2 = int(o[0]), int(o[1])
    refs, alfas = refs_de(idx_s[i]), torch.as_tensor(alf_s[i], dtype=xs.dtype, device=device)
    h1, _ = exp.mapas(xs[i:i + 1], c, refs, alfas)
    h2, _ = exp.mapas(xs[i:i + 1], c2, refs, alfas)
    base_map[i] = (c, h1)
    for met in METODOS:
        sens.append(dict(metodo=met, imagen=i, fila='clase_2a', sorteo=-1, rho=rho(h1[met], h2[met])))
    hr = eg(model, xs[i:i + 1], c, refs_de(idx_s1[i]), torch.as_tensor(alf_s1[i], dtype=xs.dtype, device=device))[0]
    sens.append(dict(metodo='EG', imagen=i, fila='control_EG_otra_semilla', sorteo=-1, rho=rho(h1['EG'], hr)))
    # diagnóstico añadido (14-11, no interviene en decisiones): mismo suelo de ruido sobre las sumas por parcela de 16 px
    sens.append(dict(metodo='EG', imagen=i, fila='control_EG_otra_semilla_parcela16', sorteo=-1, rho=rho(por_parcela(h1['EG']), por_parcela(hr))))
print('sensibilidad: clases', round(time.perf_counter() - T0), 's', flush=True)
for sorteo in range(N_SORTEOS):
    torch.manual_seed(sorteo)
    ma = aleatorizado(base_lrp, ALEAT)                  # misma red aleatorizada para todos los métodos
    torch.manual_seed(sorteo)
    mta, _ = fusionar_ln(aleatorizado(base_lrp, ALEAT))
    expa = Explicador(ma, ma, mta)
    for i in range(len(xs)):
        c, h1 = base_map[i]
        refs, alfas = refs_de(idx_s[i]), torch.as_tensor(alf_s[i], dtype=xs.dtype, device=device)
        h2, _ = expa.mapas(xs[i:i + 1], c, refs, alfas)
        for met in METODOS:
            sens.append(dict(metodo=met, imagen=i, fila='etapa3_y_cabeza', sorteo=sorteo, rho=rho(h1[met], h2[met])))
    del ma, mta, expa
    print('sorteo', sorteo, round(time.perf_counter() - T0), 's', flush=True)
s = pd.DataFrame(sens)
s['abs'] = s.rho.abs()
s.to_csv(os.path.join(SALIDA, 'sensibilidad_por_imagen.csv'), index=False)
rs = []
for (met, fila), g in s.groupby(['metodo', 'fila'], sort=False):
    img = g.groupby('imagen')['abs'].mean().dropna().values
    ic = boot_ic(img, np.median)
    r = dict(metodo=met, fila=fila, mediana_abs_rho=float(np.median(img)), IC_inf=ic[0], IC_sup=ic[1], n_nan=int(g['abs'].isna().sum()))
    if (g.sorteo >= 0).any():
        sm = g.groupby('sorteo')['abs'].median()
        r.update(mediana_entre_sorteos=float(sm.median()), sorteo_min=float(sm.min()), sorteo_max=float(sm.max()))
    rs.append(r)
rs = pd.DataFrame(rs)
rs.to_csv(os.path.join(SALIDA, 'sensibilidad_resumen.csv'), index=False)
print(rs.round(3).to_string(index=False), flush=True)

# ---------------- Reglas de decisión (decisiones.md)
dp = pd.DataFrame(pares).set_index(['metodo', 'referencia'])
al = rs[rs.fila == 'etapa3_y_cabeza'].set_index('metodo')['mediana_entre_sorteos']
dec = dict(T4_IC_ABPC_sobre_A=[float(dp.loc[('T4', 'A'), 'IC_inf']), float(dp.loc[('T4', 'A'), 'IC_sup'])],
           T4_aleat=float(al['T4']), A_aleat=float(al['A']),
           EG_IC_ABPC_sobre_A=[float(dp.loc[('EG', 'A'), 'IC_inf']), float(dp.loc[('EG', 'A'), 'IC_sup'])])
dec['T4_sustituye_a_A'] = bool(dec['T4_IC_ABPC_sobre_A'][0] > 0 and dec['T4_aleat'] <= dec['A_aleat'] + 0.1)
dec['EG_se_anade'] = bool(dec['EG_IC_ABPC_sobre_A'][0] > 0)
json.dump(dict(decision=dec, cordura_A_fase5=cordura, completitud=cres.to_dict('records'), K=K, entorno=entorno, minutos=(time.perf_counter() - T0) / 60), open(os.path.join(SALIDA, 'decision.json'), 'w'), indent=1, ensure_ascii=False, default=float)
print(json.dumps(dec, ensure_ascii=False), flush=True)
print('PASOS34_TERMINADOS', flush=True)
sys.stdout.flush()
os._exit(0)   # sin esto, procesos de carga (num_workers) dejaban el trabajo vivo tras terminar (prueba 192052)
