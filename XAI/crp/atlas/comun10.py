# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 10: conductancia (Dhamdhere et al., 2019) con el camino de Expected Gradients del 191879.
Cuerpo = model[0] (TimmBody; su salida es exactamente stages.3.blocks.2) y cabeza = model[1]."""
import os
import sys

RESULTS = '/mnt/homeGPU/slcerro/OpenOOD/results_crp'
F10 = os.path.join(RESULTS, 'fase10_eg')
sys.path.insert(0, os.path.join(RESULTS, 't4_ig', 'scripts'))

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from comun_t4 import (aleatorizado, aopc, boot_ic, cargar, curva, eg, gxe,  # noqa: E402,F401
                      ordenes, preprocesar, rho, rutas_train, seleccion, semilla_clase, sorteo_eg)

CAPA = '0.model.stages.3.blocks.2'


def cargar_refs(learn, rutas, device, hilos=None):
    """Como preprocesar, en lotes de 16 con varios procesos (mismo preprocesado y orden; ver t4_ig/decisiones.md)."""
    from pathlib import Path
    hilos = hilos or int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    dl = learn.dls.test_dl([Path(r) for r in rutas], bs=16, num_workers=hilos)
    return torch.cat([b[0].as_subclass(torch.Tensor) for b in dl]).to(device)


def conductancia(model, xi, c, refs, alfas, lote=16, con_fref=True):
    """C_k (768) por la vía eficiente: jvp del cuerpo con tangente x − x̃ y gradiente de f_c solo en la cabeza.
    Devuelve (C, f_c en las referencias (K,))."""
    cuerpo, cabeza = model[0], model[1]
    C = torch.zeros(768, device=xi.device, dtype=xi.dtype)
    fref = []
    for i in range(0, len(refs), lote):
        r, a_ = refs[i:i + lote], alfas[i:i + lote].reshape(-1, 1, 1, 1)
        z, dz = r + a_ * (xi - r), (xi - r)
        with torch.no_grad():          # la jvp (modo hacia delante) no necesita el grafo de autograd
            a, v = torch.func.jvp(cuerpo, (z,), (dz,))
        a = a.detach().requires_grad_(True)
        g = torch.autograd.grad(cabeza(a)[:, c].sum(), a)[0]
        C += (g * v).sum((0, 2, 3))
        if con_fref:
            with torch.no_grad():
                fref.append(model(r)[:, c])
    return C / len(refs), (torch.cat(fref) if con_fref else None)


def conductancia_muchas(model, xs, cs, refs_por_img, alfas_por_img, lote=16):
    """C para varias imágenes; refs_por_img: lista de tensores (K, 3, H, W)."""
    return [conductancia(model, xs[j:j + 1], cs[j], refs_por_img[j], alfas_por_img[j], lote) for j in range(len(xs))]


def mapas_canal(model, xi, c, refs, alfas, canales, lote=16):
    """φ^k para cada canal de `canales`: (len(canales), H, W), suma RGB. Señal en la capa enmascarada al canal k, con g fija."""
    cuerpo, cabeza = model[0], model[1]
    canales = list(canales)
    out = torch.zeros(len(canales), *xi.shape[-2:], device=xi.device, dtype=xi.dtype)
    for i in range(0, len(refs), lote):
        r, a_ = refs[i:i + lote], alfas[i:i + lote].reshape(-1, 1, 1, 1)
        z = (r + a_ * (xi - r)).detach().requires_grad_(True)
        a = cuerpo(z)
        ad = a.detach().requires_grad_(True)
        g = torch.autograd.grad(cabeza(ad)[:, c].sum(), ad)[0]
        for j, k in enumerate(canales):
            gz = torch.autograd.grad((g[:, k] * a[:, k]).sum(), z, retain_graph=(j < len(canales) - 1))[0]
            out[j] += ((xi - r) * gz).sum((0, 1))
    return out / len(refs)


def mapas_todos_canales(model, xi, c, refs, alfas, lote=8, trozo=16):
    """φ^k de los 768 canales con vjp por lotes de canales (is_grads_batched). Devuelve (768, H, W)."""
    cuerpo, cabeza = model[0], model[1]
    out = torch.zeros(768, *xi.shape[-2:], device=xi.device, dtype=xi.dtype)
    for i in range(0, len(refs), lote):
        r, a_ = refs[i:i + lote], alfas[i:i + lote].reshape(-1, 1, 1, 1)
        z = (r + a_ * (xi - r)).detach().requires_grad_(True)
        a = cuerpo(z)
        ad = a.detach().requires_grad_(True)
        g = torch.autograd.grad(cabeza(ad)[:, c].sum(), ad)[0]
        d = xi - r
        for k0 in range(0, 768, trozo):
            ks = torch.arange(k0, min(k0 + trozo, 768), device=xi.device)
            gm = torch.zeros(len(ks), *g.shape, device=g.device, dtype=g.dtype)
            gm[torch.arange(len(ks)), :, ks] = g[:, ks].transpose(0, 1)
            gz = torch.autograd.grad(a, z, grad_outputs=gm, is_grads_batched=True, retain_graph=True)[0]   # (n, B, 3, H, W)
            out[ks] += (d[None] * gz).sum((1, 2))
        del a
    return out / len(refs)


def comprobar_particion(model, x):
    """head(body(x)) = model(x) y body(x) = salida de CAPA."""
    sal = {}
    h = dict(model.named_modules())[CAPA].register_forward_hook(lambda _m, _i, o: sal.__setitem__('a', o))
    with torch.no_grad():
        y = model(x)
        a = model[0](x)
        y2 = model[1](a)
    h.remove()
    return float((y - y2).abs().max()), float((a - sal['a']).abs().max())


# ---- Paso 4: reserva de referencias y sorteo por imagen (el mismo de 04_relmax_bloque.py)
def cargar_reserva(learn, device):
    tr = np.load(os.path.join(RESULTS, 'fase4', 'features', 'convnext_train.npz'))
    idx = np.load(os.path.join(F10, 'paso4', 'reserva_idx.npy'))
    pool = torch.cat([cargar_refs(learn, [str(p) for p in tr['paths'][idx[a:a + 512]]], 'cpu') for a in range(0, len(idx), 512)])
    return pool.to(device)


def sorteo_reserva(conj, i, s, device, P=4096, K=128):
    rng = np.random.default_rng([s, i] if conj == 'train' else [s, 1_000_000 + i])
    return rng.integers(0, P, K), torch.as_tensor(rng.random(K), dtype=torch.float32, device=device)
