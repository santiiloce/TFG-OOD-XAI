# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9b, paso 1 (CPU, zeus): espacio M = media espacial de stages.3.blocks.2 = C[:, 768:].

1. Comprobación con hook en 100 imágenes de id_test (fp32, mismo preprocesado que f9_extraer.py).
2. features_M/<conj>_M.npy desde C (sin pasar imágenes) + enlaces a logits/pred/label/paths de la fase 9.
3. Medida de tiempo de STOOD-X en M con 1.000 imágenes de id_test (para la estimación total).
Salida: espacio_principal/comprobacion_M.json, espacio_principal/features_M/
"""
import json, multiprocessing as mp, os, sys, time
import numpy as np, torch

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
F9 = os.path.join(RAIZ, 'results_ood_convnext')
EP = os.path.join(F9, 'espacio_principal')
FEATS = os.path.join(F9, 'features')
OUT = os.path.join(EP, 'features_M')
sys.path.insert(0, os.path.join(F9, 'scripts'))
sys.path.insert(0, RAIZ)
import f9_extraer as fe   # noqa: E402  (cargar_modelo, Imagenes, TF; sin modificar)
NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
CONJ = ['id_train', 'id_val', 'id_test', 'control_res_128', 'near_nct_tum', 'near_nct_norm', 'near_nct_tum_norm',
        'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus', 'cov_panda', 'far_dtd']


@torch.no_grad()
def comprobar_hook(n=100):
    model, _, _ = fe.cargar_modelo()
    nombres = dict(model.named_modules())
    clave = [k for k in nombres if k.endswith('stages.3.blocks.2')]
    assert len(clave) == 1, clave
    capt = {}
    h = nombres[clave[0]].register_forward_hook(lambda m, i, o: capt.__setitem__('x', o.detach().clone()))
    rutas = [str(p) for p in np.load(os.path.join(FEATS, 'id_test_paths.npy'))[:n]]
    x = torch.stack([fe.TF(fe.Image.open(r).convert('RGB')) for r in rutas])
    logits = model(x); h.remove()
    a = capt['x']                                    # (n, 768, H, W)
    media, maximo = a.mean((2, 3)).numpy(), a.amax((2, 3)).numpy()
    C = np.load(os.path.join(FEATS, 'id_test_C.npy'), mmap_mode='r')[:n]
    L = np.load(os.path.join(FEATS, 'id_test_logits.npy'))[:n]
    r = {'capa': clave[0], 'forma_activacion': list(a.shape), 'n': n,
         'max_dif_media_vs_C_768_1535': float(np.abs(media - C[:, 768:]).max()),
         'max_dif_maximo_vs_C_0_767': float(np.abs(maximo - C[:, :768]).max()),
         'max_dif_media_vs_C_0_767': float(np.abs(media - C[:, :768]).max()),
         'max_abs_M': float(np.abs(C[:, 768:]).max()),
         'max_dif_logits_vs_guardados': float(np.abs(logits.numpy() - L).max())}
    print(json.dumps(r, indent=1), flush=True)
    assert r['max_dif_media_vs_C_768_1535'] < 1e-4, 'M no coincide con la mitad media de C'
    return r


def escribir_M():
    os.makedirs(OUT, exist_ok=True)
    info = {}
    for c in CONJ:
        C = np.load(os.path.join(FEATS, f'{c}_C.npy'), mmap_mode='r')
        assert C.shape[1] == 1536
        M = np.ascontiguousarray(C[:, 768:], dtype=np.float32)
        np.save(os.path.join(OUT, f'{c}_M.npy'), M)
        for k in ('logits', 'pred', 'label', 'paths'):
            dst = os.path.join(OUT, f'{c}_{k}.npy')
            if not os.path.lexists(dst):
                os.symlink(os.path.join(FEATS, f'{c}_{k}.npy'), dst)
        info[c] = {'n': int(M.shape[0]), 'frac_neg': float((M < 0).mean()), 'media_norma': float(np.linalg.norm(M, axis=1).mean())}
        print(c, info[c], flush=True)
    return info


def medir_stoodx(n=1000):
    import stoodx_paper as sx
    g = lambda c, k: np.load(os.path.join(OUT, f'{c}_{k}.npy'))
    train = {'feats': g('id_train', 'M'), 'logits': g('id_train', 'logits'), 'paths': g('id_train', 'paths')}
    sx.N_FEATURES = 768
    t0 = time.perf_counter()
    clases = sx.preparar_clases(train, 0.0, torch.device('cpu'))
    tablas = sx.precalcular(clases, 500, 'fija')
    t_pre = time.perf_counter() - t0
    idx = np.sort(np.random.default_rng(0).choice(12618, n, replace=False))
    d = {k: g('id_test', k)[idx] for k in ('logits', 'paths', 'label')}
    d['feats'] = g('id_test', 'M')[idx]
    with mp.get_context('fork').Pool(NCPU) as pool:
        t = time.perf_counter(); sx.puntuar(d, clases, tablas, 500, pool, 'fija'); dt = time.perf_counter() - t
    n_total = sum(len(g(c, 'paths')) for c in ['id_test', 'near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym',
                                                 'afin_nct_mus', 'cov_panda', 'control_res_128', 'far_dtd'])
    return {'n': n, 't_precalculo_s': t_pre, 'segundos': dt, 'img_por_s': n / dt, 'n_total_stoodx': n_total,
            'horas_stoodx_estimadas': (t_pre + n_total * dt / n) / 3600}


if __name__ == '__main__':
    torch.set_num_threads(NCPU)
    res = {'hook': comprobar_hook(), 'M': escribir_M(), 'medida_stoodx': medir_stoodx(), 'cpu': NCPU,
           'nodo': os.uname().nodename}
    json.dump(res, open(os.path.join(EP, 'comprobacion_M.json'), 'w'), indent=1)
    print(json.dumps(res['medida_stoodx'], indent=1))
