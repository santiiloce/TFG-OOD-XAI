# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase 8 — Covariate shift sintético: corrupciones de Zhang et al. sobre el test ID, características y logits del
modelo, y puntuaciones de todos los detectores. Todo en CPU de atenea (TITAN Xp, sm_61, no soportada por el torch
2.8.0+cu128 de fastai_env_ood), inferencia en fp32.

Reutiliza sin cambios:
  - my_evauator_openood.py: SwinWrapper (features = pooling tras la LayerNorm final, antes de la lineal), transform
    (Resize((224, 224)) + ToTensor + Normalize ImageNet), build_postprocessor, CachedSwin/CachedSwinASH.
  - stoodx_paper.py: preparar_clases, precalcular, puntuar, k_clase (K = 500, coseno, clase predicha, -z).
Los detectores se ajustan con las características congeladas de train/val (features/completo), como en results_ood/.

Uso (sbatch scripts/script_cpu_atenea.sh results_corrupciones/scripts/fase8_extraer.py ...):
  --modelo swin --n 500 --salida results_corrupciones/prueba_500     (prueba de tiempos, paso 2)
  --modelo swin --n 3000 --salida results_corrupciones                (o --n 0 para el test completo)
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
import my_evauator_openood as ev          # noqa: E402
import stoodx_paper as sx                 # noqa: E402
import corrupciones_zhang as cz           # noqa: E402

# Los postprocessors de OpenOOD mueven los tensores con .cuda(); en CPU se convierte en no-op.
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self

# Parametrización por modelo. Para ConvNeXt-S, añadir su entrada (envoltorio con la misma interfaz que SwinWrapper,
# carpeta de features congeladas y carpetas de resultados de la fase 9).
MODELOS = {
    'swin': {'red': lambda: ev.SwinWrapper(ev.NUM_CLASSES, ev.CHECKPOINT_PATH, ev.MODEL_NAME),
             'transform': ev.transform,
             'features': os.path.join(RAIZ, 'features/completo'),
             'faseA': os.path.join(RAIZ, 'results_ood/faseA_2026-10-20_11-19-35'),
             'faseB': os.path.join(RAIZ, 'results_ood/faseB_paper_2026-10-20')},
}
DETECTORES_OPENOOD = ['msp', 'mls', 'ebo', 'react', 'ash', 'dice', 'knn', 'mds', 'rmds', 'vim',
                      'she', 'she_cosine', 'she_euclidean']
STOODX = {'stoodx_100': '100', 'stoodx_875': '875'}
K_STOODX = 500
N_CPU = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
LOTE = 64
SEMILLA = 0


def muestra_estratificada(label, n, minimo):
    """Índices ordenados de una submuestra estratificada por clase (semilla 0): cuotas proporcionales, al menos
    `minimo` por clase si la clase las tiene, ajuste al total por restos mayores."""
    if not n or n >= len(label):
        return np.arange(len(label))
    rng = np.random.default_rng(SEMILLA)
    clases, cuentas = np.unique(label, return_counts=True)
    ideal = n * cuentas / cuentas.sum()
    cuota = np.maximum(np.floor(ideal).astype(int), np.minimum(minimo, cuentas))
    while cuota.sum() < n:
        i = np.argmax(np.where(cuota < cuentas, ideal - cuota, -np.inf))
        cuota[i] += 1
    while cuota.sum() > n:
        i = np.argmax(np.where(cuota > np.minimum(minimo, cuentas), cuota - ideal, -np.inf))
        cuota[i] -= 1
    idx = np.concatenate([rng.choice(np.where(label == c)[0], q, replace=False) for c, q in zip(clases, cuota)])
    return np.sort(idx)


def _init_pool():
    from threadpoolctl import threadpool_limits
    import cv2
    cv2.setNumThreads(1)
    threadpool_limits(1)


ORIGINALES = []   # imágenes limpias (PIL) de la muestra; se cargan una vez antes de crear el pool (fork)


def _corromper(args):
    i, ruta, c, n = args
    a = cz.corromper(ruta, c, n, ORIGINALES[i])
    return a, hashlib.md5(a.tobytes()).hexdigest()


class Detectores:
    """Ajuste único con las características congeladas (train, val); puntuación de cualquier matriz de features."""

    def __init__(self, cfg, pool):
        self.pool = pool
        cargar = lambda n: dict(np.load(os.path.join(cfg['features'], f'{n}.npz')))
        datos = {n: cargar(n) for n in ['id_train', 'id_val', 'id_test']}
        fc = cfg['red']().backbone.get_classifier().eval()
        self.redes = {'cache': ev.CachedSwin(fc).eval(), 'ash': ev.CachedSwinASH(fc).eval()}
        loader = lambda d: ev.make_loader(ev.FeaturesDataset(d['feats'], d['label']), num_workers=0)
        id_loader = {'train': loader(datos['id_train']), 'val': loader(datos['id_val']),
                     'test': loader(datos['id_test'])}
        self.pp, self.t_ajuste = {}, {}
        for m in DETECTORES_OPENOOD:
            t0 = time.perf_counter()
            pp, _ = ev.build_postprocessor(m)
            pp.setup(self.redes['ash' if m == 'ash' else 'cache'], id_loader, None)
            self.pp[m], self.t_ajuste[m] = pp, time.perf_counter() - t0
        self.sx = {}
        for m, conf in STOODX.items():
            t0 = time.perf_counter()
            clases = sx.preparar_clases(datos['id_train'], sx.CONFIGS[conf]['quantile'], torch.device('cpu'))
            tablas = sx.precalcular(clases, K_STOODX, 'fija')
            self.sx[m] = (clases, tablas)
            self.t_ajuste[m] = time.perf_counter() - t0
        print('ajuste de detectores (s):', {k: round(v, 1) for k, v in self.t_ajuste.items()}, flush=True)

    @torch.no_grad()
    def puntuar(self, feats, logits):
        etiquetas = np.full(len(feats), -1, dtype=np.int64)
        out, tiempos = {}, {}
        for m, pp in self.pp.items():
            t0 = time.perf_counter()
            loader = ev.make_loader(ev.FeaturesDataset(feats.astype(np.float32), etiquetas), num_workers=0)
            _, conf, _ = pp.inference(self.redes['ash' if m == 'ash' else 'cache'], loader, progress=False)
            out[m], tiempos[m] = conf.astype(np.float64), time.perf_counter() - t0
        for m, (clases, tablas) in self.sx.items():
            t0 = time.perf_counter()
            r = sx.puntuar({'feats': feats.astype(np.float32), 'logits': logits}, clases, tablas, K_STOODX, self.pool)
            out[m] = -r['z']
            out[f'{m}_U'] = r['U']
            kx = np.array([sx.k_clase(clases[c]['n'], K_STOODX, 'fija')[0] for c in range(ev.NUM_CLASSES)])
            kr = np.array([sx.k_clase(clases[c]['n'], K_STOODX, 'fija')[1] for c in range(ev.NUM_CLASSES)])
            out[f'{m}_n1n2'] = (kx * kx * kr)[r['pred']].astype(np.float64)
            tiempos[m] = time.perf_counter() - t0
        return out, tiempos


@torch.no_grad()
def extraer(red, transform, imagenes):
    feats, logits = [], []
    from PIL import Image
    for i in range(0, len(imagenes), LOTE):
        x = torch.stack([transform(Image.fromarray(a)) for a in imagenes[i:i + LOTE]])
        lo, fe = red(x, return_feature=True)
        feats.append(fe.float().numpy())
        logits.append(lo.float().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--modelo', default='swin', choices=list(MODELOS))
    p.add_argument('--n', type=int, default=0, help='tamaño total (0 = test completo, o igual a --n_sub)')
    p.add_argument('--n_sub', type=int, default=3000, help='submuestra que se completa primero')
    p.add_argument('--minimo', type=int, default=10, help='mínimo por clase en la submuestra')
    p.add_argument('--salida', required=True)
    p.add_argument('--corrupciones', nargs='+', default=cz.CORRUPCIONES)
    args = p.parse_args()
    salida = args.salida if os.path.isabs(args.salida) else os.path.join(RAIZ, args.salida)
    cfg = MODELOS[args.modelo]

    torch.set_num_threads(N_CPU)
    torch.manual_seed(SEMILLA)
    print(f'CPU asignadas (SLURM_CPUS_PER_TASK) = {N_CPU}; torch.get_num_threads() = {torch.get_num_threads()}; '
          f'cuda disponible = {torch.cuda.is_available()}', flush=True)
    test = dict(np.load(os.path.join(cfg['features'], 'id_test.npz')))
    # Bloques: primero la submuestra estratificada (--n_sub, todas las celdas) y después el resto del test (si --n 0),
    # para que, si el trabajo se corta, la submuestra esté completa. Cada celda de cada bloque se guarda aparte.
    idx_sub = muestra_estratificada(test['label'], args.n_sub, args.minimo)
    bloques = [('sub', idx_sub)]
    idx_total = muestra_estratificada(test['label'], args.n, args.minimo)
    resto = np.setdiff1d(idx_total, idx_sub)
    if len(resto):
        bloques.append(('resto', resto))
    elif not np.array_equal(idx_total, idx_sub):
        sys.exit('--n debe ser 0 (test completo) o igual a --n_sub')
    print('bloques:', {b: len(i) for b, i in bloques}, flush=True)

    registro_path = os.path.join(salida, 'tiempos_celdas.json')
    registro = json.load(open(registro_path)) if os.path.exists(registro_path) else {}
    celdas = [('limpia', 0)] + [(c, n) for c in args.corrupciones for n in cz.NIVELES]
    det = red = None
    from PIL import Image
    for bloque, idx in bloques:
        celdas_dir = os.path.join(salida, 'celdas', bloque)
        os.makedirs(celdas_dir, exist_ok=True)
        rutas = [str(r) for r in test['paths'][idx]]
        np.savez(os.path.join(salida, f'muestra_{bloque}.npz'), idx=idx, paths=np.array(rutas),
                 label=test['label'][idx])
        print(f'bloque {bloque}: {len(idx)} imágenes; por clase '
              f'{np.bincount(test["label"][idx], minlength=14).tolist()}', flush=True)
        pendientes = [(c, n) for c, n in celdas if not os.path.exists(os.path.join(celdas_dir, f'{c}_{n}.npz'))]
        if not pendientes:
            continue
        t0 = time.perf_counter()
        ORIGINALES.clear()
        for r in rutas:
            with Image.open(r) as im:
                ORIGINALES.append(im.convert('RGB'))
        print(f'lectura de {len(rutas)} imágenes: {time.perf_counter() - t0:.0f} s', flush=True)
        pool = mp.get_context('fork').Pool(N_CPU, initializer=_init_pool)
        if det is None:
            t0 = time.perf_counter()
            det = Detectores(cfg, pool)
            registro['_global'] = {'cpu': N_CPU, 't_ajuste_detectores_s': det.t_ajuste,
                                   't_ajuste_total_s': time.perf_counter() - t0,
                                   'torch_threads': torch.get_num_threads(), 'nodo': os.uname().nodename}
            red = cfg['red']().eval()
        det.pool = pool

        # Referencia: los detectores (ajustados de nuevo en CPU) puntuando las features congeladas del bloque.
        ref_path = os.path.join(celdas_dir, 'congeladas_reajuste_cpu.npz')
        if not os.path.exists(ref_path):
            sc, _ = det.puntuar(test['feats'][idx], test['logits'][idx])
            np.savez(ref_path, **sc)

        for c, n in pendientes:
            nombre = f'{c}_{n}'
            t0 = time.perf_counter()
            res = pool.map(_corromper, [(i, r, c, n) for i, r in enumerate(rutas)], chunksize=8)
            imagenes = [a for a, _ in res]
            hashes = np.array([h for _, h in res])
            t1 = time.perf_counter()
            feats, logits = extraer(red, cfg['transform'], imagenes)
            t2 = time.perf_counter()
            del imagenes, res
            sc, t_det = det.puntuar(feats, logits)
            t3 = time.perf_counter()
            destino = os.path.join(celdas_dir, f'{nombre}.npz')
            np.savez(destino + '.tmp.npz', feats=feats.astype(np.float32), logits=logits.astype(np.float32),
                     pred=logits.argmax(1), md5=hashes, **sc)
            os.replace(destino + '.tmp.npz', destino)       # escritura atómica: una celda está completa o no existe
            registro[f'{bloque}/{nombre}'] = {'t_corrupcion_s': t1 - t0, 't_extraccion_s': t2 - t1,
                                              't_detectores_s': t_det, 't_detectores_total_s': t3 - t2,
                                              'n': len(rutas)}
            json.dump(registro, open(registro_path, 'w'), indent=1)
            print(f'{bloque}/{nombre:<22} corr {t1 - t0:6.1f} s | extr {t2 - t1:7.1f} s | det {t3 - t2:6.1f} s '
                  f'(stoodx {t_det["stoodx_100"] + t_det["stoodx_875"]:.1f} s) | acc '
                  f'{np.mean(logits.argmax(1) == test["label"][idx]):.4f}', flush=True)
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
