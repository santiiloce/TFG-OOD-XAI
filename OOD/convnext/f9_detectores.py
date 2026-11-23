# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9, paso 2: detectores OOD sobre las características del ConvNeXt-S (CPU).

Reutiliza sin modificarlos my_evauator_openood.py (CachedSwin, CachedSwinASH, FeaturesDataset, configs) y
stoodx_paper.py (preparar_clases, precalcular, puntuar). Mismos .yml por defecto, APS_mode=False, ajuste solo con ID.
Los postprocessors de OpenOOD llaman a .cuda(); en CPU se sustituye por la identidad (no cambia ningún cálculo).

    python f9_detectores.py --espacio B                      # 13 métodos + vim_256 + STOOD-X
    python f9_detectores.py --espacio C --metodos knn mds rmds --stoodx_conjuntos near_nct_tum near_nct_norm
Salida: results_ood_convnext/espacio_<B|C>/scores_<metodo>.npz (mismo formato que results_ood/faseA_*), config.json
"""
import argparse, json, multiprocessing as mp, os, sys, time
import numpy as np, torch, torch.nn as nn

torch.Tensor.cuda = lambda self, *a, **k: self          # CPU: .cuda() de OpenOOD -> identidad
nn.Module.cuda = lambda self, *a, **k: self

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)
import my_evauator_openood as me          # noqa: E402  (solo se usan sus clases y constantes)
import stoodx_paper as sx                 # noqa: E402
import metricas_ood                       # noqa: E402
from openood.postprocessors import get_postprocessor   # noqa: E402
from openood.utils.config import Config, merge_configs  # noqa: E402

FEATS = os.environ.get('F9_FEATS', os.path.join(RAIZ, 'results_ood_convnext/features'))   # F9_*: solo pruebas
NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
NUM_CLASSES = 14
VIM_DIM = 171                             # principal: 512/3 (misma proporción que 256/768 en el Swin)
METODOS = ['msp', 'mls', 'ebo', 'knn', 'mds', 'rmds', 'vim', 'vim_256',
           'she', 'she_cosine', 'she_euclidean', 'react', 'ash', 'dice']
A_PUNTUAR = ['id_val', 'id_test'] + me.CONJUNTOS_PRINCIPALES
STOODX_DISCO = ['id_test', 'near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
                'cov_panda', 'control_res_128', 'far_dtd']
K_STOODX = 500


def cargar(nombre, espacio):
    g = lambda k: np.load(os.path.join(FEATS, f'{nombre}_{k}.npy'))
    return {'feats': g(espacio).astype(np.float32), 'logits': g('logits'), 'pred': g('pred'),
            'label': g('label'), 'paths': g('paths')}


class CachedC(nn.Module):
    """Espacio C: no hay capa lineal de C a los logits, así que la entrada es [C | logits guardados]."""
    def __init__(self, dim): super().__init__(); self.dim = dim
    def forward(self, x, return_feature=False):
        f, lo = x[:, :self.dim], x[:, self.dim:]
        return (lo, f) if return_feature else lo


def construir(metodo):
    base, _, var = metodo.partition('_')
    if base not in ('she', 'vim'):
        base, var = metodo, ''
    cfg = Config(os.path.join(me.CONFIG_ROOT, 'postprocessors', f'{base}.yml'))
    cfg = merge_configs(cfg, Config(**{'dataset': {'name': me.ID_DATA_NAME}}))
    cfg.postprocessor.APS_mode = False
    if base == 'she' and var:
        cfg.postprocessor.postprocessor_args.metric = var
    if base == 'vim':
        cfg.postprocessor.postprocessor_args.dim = int(var) if var else VIM_DIM
    return get_postprocessor(cfg), cfg


def detectores(args, salida):
    datos = {n: cargar(n, args.espacio) for n in ['id_train'] + A_PUNTUAR}
    if args.espacio == 'B':
        fcd = np.load(os.path.join(FEATS, 'fc_B.npz'))
        fc = nn.Linear(fcd['W'].shape[1], NUM_CLASSES)
        fc.weight.data = torch.from_numpy(fcd['W']).float(); fc.bias.data = torch.from_numpy(fcd['b']).float()
        redes = {'cache': me.CachedSwin(fc).eval(), 'ash': me.CachedSwinASH(fc).eval()}
        entrada = lambda d: d['feats']
    else:
        dim = datos['id_train']['feats'].shape[1]
        redes = {'cache': CachedC(dim)}
        entrada = lambda d: np.concatenate([d['feats'], d['logits']], 1).astype(np.float32)

    loader = lambda n: me.make_loader(me.FeaturesDataset(entrada(datos[n]), datos[n]['label']), num_workers=0)
    id_loaders = {'train': loader('id_train'), 'val': loader('id_val'), 'test': loader('id_test')}
    registro = {}
    for metodo in args.metodos:
        t0 = time.perf_counter()
        pp, cfg = construir(metodo)
        net = redes['ash' if metodo == 'ash' else 'cache']
        pp.setup(net, id_loaders, None)
        punt = {}
        for n in A_PUNTUAR:
            pred, conf, label = pp.inference(net, loader(n), progress=False)
            punt[n] = {'conf': conf.astype(np.float64), 'pred': pred, 'label': label, 'paths': datos[n]['paths']}
        np.savez(os.path.join(salida, f'scores_{metodo}.npz'),
                 **{f'{n}_{k}': v for n, d in punt.items() for k, v in d.items()})
        idt = punt['id_test']
        extra = {}
        if metodo == 'react': extra['umbral_react'] = float(pp.threshold)
        if metodo == 'ash': extra['s1_val_id'] = me.estadisticas_s1_ash(datos['id_val']['feats'], pp.percentile)
        if metodo == 'dice': extra['umbral_contribucion_dice'] = float(pp.thresh)
        registro[metodo] = {'hiperparametros': me.hiperparametros(cfg), 'tiempo_s': time.perf_counter() - t0,
                            'acc_id_test': metricas_ood.exactitud(idt['pred'], idt['label']), **extra}
        auroc = metricas_ood.auroc(idt['conf'], punt['near_nct_tum_norm']['conf'])
        print(f'{metodo}: {registro[metodo]["tiempo_s"]:.0f} s, acc={registro[metodo]["acc_id_test"]:.4f}, '
              f'AUROC TUM+NORM={auroc:.4f}', flush=True)
        json.dump(registro, open(os.path.join(salida, 'config_detectores.json'), 'w'), indent=1, default=float)


def stoodx(args, salida):
    train = cargar('id_train', args.espacio)
    sx.N_FEATURES = train['feats'].shape[1]            # solo se usa para la máscara (100 %: todas)
    dev = torch.device('cpu')
    t0 = time.perf_counter()
    clases = sx.preparar_clases(train, 0.0, dev)
    tablas = sx.precalcular(clases, K_STOODX, 'fija')
    t_pre = time.perf_counter() - t0
    n_c = {c: d['n'] for c, d in clases.items()}
    print(f'STOOD-X {args.espacio}: precálculo {t_pre:.0f} s; n_c = {n_c}', flush=True)
    partes = os.path.join(salida, 'partes_stoodx_100'); os.makedirs(partes, exist_ok=True)
    with mp.get_context('fork').Pool(NCPU) as pool:
        for n in args.stoodx_conjuntos:
            dest = os.path.join(partes, f'{n}.npz')
            if os.path.exists(dest): print(f'  {n}: ya hecho'); continue
            d = cargar(n, args.espacio)
            t = time.perf_counter()
            r = sx.puntuar(d, clases, tablas, K_STOODX, pool, 'fija')
            n1 = np.array([min(K_STOODX, n_c[c]) for c in r['pred']])
            n2 = n1 * np.array([min(K_STOODX, n_c[c] - 1) for c in r['pred']])
            np.savez(dest, conf=-r['z'], p=r['p'], U=r['U'], z=r['z'], pred=r['pred'], label=d['label'],
                     paths=d['paths'], saturada=(r['U'] == n1 * n2), vecinos_idx=r['vecinos_idx'],
                     vecinos_dist=r['vecinos_dist'], segundos=time.perf_counter() - t, t_precalculo_s=t_pre)
            print(f'  {n}: {len(d["paths"])} imágenes en {time.perf_counter() - t:.0f} s; '
                  f'saturadas {np.mean(r["U"] == n1 * n2):.3f}; min(-z)={(-r["z"]).min():.4f}', flush=True)
    # Unión TUM+NORM y fichero con el formato de la fase B
    punt = {n: dict(np.load(os.path.join(partes, f'{n}.npz'))) for n in args.stoodx_conjuntos}
    if {'near_nct_tum', 'near_nct_norm'} <= set(punt):
        j = {k: np.concatenate([punt['near_nct_tum'][k], punt['near_nct_norm'][k]])
             for k in punt['near_nct_tum'] if punt['near_nct_tum'][k].ndim > 0}
        o = np.argsort(j['paths'], kind='stable')
        punt['near_nct_tum_norm'] = {k: v[o] for k, v in j.items()}
    keys = ['conf', 'p', 'U', 'z', 'pred', 'label', 'paths', 'saturada']
    np.savez(os.path.join(salida, 'scores_stoodx_100.npz'),
             **{f'{n}_{k}': d[k] for n, d in punt.items() for k in keys})
    json.dump({'K': K_STOODX, 'n_c_train_predicha': n_c, 't_precalculo_s': t_pre, 'distancia': 'coseno',
               'referencia': 'train ID de la clase predicha', 'puntuacion': '-z', 'features': '100 %'},
              open(os.path.join(salida, 'config_stoodx.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--espacio', choices=['B', 'C'], required=True)
    ap.add_argument('--metodos', nargs='*', default=METODOS)
    ap.add_argument('--stoodx_conjuntos', nargs='*', default=STOODX_DISCO)
    args = ap.parse_args()
    torch.set_num_threads(NCPU); torch.manual_seed(0); np.random.seed(0)
    salida = os.path.join(os.environ.get('F9_SALIDA', os.path.join(RAIZ, 'results_ood_convnext')), f'espacio_{args.espacio}'); os.makedirs(salida, exist_ok=True)
    print(f'Espacio {args.espacio}; CPU {NCPU}; métodos {args.metodos}; STOOD-X {args.stoodx_conjuntos}', flush=True)
    if args.metodos: detectores(args, salida)
    if args.stoodx_conjuntos: stoodx(args, salida)


if __name__ == '__main__':
    main()
