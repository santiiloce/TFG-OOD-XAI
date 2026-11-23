# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""Fase 9, paso 1: extracción con ConvNeXt-S (CPU, fp32) en una sola pasada por conjunto.

Por conjunto guarda en results_ood_convnext/features/:
  <conj>_B.npy       salida de la ReLU de la cabeza (head[5]), 512, >= 0   (espacio principal)
  <conj>_C.npy       concat pooling (head[1]), 1536                        (sensibilidad)
  <conj>_logits.npy, <conj>_pred.npy, <conj>_label.npy, <conj>_paths.npy
y fc_B.npz con W' y b' (BN(512) plegada en la última Linear): logits = W'·B + b'.

Las rutas, su orden y las etiquetas se toman de features/completo/<conj>.npz (Swin, solo lectura), para poder
emparejar por imagen. Preprocesado: Resize((256,256)) bilineal sobre PIL + ToTensor + Normalize(imagenet),
equivalente al test_dl del learner (comprobado: dif. de logits 6e-5). Control: Resize((128,128)) y después
lo mismo (como my_evauator_openood.py:232-235, pero a 256).

    python f9_extraer.py --medir 1000      # solo tiempo, no guarda features
    python f9_extraer.py                   # todos los conjuntos (se salta los ya hechos)
"""
import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fastai.vision.all import load_learner

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
PKL = '/mnt/homeGPU/slcerro/tfg/executions/def1/param2_2026-03-24_16-19-23/convnext_small_model.pkl'
SWIN = os.path.join(RAIZ, 'features/completo')
OUT = os.path.join(RAIZ, 'results_ood_convnext/features')
CONJUNTOS = ['id_train', 'id_val', 'id_test', 'control_res_128', 'near_nct_tum', 'near_nct_norm',
             'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus', 'cov_panda', 'far_dtd']
UNIONES = {'near_nct_tum_norm': ['near_nct_tum', 'near_nct_norm']}
NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
WORKERS = max(2, NCPU // 5)
BS = 64

norm = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
TF = transforms.Compose([transforms.Resize((256, 256))] + norm)
TF_CONTROL = transforms.Compose([transforms.Resize((128, 128)), transforms.Resize((256, 256))] + norm)


class Imagenes(Dataset):
    def __init__(self, rutas, tf): self.rutas, self.tf = rutas, tf
    def __len__(self): return len(self.rutas)
    def __getitem__(self, i): return self.tf(Image.open(self.rutas[i]).convert('RGB'))


def cargar_modelo():
    sys.path.insert(0, '/mnt/homeGPU/slcerro/tfg/executions/def1')
    model = load_learner(PKL, cpu=True).model.float().eval()
    head = model[1]
    W = head[8].weight.detach().double()
    assert head[8].bias is None
    bn = head[6]
    g = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
    Wp = W * g[None, :]
    bp = W @ (bn.bias.detach().double() - g * bn.running_mean.double())
    return model, Wp.numpy(), bp.numpy()


@torch.no_grad()
def pasar(model, rutas, tf):
    head, capt = model[1], {}
    hs = [head[1].register_forward_hook(lambda m, i, o: capt.__setitem__('C', o.detach().clone())),
          head[5].register_forward_hook(lambda m, i, o: capt.__setitem__('B', o.detach().clone()))]
    B, C, L = [], [], []
    dl = DataLoader(Imagenes(rutas, tf), batch_size=BS, shuffle=False, num_workers=WORKERS)
    for x in dl:
        L.append(model(x)); B.append(capt['B']); C.append(capt['C'])
    for h in hs: h.remove()
    return torch.cat(B).numpy(), torch.cat(C).numpy(), torch.cat(L).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--medir', type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(NCPU)
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    model, Wp, bp = cargar_modelo()
    print(f'CPU asignadas: {NCPU}; hilos torch: {torch.get_num_threads()}; workers: {WORKERS}', flush=True)

    if args.medir:
        rutas = [str(p) for p in np.load(os.path.join(SWIN, 'id_train.npz'))['paths']]
        idx = np.sort(np.random.default_rng(0).choice(len(rutas), args.medir, replace=False))
        pasar(model, [rutas[i] for i in idx[:BS]], TF)          # calentamiento
        t = time.perf_counter()
        B, C, L = pasar(model, [rutas[i] for i in idx], TF)
        dt = time.perf_counter() - t
        n_total = sum(len(np.load(os.path.join(SWIN, f'{c}.npz'))['paths']) for c in CONJUNTOS)
        dif = float(np.abs(B.astype(np.float64) @ Wp.T + bp - L).max())
        r = {'n': args.medir, 'segundos': dt, 'img_por_s': args.medir / dt, 'n_total': n_total,
             'horas_extraccion_estimadas': n_total / (args.medir / dt) / 3600, 'cpu': NCPU,
             'nodo': os.uname().nodename, 'max_dif_logits_BN_plegada': dif}
        print(json.dumps(r, indent=1), flush=True)
        json.dump(r, open(os.path.join(OUT, '..', 'medida_tiempo.json'), 'w'), indent=1)
        return

    np.savez(os.path.join(OUT, 'fc_B.npz'), W=Wp, b=bp)
    info_path = os.path.join(OUT, 'info.json')
    info = json.load(open(info_path)) if os.path.exists(info_path) else {'conjuntos': {}}
    for c in CONJUNTOS:
        if os.path.exists(os.path.join(OUT, f'{c}_paths.npy')):
            print(f'{c}: ya hecho', flush=True); continue
        ref = np.load(os.path.join(SWIN, f'{c}.npz'))
        rutas = [str(p) for p in ref['paths']]
        t = time.perf_counter()
        B, C, L = pasar(model, rutas, TF_CONTROL if c == 'control_res_128' else TF)
        dt = time.perf_counter() - t
        pred = L.argmax(1)
        dif = float(np.abs(B.astype(np.float64) @ Wp.T + bp - L).max())
        assert dif < 1e-4, f'{c}: W·B + b difiere de los logits en {dif}'
        assert B.min() >= 0
        for k, v in [('B', B.astype(np.float32)), ('C', C.astype(np.float32)), ('logits', L.astype(np.float32)),
                     ('pred', pred.astype(np.int64)), ('label', ref['label'].astype(np.int64)), ('paths', ref['paths'])]:
            np.save(os.path.join(OUT, f'{c}_{k}.npy'), v)
        acc = float(np.mean(pred == ref['label'])) if ref['label'][0] >= 0 else None
        info['conjuntos'][c] = {'n': len(rutas), 'segundos': dt, 'max_dif_logits_BN_plegada': dif, 'exactitud': acc,
                                'frac_B_cero': float(np.mean(B == 0)),
                                'coincidencia_pred_swin': float(np.mean(pred == ref['pred']))}
        print(f'{c}: {len(rutas)} imágenes, {dt:.0f} s, dif={dif:.2e}, acc={acc}', flush=True)
        json.dump(info, open(info_path, 'w'), indent=1)

    for u, partes in UNIONES.items():
        d = {k: np.concatenate([np.load(os.path.join(OUT, f'{p}_{k}.npy')) for p in partes])
             for k in ['B', 'C', 'logits', 'pred', 'label', 'paths']}
        orden = np.argsort(d['paths'], kind='stable')
        assert np.array_equal(d['paths'][orden], np.load(os.path.join(SWIN, f'{u}.npz'))['paths'])
        for k, v in d.items():
            np.save(os.path.join(OUT, f'{u}_{k}.npy'), v[orden])
    info.update({'pkl': PKL, 'cpu': NCPU, 'nodo': os.uname().nodename, 'precision': 'fp32',
                 'torch': torch.__version__,
                 'B': 'salida de la ReLU head[5] (512, >= 0); logits = W_B·B + b_B (fc_B.npz, BN(512) plegada)',
                 'C': 'salida de AdaptiveConcatPool2d+Flatten head[1] (1536 = avg 768 + max 768), antes de BN(1536)'})
    json.dump(info, open(info_path, 'w'), indent=1)
    print('Extracción terminada', flush=True)


if __name__ == '__main__':
    main()
