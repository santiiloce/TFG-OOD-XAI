# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
"""
Fase A — Evaluación OOD del mejor modelo del TFG (swin_small_patch4_window7_224) con OpenOOD v1.5.

    ID                     -> dataset de próstata, 14 clases (tfg/TrainTest)
    near_nct_tum_norm      -> NCT-CRC-HE-100K-NONORM, TUM + NORM (near-OOD principal)
    near_nct_tum/_norm     -> las mismas clases por separado
    afin_nct_str/lym/mus   -> clases de tejido análogas al ID (control interno: centro vs tejido)
    cov_panda              -> PANDA (próstata, otro centro y escáner)
    control_res_128        -> ID test reducido a 128 px y ampliado a 224
    far_dtd                -> Textures / DTD
    anexo_nct_*            -> ADI, BACK, DEB, MUC (solo anexo)

Dos etapas:
    1. exportar: una única pasada de la red por cada conjunto; se guardan en features/<tag>/
       las features de 768 dimensiones (antes de la capa lineal), logits, predicciones,
       etiquetas y rutas en .npz (legibles sin PyTorch; también los usa la fase B, STOOD-X).
    2. evaluar: los postprocessors de OpenOOD se ejecutan sobre las features guardadas con
       CachedSwin, que reproduce la interfaz de SwinWrapper. Todos los métodos evaluados actúan
       después del pooling, así que no hace falta volver a pasar las imágenes por la red.
       Con --verificar se comprueba que MSP directo sobre imágenes da lo mismo que con la caché.

No se usa la API de alto nivel `Evaluator(id_name=...)` porque solo admite los datasets
predefinidos de OpenOOD; se usan directamente los postprocessors con DataLoaders propios.

Entorno: /mnt/homeGPU/slcerro/fastai_env_ood (numpy < 2). Necesita GPU: sbatch script_ood.sh
    sbatch script_ood.sh --prueba --verificar     # prueba rápida (~500 imágenes por conjunto)
    sbatch script_ood.sh                          # ejecución completa
    sbatch script_ood.sh --etapa evaluar --metodos msp rmds   # reutiliza las features ya exportadas
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import timm
import metricas_ood
from openood.evaluators.metrics import compute_all_metrics
from openood.networks.ash_net import ash_b
from openood.postprocessors import get_postprocessor
from openood.postprocessors.info import num_classes_dict
from openood.utils.config import Config, merge_configs


# ---------------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------------

NUM_CLASSES = 14
MODEL_NAME = 'swin_small_patch4_window7_224'
CHECKPOINT_PATH = '/mnt/homeGPU/slcerro/tfg/executions/def1/param4_2026-03-24_21-55-59/swin_small_patch4_window7_224_pesos_finales.pth'

RAIZ = '/mnt/homeGPU/slcerro/OpenOOD'
ID_DIRS = {
    'id_train': '/mnt/homeGPU/slcerro/tfg/TrainTest/train',
    'id_val': '/mnt/homeGPU/slcerro/tfg/TrainTest/val',
    'id_test': '/mnt/homeGPU/slcerro/tfg/TrainTest/test',
}
NCT_DIR = os.environ.get('NCT_DIR', os.path.join(RAIZ, 'ood_data/nct_crc_nonorm'))   # NCT_DIR solo para depurar

# Conjuntos OOD que se leen de disco (carpeta, recursivo). near_nct_tum_norm no se exporta:
# es la unión ordenada de near_nct_tum y near_nct_norm (mismas features, sin segunda pasada).
OOD_DIRS = {
    'near_nct_tum': os.path.join(NCT_DIR, 'TUM'),
    'near_nct_norm': os.path.join(NCT_DIR, 'NORM'),
    'afin_nct_str': os.path.join(NCT_DIR, 'STR'),
    'afin_nct_lym': os.path.join(NCT_DIR, 'LYM'),
    'afin_nct_mus': os.path.join(NCT_DIR, 'MUS'),
    'cov_panda': os.path.join(RAIZ, 'ood_data/panda/ood_class'),
    'far_dtd': os.path.join(RAIZ, 'ood_data/dtd/dtd/images'),
    'anexo_nct_adi': os.path.join(NCT_DIR, 'ADI'),
    'anexo_nct_back': os.path.join(NCT_DIR, 'BACK'),
    'anexo_nct_deb': os.path.join(NCT_DIR, 'DEB'),
    'anexo_nct_muc': os.path.join(NCT_DIR, 'MUC'),
}
UNIONES = {'near_nct_tum_norm': ['near_nct_tum', 'near_nct_norm']}

# Control: las mismas imágenes de ID test reducidas a la resolución de PANDA (128x128) y
# ampliadas de nuevo a 224. Su contenido es ID: un AUROC alto indica que el método detecta la
# pérdida de resolución, no el cambio de centro. Conserva sus etiquetas reales (para la accuracy);
# en las métricas se trata como OOD.
CONTROL_RESOLUCION = 128
CONTROL_NAME = 'control_res_128'

# Orden de los conjuntos en todas las salidas
CONJUNTOS_PRINCIPALES = ['near_nct_tum_norm', 'near_nct_tum', 'near_nct_norm',
                         'afin_nct_str', 'afin_nct_lym', 'afin_nct_mus',
                         'cov_panda', CONTROL_NAME, 'far_dtd']
CONJUNTOS_ANEXO = ['anexo_nct_adi', 'anexo_nct_back', 'anexo_nct_deb', 'anexo_nct_muc']
CONJUNTOS_OOD = CONJUNTOS_PRINCIPALES + CONJUNTOS_ANEXO
CONJUNTOS_NCT_POR_CLASE = ['near_nct_tum', 'near_nct_norm', 'afin_nct_str', 'afin_nct_lym',
                           'afin_nct_mus'] + CONJUNTOS_ANEXO

# Métodos OOD. Los hiperparámetros se leen de configs/postprocessors/<metodo>.yml (APS_mode=False)
#   msp, mls, ebo                -> solo logits
#   knn, mds, rmds, vim, she*    -> se ajustan con las features de ID train
#   react                        -> umbral = percentil 90 de las features de ID val
#   ash                          -> ash_b sobre las 768 features agregadas (ver CachedSwinASH)
#   dice                         -> máscara de pesos con la media de las features de ID train
# DICE + ReAct no se evalúa: OpenOOD no lo implementa (DICEPostprocessor nunca llama a
# forward_threshold).
POSTPROCESSORS = ['msp', 'mls', 'ebo', 'knn', 'mds', 'rmds', 'vim',
                  'she', 'she_cosine', 'she_euclidean', 'react', 'ash', 'dice']

CONFIG_ROOT = os.path.join(RAIZ, 'configs')
ID_DATA_NAME = 'prostata_ugr'           # nombre que se registra en num_classes_dict de OpenOOD
RESULTS_ROOT = os.path.join(RAIZ, 'results_ood')
FEATURES_ROOT = os.path.join(RAIZ, 'features')

BATCH_SIZE = 64
NUM_WORKERS = 6
DEVICE = 'cuda'
SEMILLA = 0
N_PRUEBA = 500

# Evita el paso de descriptores de fichero entre workers del DataLoader (error "Bad file descriptor"
# de multiprocessing/resource_sharer visto en la prueba rápida, inocuo pero ruidoso)
torch.multiprocessing.set_sharing_strategy('file_system')

# SHE y MDS consultan num_classes_dict[config.dataset.name]; registramos nuestro dataset
num_classes_dict[ID_DATA_NAME] = NUM_CLASSES


# ---------------------------------------------------------------------------
# 2. WRAPPERS DEL MODELO
# ---------------------------------------------------------------------------

class SwinWrapper(nn.Module):
    """
    Envuelve el Swin de timm (entrenado con un Learner genérico de fastai y guardado
    con learn12.save -> {'model': state_dict, 'opt': ...}) para exponer la interfaz
    que esperan los postprocessors de OpenOOD:
        forward(x, return_feature=True) -> (logits, features)   # todos
        get_fc()                        -> (W, b) en numpy       # ViM, DICE
        forward_threshold(x, umbral)    -> logits               # ReAct
    Las features son la salida del average pooling (768 dimensiones), antes de la capa lineal.
    Se usa para exportar las features y para la verificación directa de MSP.
    """

    def __init__(self, num_classes, checkpoint_path, model_name):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        self.backbone.load_state_dict(state_dict)

    def get_features(self, x):
        return self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)

    def forward(self, x, return_feature=False):
        features = self.get_features(x)
        logits = self.backbone.get_classifier()(features)
        if return_feature:
            return logits, features
        return logits

    def forward_threshold(self, x, threshold):
        features = self.get_features(x).clip(max=threshold)
        return self.backbone.get_classifier()(features)

    def get_fc(self):
        fc = self.backbone.get_classifier()
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()


class CachedSwin(nn.Module):
    """
    Misma interfaz que SwinWrapper, pero la entrada `x` son las features de 768 dimensiones ya
    exportadas: solo se aplica la capa lineal del Swin. Válido para todos los postprocessors
    evaluados, porque ninguno necesita nada anterior al pooling.
    """

    def __init__(self, fc):
        super().__init__()
        self.fc = fc

    def forward(self, x, return_feature=False):
        logits = self.fc(x)
        if return_feature:
            return logits, x
        return logits

    def forward_threshold(self, x, threshold):
        # ReAct: recorte superior de las features
        return self.fc(x.clip(max=threshold))

    def get_fc(self):
        return self.fc.weight.cpu().detach().numpy(), self.fc.bias.cpu().detach().numpy()


class CachedSwinASH(CachedSwin):
    """
    ASH en OpenOOD (ASHPostprocessor) llama a net.forward_threshold(x, percentil), con un
    significado distinto al de ReAct; por eso tiene su propio wrapper, igual que ASHNet en
    openood/networks/ash_net.py. Como ASHNet, aplica ash_b sobre el vector de features ya
    agregado (B, 768, 1, 1), no sobre mapas de activación ni sobre los tokens del Swin.
    Ojo: ash_b reparte la suma total s1 de las features entre las top-k; tras la LayerNorm del
    Swin las features pueden ser negativas y s1 puede ser ~0 o negativa (ver config.json).
    """

    def forward_threshold(self, x, percentile):
        x = ash_b(x.clone().view(x.size(0), -1, 1, 1), percentile).view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 3. PREPROCESADO — igual que en la validación de fastai
#    Resize(224, method='squish') + Normalize.from_stats(*imagenet_stats)
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_control = transforms.Compose([
    transforms.Resize((CONTROL_RESOLUCION, CONTROL_RESOLUCION)),
    transform,
])


# ---------------------------------------------------------------------------
# 4. LISTAS DE IMÁGENES Y DATASETS
#    Orden determinista: rutas ordenadas, guardadas en features/<tag>/listas/<conjunto>.txt
# ---------------------------------------------------------------------------

EXTENSIONES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def listar_carpeta(root):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(root) for f in fs
                  if f.lower().endswith(EXTENSIONES))


def submuestra(rutas, etiquetas, n):
    """n imágenes elegidas con semilla fija y devueltas en el orden original (ordenado)."""
    if n is None or len(rutas) <= n:
        return rutas, etiquetas
    idx = np.sort(np.random.default_rng(SEMILLA).choice(len(rutas), n, replace=False))
    return [rutas[i] for i in idx], [etiquetas[i] for i in idx]


def listas_de_conjuntos(prueba):
    """{conjunto: (rutas, etiquetas, transform)}. En OOD la etiqueta es -1 (convención OpenOOD).
    En la prueba rápida, train y test ID van completos (test para reproducir la exactitud) y el
    resto se reduce a N_PRUEBA imágenes."""
    conjuntos, clases = {}, None
    for nombre, ruta in ID_DIRS.items():
        folder = datasets.ImageFolder(ruta)          # clases y ficheros en orden alfabético
        clases = folder.classes
        rutas, etiquetas = [s[0] for s in folder.samples], [s[1] for s in folder.samples]
        if prueba and nombre == 'id_val':
            rutas, etiquetas = submuestra(rutas, etiquetas, N_PRUEBA)
        conjuntos[nombre] = (rutas, etiquetas, transform)

    rutas, etiquetas, _ = conjuntos['id_test']
    if prueba:
        rutas, etiquetas = submuestra(rutas, etiquetas, N_PRUEBA)
    conjuntos[CONTROL_NAME] = (rutas, etiquetas, transform_control)

    for nombre, ruta in OOD_DIRS.items():
        rutas = listar_carpeta(ruta)
        rutas, etiquetas = submuestra(rutas, [-1] * len(rutas), N_PRUEBA if prueba else None)
        conjuntos[nombre] = (rutas, etiquetas, transform)
    return conjuntos, clases


class ImagenesDataset(Dataset):
    def __init__(self, rutas, etiquetas, transform):
        self.rutas, self.etiquetas, self.transform = rutas, etiquetas, transform

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, idx):
        img = Image.open(self.rutas[idx]).convert('RGB')
        return {'data': self.transform(img), 'label': self.etiquetas[idx]}


class FeaturesDataset(Dataset):
    """Features exportadas con la interfaz de batch de OpenOOD: {'data': features, 'label': y}."""

    def __init__(self, feats, etiquetas):
        self.feats = torch.from_numpy(feats)
        self.etiquetas = torch.from_numpy(etiquetas)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return {'data': self.feats[idx], 'label': self.etiquetas[idx]}


def make_loader(dataset, num_workers=NUM_WORKERS):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# 5. ETAPA 1: EXPORTACIÓN DE FEATURES
# ---------------------------------------------------------------------------

@torch.no_grad()
def extraer(net, loader):
    feats, logits, labels = [], [], []
    for batch in loader:
        lo, fe = net(batch['data'].to(DEVICE), return_feature=True)
        feats.append(fe.float().cpu())
        logits.append(lo.float().cpu())
        labels.append(batch['label'])
    feats, logits = torch.cat(feats).numpy(), torch.cat(logits).numpy()
    return feats, logits, logits.argmax(1), torch.cat(labels).numpy().astype(np.int64)


def guardar_conjunto(carpeta, nombre, feats, logits, pred, label, rutas):
    np.savez(os.path.join(carpeta, f'{nombre}.npz'), feats=feats.astype(np.float32),
             logits=logits.astype(np.float32), pred=pred.astype(np.int64),
             label=label.astype(np.int64), paths=np.array(rutas, dtype=str))
    with open(os.path.join(carpeta, 'listas', f'{nombre}.txt'), 'w') as f:
        f.write('\n'.join(rutas) + '\n')


def exportar(carpeta_feats, prueba, net):
    os.makedirs(os.path.join(carpeta_feats, 'listas'), exist_ok=True)
    conjuntos, clases = listas_de_conjuntos(prueba)
    tiempos, tamanos = {}, {}
    for nombre, (rutas, etiquetas, tf) in conjuntos.items():
        inicio = time.perf_counter()
        feats, logits, pred, label = extraer(net, make_loader(ImagenesDataset(rutas, etiquetas, tf)))
        guardar_conjunto(carpeta_feats, nombre, feats, logits, pred, label, rutas)
        tiempos[nombre] = time.perf_counter() - inicio
        tamanos[nombre] = len(rutas)
        print(f'  exportado {nombre}: {len(rutas)} imágenes en {tiempos[nombre]:.0f} s')

    # Uniones: se construyen con las features ya calculadas, ordenadas por ruta
    for nombre, partes in UNIONES.items():
        datos = [np.load(os.path.join(carpeta_feats, f'{p}.npz')) for p in partes]
        juntos = {k: np.concatenate([d[k] for d in datos]) for k in ['feats', 'logits', 'pred', 'label', 'paths']}
        orden = np.argsort(juntos['paths'], kind='stable')
        juntos = {k: v[orden] for k, v in juntos.items()}
        guardar_conjunto(carpeta_feats, nombre, juntos['feats'], juntos['logits'], juntos['pred'],
                         juntos['label'], list(juntos['paths']))
        tamanos[nombre] = len(orden)
        print(f'  unión {nombre} = {" + ".join(partes)}: {len(orden)} imágenes')

    info = {'clases_id': clases, 'num_imagenes': tamanos, 'tiempos_exportacion_s': tiempos,
            'prueba': prueba, 'n_prueba': N_PRUEBA if prueba else None, 'semilla_submuestra': SEMILLA,
            'modelo': MODEL_NAME, 'checkpoint': CHECKPOINT_PATH,
            'descripcion': 'feats = salida de SwinWrapper.get_features (768, pooling tras la LayerNorm '
                           'final, antes de la capa lineal); logits = capa lineal; pred = argmax(logits); '
                           'label = clase real en ID y control_res_128, -1 en OOD; paths = ruta de cada imagen.'}
    with open(os.path.join(carpeta_feats, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


# ---------------------------------------------------------------------------
# 6. ETAPA 2: EVALUACIÓN CON LOS POSTPROCESSORS
# ---------------------------------------------------------------------------

def build_postprocessor(name):
    # Variantes de SHE con otra métrica: 'she_cosine', 'she_euclidean' (por defecto inner_product)
    base, _, metrica = name.partition('_')
    if base != 'she':
        base, metrica = name, ''
    config = Config(os.path.join(CONFIG_ROOT, 'postprocessors', f'{base}.yml'))
    config = merge_configs(config, Config(**{'dataset': {'name': ID_DATA_NAME}}))
    # Sin búsqueda automática de hiperparámetros (APS): se usan los valores por defecto del .yml
    config.postprocessor.APS_mode = False
    if metrica:
        config.postprocessor.postprocessor_args.metric = metrica
    return get_postprocessor(config), config


def cargar_features(carpeta_feats, nombres):
    return {n: dict(np.load(os.path.join(carpeta_feats, f'{n}.npz'))) for n in nombres}


def estadisticas_s1_ash(feats_val, percentil):
    """Distribución de s1 = suma de las 768 features (lo que ash_b reparte entre las top-k)."""
    s1 = feats_val.sum(1)
    n = feats_val.shape[1]
    k = n - int(np.round(n * percentil / 100.0))
    q = np.percentile(s1, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {'percentil': percentil, 'k_features_conservadas': k, 'n_features': n,
            'fraccion_features_negativas_val': float(np.mean(feats_val < 0)),
            's1_media': float(s1.mean()), 's1_std': float(s1.std()),
            's1_fraccion_negativa': float(np.mean(s1 < 0)),
            's1_cuantiles_0_1_5_25_50_75_95_99_100': [float(v) for v in q],
            'valor_relleno_s1_entre_k_mediana': float(np.median(s1) / k)}


def hiperparametros(config):
    args = config.postprocessor.get('postprocessor_args')
    return {k: v for k, v in dict(args).items()} if args else {}


def evaluar(carpeta_feats, carpeta_res, metodos, verificar, info_feats):
    nombres_id = ['id_train', 'id_val', 'id_test']
    nombres_ood_disco = [n for n in CONJUNTOS_OOD]
    datos = cargar_features(carpeta_feats, nombres_id + nombres_ood_disco)
    clases = info_feats['clases_id']

    # Capa lineal del Swin para CachedSwin
    swin_cpu = SwinWrapper(NUM_CLASSES, CHECKPOINT_PATH, MODEL_NAME)
    fc = swin_cpu.backbone.get_classifier()
    redes = {'cache': CachedSwin(fc).to(DEVICE).eval(),
             'ash': CachedSwinASH(fc).to(DEVICE).eval()}
    del swin_cpu

    def loader_feats(nombre):
        d = datos[nombre]
        return make_loader(FeaturesDataset(d['feats'], d['label']), num_workers=0)

    id_loader_dict = {'train': loader_feats('id_train'), 'val': loader_feats('id_val'),
                      'test': loader_feats('id_test')}

    registro = {'metodos': {}, 'verificacion_metricas_vs_openood_max_dif': 0.0}
    filas_principal, filas_anexo, filas_nct = [], [], []

    for metodo in metodos:
        print(f'\n################ {metodo} ################')
        inicio = time.perf_counter()
        postprocessor, config = build_postprocessor(metodo)
        net = redes['ash' if metodo == 'ash' else 'cache']
        hp = hiperparametros(config)
        print(f'Hiperparámetros: {hp}  (APS_mode={config.postprocessor.APS_mode})')

        # Ajuste solo con datos ID (train / val); ningún dato OOD
        postprocessor.setup(net, id_loader_dict, None)
        t_setup = time.perf_counter() - inicio

        puntuaciones = {}
        for nombre in ['id_val', 'id_test'] + CONJUNTOS_OOD:
            pred, conf, label = postprocessor.inference(net, loader_feats(nombre), progress=False)
            puntuaciones[nombre] = {'conf': conf.astype(np.float64), 'pred': pred, 'label': label,
                                    'paths': datos[nombre]['paths']}

        idt = puntuaciones['id_test']
        acc_id = metricas_ood.exactitud(idt['pred'], idt['label'])
        for nombre in CONJUNTOS_OOD:
            ood = puntuaciones[nombre]
            m = metricas_ood.todas_las_metricas(idt['conf'], ood['conf'], idt['pred'], idt['label'], NUM_CLASSES)

            # Comprobación: AUROC, FPR@95 y AUPR iguales a los de OpenOOD
            ref = compute_all_metrics(np.concatenate([idt['conf'], ood['conf']]),
                                      np.concatenate([idt['label'], -np.ones(len(ood['conf']), int)]),
                                      np.concatenate([idt['pred'], ood['pred']]))
            dif = max(abs(ref[0] - m['FPR@95']), abs(ref[1] - m['AUROC']),
                      abs(ref[2] - m['AUPR_In']), abs(ref[3] - m['AUPR_Out']))
            registro['verificacion_metricas_vs_openood_max_dif'] = max(
                registro['verificacion_metricas_vs_openood_max_dif'], float(dif))

            acc_conj = (metricas_ood.exactitud(ood['pred'], ood['label'])
                        if nombre == CONTROL_NAME else np.nan)
            base = {'metodo': metodo, 'conjunto': nombre, 'N_ID': len(idt['conf']), 'N_OOD': len(ood['conf'])}
            fila = {**base, 'AUROC': m['AUROC'], 'FPR@95': m['FPR@95'], 'CCR@FPR5': m['CCR@FPR5'],
                    'ACC_ID': acc_id}
            if nombre in CONJUNTOS_PRINCIPALES:
                filas_principal.append({**fila, 'ACC_control_res_128': acc_conj})
            filas_anexo.append({**fila, 'AUPR_In': m['AUPR_In'], 'AUPR_Out': m['AUPR_Out'],
                                'FPR_real_CCR': m['FPR_real_CCR'], 'tau_CCR': m['tau_CCR'],
                                **{f'CCR@FPR5_{c}': v for c, v in zip(clases, m['CCR@FPR5_por_clase'])}})
            if nombre in CONJUNTOS_NCT_POR_CLASE:
                filas_nct.append({**base, 'clase_nct': nombre.split('_')[-1].upper(),
                                  'papel': nombre.split('_')[0], 'AUROC': m['AUROC'],
                                  'FPR@95': m['FPR@95'], 'CCR@FPR5': m['CCR@FPR5'],
                                  'AUPR_In': m['AUPR_In'], 'AUPR_Out': m['AUPR_Out']})
            print(f'{metodo:>14} - {nombre:<18} AUROC={m["AUROC"]:.4f}  FPR@95={m["FPR@95"]:.4f}  '
                  f'CCR@FPR5={m["CCR@FPR5"]:.4f}'
                  + (f'  ACC_control={acc_conj:.4f}' if nombre == CONTROL_NAME else ''))

        # Puntuaciones de todos los conjuntos (incluida la validación ID) y rutas
        np.savez(os.path.join(carpeta_res, f'scores_{metodo}.npz'),
                 **{f'{n}_{k}': v for n, d in puntuaciones.items() for k, v in d.items()})

        extra = {}
        if metodo == 'react':
            extra['umbral_react'] = float(postprocessor.threshold)
        if metodo == 'ash':
            extra['s1_val_id'] = estadisticas_s1_ash(datos['id_val']['feats'], postprocessor.percentile)
        if metodo == 'dice':
            extra['umbral_contribucion_dice'] = float(postprocessor.thresh)
        registro['metodos'][metodo] = {'hiperparametros': hp, 'APS_mode': False,
                                       'acc_id_test': acc_id, 'tiempo_setup_s': t_setup,
                                       'tiempo_total_s': time.perf_counter() - inicio, **extra}
        print(f'ACC ID test ({metodo}) = {acc_id:.4f}   tiempo = {time.perf_counter() - inicio:.0f} s')
        if extra:
            print(f'Extra: {extra}')

        # Guardado incremental
        pd.DataFrame(filas_principal).to_csv(os.path.join(carpeta_res, 'resultados_principales.csv'), index=False)
        pd.DataFrame(filas_anexo).to_csv(os.path.join(carpeta_res, 'resultados_anexo.csv'), index=False)
        pd.DataFrame(filas_nct).to_csv(os.path.join(carpeta_res, 'resultados_nct_por_clase.csv'), index=False)
        del postprocessor
        torch.cuda.empty_cache()

    if verificar and 'msp' in metodos:
        registro['verificacion_cache'] = verificar_msp_directo(carpeta_res, datos)

    df = pd.DataFrame(filas_principal)
    print('\n================ AUROC (método x conjunto) ================')
    print(df.pivot(index='metodo', columns='conjunto', values='AUROC')
            .reindex(index=metodos, columns=CONJUNTOS_PRINCIPALES).to_string(float_format='%.4f'))
    print('\n================ FPR@95 ================')
    print(df.pivot(index='metodo', columns='conjunto', values='FPR@95')
            .reindex(index=metodos, columns=CONJUNTOS_PRINCIPALES).to_string(float_format='%.4f'))
    return registro


def verificar_msp_directo(carpeta_res, datos):
    """MSP directo sobre las imágenes (SwinWrapper) frente a MSP con features en caché."""
    print('\n################ Verificación: MSP directo vs caché ################')
    net = SwinWrapper(NUM_CLASSES, CHECKPOINT_PATH, MODEL_NAME).to(DEVICE).eval()
    postprocessor, _ = build_postprocessor('msp')
    cache = np.load(os.path.join(carpeta_res, 'scores_msp.npz'))
    resultado = {}
    for nombre in ['id_test', 'cov_panda']:
        rutas = list(datos[nombre]['paths'])
        loader = make_loader(ImagenesDataset(rutas, list(datos[nombre]['label']), transform))
        pred, conf, _ = postprocessor.inference(net, loader, progress=False)
        dif = float(np.max(np.abs(conf.astype(np.float64) - cache[f'{nombre}_conf'])))
        iguales = bool(np.array_equal(conf.astype(np.float64), cache[f'{nombre}_conf']))
        preds_iguales = bool(np.array_equal(pred, cache[f'{nombre}_pred']))
        resultado[nombre] = {'max_dif_conf': dif, 'conf_identicas_bit_a_bit': iguales,
                             'pred_identicas': preds_iguales, 'n': len(rutas)}
        print(f'{nombre}: max|dif conf| = {dif:.3e}  idénticas={iguales}  pred idénticas={preds_iguales}')
    return resultado


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def versiones():
    try:
        commit = subprocess.check_output(['git', '-C', RAIZ, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        commit = None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__,
            'numpy': np.__version__, 'sklearn': sklearn.__version__, 'pandas': pd.__version__,
            'openood_commit': commit, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prueba', action='store_true', help=f'~{N_PRUEBA} imágenes por conjunto')
    parser.add_argument('--etapa', choices=['todo', 'exportar', 'evaluar'], default='todo')
    parser.add_argument('--metodos', nargs='+', default=POSTPROCESSORS)
    parser.add_argument('--verificar', action='store_true', help='MSP directo vs caché')
    parser.add_argument('--features_tag', default=None, help='subcarpeta de features/ (por defecto prueba_500 o completo)')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit('No hay GPU disponible: lanza este script con SLURM (sbatch script_ood.sh)')
    print(f'GPU detectada: {torch.cuda.get_device_name(0)}')
    torch.manual_seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tag = args.features_tag or (f'prueba_{N_PRUEBA}' if args.prueba else 'completo')
    carpeta_feats = os.path.join(FEATURES_ROOT, tag)
    inicio_total = time.perf_counter()

    if args.etapa in ('todo', 'exportar'):
        print(f'Exportando features en {carpeta_feats}')
        net = SwinWrapper(NUM_CLASSES, CHECKPOINT_PATH, MODEL_NAME).to(DEVICE).eval()
        info_feats = exportar(carpeta_feats, args.prueba, net)
        del net
        torch.cuda.empty_cache()
        if args.etapa == 'exportar':
            return
    with open(os.path.join(carpeta_feats, 'info.json')) as f:
        info_feats = json.load(f)

    sufijo = '_prueba' if args.prueba else ''
    carpeta_res = os.path.join(RESULTS_ROOT, f'faseA_{datetime.now():%Y-%m-%d_%H-%M-%S}{sufijo}')
    os.makedirs(carpeta_res, exist_ok=True)
    print(f'Resultados en: {carpeta_res}')
    for nombre, n in info_feats['num_imagenes'].items():
        print(f'  {nombre}: {n} imágenes')

    registro = evaluar(carpeta_feats, carpeta_res, args.metodos, args.verificar, info_feats)

    config = {
        'fase': 'A', 'entorno': sys.prefix, 'versiones': versiones(), 'argumentos': vars(args),
        'carpeta_features': carpeta_feats, 'semilla': SEMILLA, 'batch_size': BATCH_SIZE,
        'num_imagenes': info_feats['num_imagenes'], 'clases_id': info_feats['clases_id'],
        'tiempos_exportacion_s': info_feats['tiempos_exportacion_s'],
        'tiempo_total_s': time.perf_counter() - inicio_total,
        'convencion': 'Clase positiva = OOD (openood/evaluators/metrics.py:41-48). conf: más alto = más ID. '
                      'CCR@FPR5: tau con el 5% de OOD aceptado (conf >= tau); CCR = #(ID conf>=tau y '
                      'bien clasificada)/#ID.',
        'notas': {
            'ash': 'ash_b (percentil por defecto) sobre las 768 features agregadas, como ASHNet de OpenOOD '
                   '(no sobre tokens antes del pooling). Las features salen de una LayerNorm y pueden ser '
                   'negativas; ash_b rellena las top-k con s1/k (s1 = suma de las features), ver '
                   'metodos.ash.s1_val_id.',
            'dice_react': 'Omitido: OpenOOD no implementa DICE+ReAct (DICEPostprocessor no llama a '
                          'forward_threshold).',
            'cache': 'Todos los métodos se evalúan sobre las features exportadas (CachedSwin aplica solo '
                     'la capa lineal); verificación con MSP directo en verificacion_cache si --verificar.',
            'no_usados': 'Métodos basados en gradientes (ODIN, GradNorm) y generativos excluidos.',
            'ajuste': 'Ningún dato OOD se usa para ajustar; APS_mode=False; hiperparámetros por defecto.',
        },
        **registro,
    }
    with open(os.path.join(carpeta_res, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=float)
    print(f'\nTiempo total: {(time.perf_counter() - inicio_total) / 60:.1f} min')


if __name__ == '__main__':
    main()
