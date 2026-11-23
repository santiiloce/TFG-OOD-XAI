# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
##### IMPORTS ######
import os # To count files in directories
from datetime import datetime # Obtener fecha actual
import sys # Para poder escribir en archivo 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import os
import math


work_folder = os.getcwd()
base = '/mnt/homeGPU/slcerro/tfg/TrainTest'
path_train = base + '/train'
path_val = base + '/val'
path_test = base + '/test'

etiquetas = np.array([
    "Adenocarcinoma_ductal",
    "Glandulas_no_neoplasicas",
    "Gleason_4_fusionadas",
    "Gleason_5_comedonecrosis",
    "No_Cancer",
    "Carcinoma_intraductal",
    "Gleason_3",
    "Gleason_4_glomeruloide",
    "Gleason_5_no_glandulas",
    "PIN_de_alto_grado",
    "Estroma",
    "Gleason_4_cribiformes",
    "Gleason_4_pobremente_formadas",
    "Inflamatorio"
], dtype='U')

# Funcion que devuelve un array de numpy con tuplas de (nombre, cantidad) de imágenes por cada tipo de categoria
def GetQuantity(folder):
   datos_lista = []
   for etiqueta in etiquetas:
        new_path = os.path.join(folder, etiqueta)
        
        # Filtrado y conteo de archivos
        archivos = [nombre for nombre in os.listdir(new_path) if os.path.isfile(os.path.join(new_path, nombre))]
        total_archivos = len(archivos)
        
        # Almacenar como sublista
        datos_lista.append([etiqueta, total_archivos])
  # 2. Convertir a Array de NumPy especificando tipo objeto si desea mantener tipos mixtos o simplemente trabajar con la estructura bidimensional
   return np.array(datos_lista) 

# Operacion interna
def GetLabels(folder):
   sizes = GetQuantity(folder)
   labels = np.array([])
   for i in range(len(sizes)):
    labels = np.append(labels,sizes[i][0])
   return labels

# Operacion interna
def GetSizes(folder):
   train_sizes = GetQuantity(folder)
   sizes = np.array([])
   for i in range(len(train_sizes)):
    sizes = np.append(sizes,train_sizes[i][1])
   return sizes

def GetTotalImages(folder):
   total_images = 0 
   sizes = GetSizes(folder)
   for i in sizes:
    total_images = float(i) + total_images
   return total_images

def GetProportions(folder):
   porcentajes = np.array([])
   total_images = GetTotalImages(folder)
   sizes = GetSizes(folder)
   for i in sizes: 
     porcentajes = np.append(porcentajes, float(i)/total_images)
   return porcentajes
   
def PlotNumberOfImages(folder, name):
   labels = GetLabels(folder)
   sizes = GetSizes(folder)
   plt.pie(sizes, labels= labels)
   plt.savefig(name)









