# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
# Trabajo de fin de Grado realizado por Santiago Lopez Cerro
import funciones_aux
import fastai
import fastcore
import timm
import sys
import time
import matplotlib.pyplot as plt 
print(f'fastcore version {fastcore.__version__} installed')
print(f'fastai version {fastai.__version__} installed')
from fastai.vision.all import *
from fastai.vision.all import *
from pathlib import Path 
import torch
torch.cuda.empty_cache() # Libero cache al iniciar la ejecución del codigo
torch.backends.cudnn.benchmark = True
torch.device('cuda') 

# Realizo el mismo estudio por cada tipo de clase
"""
Funcion de entrenamiento y testing para el modelo de clasificación de imágenes en 14 categorías.
Inputs: arquitecture -> modelo preentrenado que usaremos como punto de partida para hacer el fine tuning
        output -> tamaño de las imágenes de entreda requerido por arquitecture
        folder -> directorio donde guardar las imágenes y gráficas elaboradas durante la ejecución
        fichero -> archivo donde ir guardando los resultados del programaccls
"""
def training(arquitecture, output, folder, print_file):
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}",file = print_file)
    else:
        print("Usando CPU", file = print_file) 

    workers_disponibles = 4 # 4 
    batch_size = 64 # Defino el mismo tamaño train y test

    # Creo el DataBLock
    UGR_2026 = DataBlock(
            blocks = (ImageBlock, CategoryBlock),
            get_items = get_image_files,
            splitter = GrandparentSplitter(train_name= 'train', valid_name= 'val'),
            get_y = parent_label,
            item_tfms = [Resize(int(output), method = 'squish')],
            batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    ).dataloaders(funciones_aux.base, bs = batch_size, num_workers= workers_disponibles)
    csv_file_name = os.path.join(folder, 'training_history.csv')


    # 1. Creamos el modelo Transformer puro indicando que queremos 14 clases
    # dls.c nos da automáticamente el número de clases (14 en tu caso)
    modelo_timm = timm.create_model(arquitecture, pretrained=True, num_classes=UGR_2026.c)

    # 2. Usamos Learner genérico en lugar de vision_learner
    learn12 = Learner(dls=UGR_2026,
                    model=modelo_timm,
                    loss_func=LabelSmoothingCrossEntropy(),
                    metrics=[accuracy, F1Score(average='macro')],
                    opt_func=Adam,
                    wd=0.1,
                    cbs=[
                        EarlyStoppingCallback(monitor='valid_loss', min_delta=0.005, patience=10)
                    ]).to_fp16()

    # Como es un Learner genérico, el fine_tune no sabe qué capas congelar.
    # Por tanto entrenaremos el  modelo de golpe usando fit_one_cycle directamente

    inicio = time.perf_counter()
    learn12.fit_one_cycle(25, lr_max=1e-4) # Ajusta los epochs
    fin = time.perf_counter()
    transcurrido = fin - inicio
    print(f"Tiempo transcurrido para el entrenamiento: {transcurrido/60} minutos", file = print_file)

    ruta_pesos = os.path.join(nombre_carpeta, f"{arquitecture}_pesos_finales")
    learn12.save(ruta_pesos)

    # 4. Interpretación de resultados (Matriz de Confusión y Errores)
    # Usamos el set de validación (ds_idx=1) para ver cómo generaliza
    interp = ClassificationInterpretation.from_learner(learn12)
    
    # --- Guardar Matriz de Confusión ---
    interp.plot_confusion_matrix(figsize=(12, 12))
    plt.title(f'Matriz de Confusión: {arquitecture}')
    plt.savefig(os.path.join(folder, f"Confusion_Matrix_{arquitecture}.jpg"))
    plt.close()

    # --- Guardar Top Losses (Top 10 errores) ---
    interp.plot_top_losses(10, nrows=2, figsize=(30, 10))
    plt.savefig(os.path.join(folder, f"Top_Losses_{arquitecture}.jpg"))
    plt.close()

    # --- Guardar los "Most Confused" en el log ---
    print(f"\nClases más confundidas ({arquitecture}):",file = print_file)
    print(interp.most_confused(min_val=5),file = print_file)

    learn12.remove_cb(CSVLogger)

    # 5. Exportar el modelo para no perderlo nunca
    learn12.export(os.path.join(folder, f"{arquitecture}_model.pkl"))

    ### Realizo test sobre el modelo
    path_test = Path(funciones_aux.path_test)
    test_files = get_image_files(path_test)

    # Crear el DataLoader de test usando la "plantilla" del learner
    tst_dl = learn12.dls.test_dl(test_files, with_labels=True, bs = batch_size, num_workers = workers_disponibles)
    results = learn12.validate(dl=tst_dl)
    print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}",file = print_file)

    print(learn12.summary() ,file = print_file)

    # 2. Le decimos a FastAI que interprete ese DataLoader específico
    interp_test = ClassificationInterpretation.from_learner(learn12, dl=tst_dl)
    
    # 3. Dibujamos y guardamos
    interp_test.plot_confusion_matrix(figsize=(12,12))
    plt.title(f'Matriz de Confusión TEST: {arquitecture}')
    plt.savefig(os.path.join(folder, f"Confusion_Matrix_TEST_{arquitecture}.jpg"))
    plt.close()
    
    # 4. También podemos ver los errores más graves del test en el log
    print(f"\nErrores más comunes en TEST ({arquitecture}):",file = print_file)
    print(interp_test.most_confused(min_val=1),file = print_file)

    # Liberar memoria
    del learn12
    torch.cuda.empty_cache()
    
    return {
        "modelo": arquitecture,
        "test_loss": results[0],
        "test_accuracy": results[1]
    }

# Ejemplo: python script.py hola
if len(sys.argv) > 1: 
    parametro = int(sys.argv[1])


#### GENERAR GRAFICA, EXPORTAR RESULTADOS A ARCHIVO
# Los nombres en timm suelen llevar prefijos del autor o la arquitectura específica
models_to_train = np.array([
    "vit_base_patch16_224",     
    "resnet50",                  
    "swin_small_patch4_window7_224",
    "tf_efficientnetv2_s",
    "swin_small_patch4_window7_224",        
])
outputs_to_train = np.array([224, 224,224, 224]) 

# Para resnet lo importante es que sean divisibles entre 32
# Genero los archivos con un formato más seguro

# 1. Obtener la fecha y hora actual
ahora = datetime.now()

# 2. Formatear la fecha y hora para usarla como nombre de carpeta
solo_nombre_carpeta= ahora.strftime("%Y-%m-%d_%H-%M-%S")

# 3. Crear la carpeta
# exist_ok=True evita que el programa falle si la carpeta ya existe por alguna razón

print(f"Carpeta para esta ejecución creada: {solo_nombre_carpeta}")
solo_nombre_carpeta = "param" + str(parametro) + "_" + solo_nombre_carpeta
nombre_carpeta = os.path.join(funciones_aux.work_folder,solo_nombre_carpeta )

parametro = 2 # Solo me interesaba el valor para el nombre de la carpeta, para el resto el dos es mejor

os.makedirs(nombre_carpeta, exist_ok=True)

# Representacion de los tamaños de cada conjunto de datos
train_images_name = os.path.join(nombre_carpeta, "distribution_train.jpg")
val_images_name = os.path.join(nombre_carpeta, "distribution_val.jpg")
test_images_name = os.path.join(nombre_carpeta, "distribution_test.jpg")
funciones_aux.PlotNumberOfImages(funciones_aux.path_train, train_images_name)
funciones_aux.PlotNumberOfImages(funciones_aux.path_val, val_images_name)
funciones_aux.PlotNumberOfImages(funciones_aux.path_test, test_images_name)
log_file = os.path.join(nombre_carpeta,"resultados.txt" )


with open(log_file, 'w') as f: 
    separador = "#############################---" + models_to_train[parametro] + "---######################################" 
    print(separador, file = f)
    result = training(models_to_train[parametro], outputs_to_train[parametro], nombre_carpeta, f)
    print(result, file = f)
    