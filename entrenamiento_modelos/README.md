# Entrenamiento de los modelos

Entrenamiento de los cinco clasificadores (ViT-B/16, ResNet-50, ConvNeXt-S, EfficientNetV2-S y Swin-S) sobre las 14 clases del conjunto ID, y su evaluación sobre el mismo test ID.

## Ficheros

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `entrenamiento/clasificador.py` | Entrena ViT-B/16, ResNet-50, ConvNeXt-S y EfficientNetV2-S con `vision_learner` (`ps = 0.5`), `lr_find` (*valley*) y `fine_tune` (10 épocas congeladas y hasta 90 descongeladas), y los evalúa en el test. | «Materiales y métodos → Protocolo de entrenamiento»; «Clasificación ID» | Un argumento de 0 a 3 que elige el modelo. Datos en `funciones_aux.base`, con `train/`, `val/` y `test/` y una subcarpeta por clase. Escribe en la carpeta actual `param<k>_<fecha>/`: el learner exportado (`*_model.pkl`), los pesos, `resultados.txt`, `training_history.csv` y las matrices de confusión. |
| `entrenamiento/clasificador_svim.py` | Entrena el Swin-S (`swin_small_patch4_window7_224` de timm) con `Learner` y `fit_one_cycle(25, lr_max=1e-4)`, sin congelación previa. | «Materiales y métodos → Protocolo de entrenamiento»; «Clasificación ID» | Las mismas que `clasificador.py`. El modelo lo fija el código (`parametro = 2`); el argumento solo da nombre a la carpeta (`param4` en la ejecución de la memoria). |
| `entrenamiento/funciones_aux.py` | Rutas de los datos, las 14 etiquetas, recuento de imágenes por clase y gráfico circular de la distribución. Lo importan los dos scripts anteriores. | «Materiales y métodos → Partición e integridad»; «Materiales y métodos → Protocolo de entrenamiento» | La ruta `base` a la carpeta de datos. |
| `evaluacion/fase1_predicciones.py` | Predicciones de los cinco modelos sobre el mismo test ID, cargando cada learner exportado y usando su propio preprocesado de validación. | «Clasificación ID → Comparación de los cinco modelos» | Los `*_model.pkl` de `EJECUCIONES/param*/` y el orden de imágenes de `features/completo/id_test.npz` (lo genera `OOD/openood/my_evauator_openood.py`). Escribe `results_fase1/predicciones_<modelo>.npz`. Necesita GPU. |
| `evaluacion/fase1_metricas.py` | Exactitud, F1 macro, *balanced accuracy* y sensibilidad por clase, con IC por *bootstrap*; test de McNemar Swin-S frente a ConvNeXt-S; *bootstrap* por lámina como análisis de sensibilidad. | «Clasificación ID → Comparación de los cinco modelos»; «Clasificación ID → Elección del modelo» | `results_fase1/predicciones_<modelo>.npz`. Solo CPU. |

## Cambios respecto al código ejecutado

En `clasificador.py` y `clasificador_svim.py` se ha quitado la función `plot_metrics`, que dibujaba las curvas de entrenamiento (`Train_Validation_Results.jpg`). Se ha quitado también su llamada y la línea que construía el nombre de esa figura. La función era código de terceros (Ignacio Oguiza, foro de fast.ai) sin licencia conocida. El entrenamiento no cambia. Las curvas de la memoria se dibujaron con esa función.

## Ejecución

- Entrenamiento: entorno `fastai_env` (ver el README de la raíz), ejecutado desde `entrenamiento/` para que `import funciones_aux` funcione.
- Evaluación: entorno `fastai_env_ood`. `fase1_predicciones.py` añade la carpeta de las ejecuciones al `sys.path` por si el learner exportado necesita módulos del entrenamiento.

## Fuentes

- **fastai**: Howard, J. y Gugger, S. (2020). *fastai: A Layered API for Deep Learning*. arXiv:2002.04688. <https://github.com/fastai/fastai> (licencia Apache-2.0).
- **timm** (modelos preentrenados): Wightman, R. (2019). *PyTorch Image Models*. doi:10.5281/zenodo.4414861. <https://github.com/huggingface/pytorch-image-models> (licencia Apache-2.0).
