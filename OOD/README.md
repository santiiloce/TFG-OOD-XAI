# Detección OOD

Evaluación de detectores OOD *post hoc* sobre el Swin-S: 13 métodos de OpenOOD, una implementación propia de STOOD-X, corrupciones sintéticas y la repetición con ConvNeXt-S.

## Ficheros

### `openood/`: benchmark OpenOOD sobre el Swin-S

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `my_evauator_openood.py` | Fase A. (1) Pasa cada conjunto (ID, NCT-CRC, PANDA, control de 128 px, DTD) una vez por el Swin-S y guarda características de 768 dimensiones, logits, predicciones y rutas en `features/<tag>/`. (2) Ejecuta los postprocesadores de OpenOOD sobre esas características (`CachedSwin`, que reproduce la interfaz del `SwinWrapper` que carga el Swin-S). | «Detección OOD»: secciones «Near-OOD principal», «Control interno: centro frente a tejido», «PANDA: cambio de centro y de escala», «Far-OOD», «Métodos basados en logits y en modificación de activaciones» e «Hipótesis gaussiana de MDS» | Pesos del Swin-S (`CHECKPOINT_PATH`), las carpetas de datos ID y OOD y la carpeta `configs/` de OpenOOD (`RAIZ`). Hay que ejecutarlo con el paquete `openood` importable. Necesita GPU. |
| `metricas_ood.py` | AUROC, FPR@95 y CCR@FPR5 (OOD como clase positiva), AUPR e IC por *bootstrap* (también pareado). Lo usan casi todos los scripts de esta carpeta. | «Formalización de la detección OOD y métricas»; «Detección OOD» | — |
| `comparar_resultados.py` | Fase C. Une las fases A (OpenOOD) y B (STOOD-X): tabla principal con IC, *bootstrap* pareado RMDS − método y tabla por clase de NCT-CRC. También dibuja figuras. | «Detección OOD → Discusión» (tabla de resultados globales) | `--faseA` y `--faseB`, carpetas con `scores_<método>.npz`. |

### `stoodx/`: implementación propia de STOOD-X

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `stoodx_paper.py` | STOOD-X implementado a partir del texto del artículo, sin el repositorio oficial. Para cada imagen se hace un único test de Mann-Whitney entre las distancias coseno a sus K vecinos de entrenamiento de la clase predicha y las distancias de esos vecinos a los suyos. La puntuación es −z. Modos: `validacion` y `analisis_validacion` (barrido de K), `seleccion_k` (criterio *leave-one-class-out* con la clase ID retenida, que da K = 144), `completo` y `combinar`. | «Detección OOD → Implementación de STOOD-X» (selección de K y sensibilidad a K) | Características de la fase A (`features/<tag>/<conjunto>.npz`); `--modo`, `--k`, `--config`, `--salida`. Usa la GPU para las distancias. |

### `corrupciones/`: *covariate shift* sintético

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `corrupciones_zhang.py` | Llama a `utils.Distortions` del banco de Zhang et al. sin modificarlo: sustituye `visdom` y `wandb` por módulos vacíos, carga los recursos de 224 px y fija una semilla determinista por imagen, corrupción y nivel (4 primeros bytes del SHA-256). | «Materiales y métodos → Conjuntos OOD → Corrupciones sintéticas» | El clon del repositorio de Zhang et al. en `ZHANG` (no se distribuye; ver «Fuentes»). |
| `parametros_corrupciones.py` | Lee con `ast` los parámetros de cada corrupción y nivel directamente del `utils.py` de Zhang et al. (commit `2b288c1`, comprobado con `git`) y los tamaños de `blots.npy`. | «Materiales y métodos → Conjuntos OOD → Corrupciones sintéticas» | El clon de Zhang et al. en ese commit. |
| `fase8_extraer.py` | Corrompe el test ID (9 corrupciones × niveles 1, 3 y 5), extrae características y logits del Swin-S en CPU y puntúa todos los detectores con las referencias de la fase A. | «Materiales y métodos → Conjuntos OOD → Corrupciones sintéticas»; «Detección OOD → Corrupciones sintéticas: *covariate shift*» | `--modelo swin`, `--n`, `--salida`. Importa `my_evauator_openood` y `stoodx_paper`. |
| `fase8_analizar.py` | Comprobaciones, exactitud y métricas por celda, CEC y saturación de STOOD-X. | «Detección OOD → Corrupciones sintéticas: *covariate shift*» (subsecciones «Exactitud del clasificador ante las corrupciones», «¿Detectan la alteración? AUROC y FPR@95» y «¿Gradúan la severidad? CEC»); «Material complementario → Corrupciones sintéticas» | La salida de `fase8_extraer.py` y las referencias congeladas de las fases A y B. **Importa `fase8_figuras`, que no se distribuye: hay que ejecutarlo con `--sin_figuras`.** |
| `fase8b.py` | ¿Rechazan los detectores las imágenes corruptas mal clasificadas? Umbral fijo calculado con `id_val`, AUROC de error y AURC. | «Detección OOD → Corrupciones sintéticas: *covariate shift* → Saturación de STOOD-X» | Celdas de la fase 8 y puntuaciones de `id_val` de las fases A y B. **Al final importa `fase8b_figuras`, que no se distribuye: el script falla en esa línea después de haber guardado todos los CSV.** |
| `fase8b_empates_reales.py` | Variante de STOOD-X con las imágenes saturadas tratadas como empate real, con IC por *bootstrap*. Reutiliza `fase8b.py`. | «Detección OOD → Corrupciones sintéticas: *covariate shift* → Saturación de STOOD-X» | Las mismas que `fase8b.py`. |
| `fase8_stoodx_k.py` | STOOD-X con otro K sobre las características ya guardadas de las celdas; compara el FPR@95 con el de K = 500. | «Detección OOD → Corrupciones sintéticas: *covariate shift* → Saturación de STOOD-X» | `--K`; celdas de la fase 8. |

### `convnext/`: repetición con ConvNeXt-S

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `f9_extraer.py` | Una pasada del ConvNeXt-S por conjunto (CPU, fp32). Guarda los espacios B (ReLU de la cabeza, 512) y C (*concat pooling*, 1536), los logits y la capa final plegada. | «Detección OOD con ConvNeXt-S → Elección del vector de características» | Learner exportado del ConvNeXt-S (`PKL`) y rutas de `features/completo/` del Swin-S. |
| `f9_detectores.py` | Los detectores de OpenOOD y STOOD-X sobre el espacio elegido (`--espacio B` o `C`), en CPU. | «Detección OOD con ConvNeXt-S → Resultados» | Salida de `f9_extraer.py`. Importa `my_evauator_openood`, `stoodx_paper` y `metricas_ood`. |
| `f9_tablas.py` | Métricas del ConvNeXt-S con IC y diferencias con el Swin-S por *bootstrap* pareado (CSV). | «Detección OOD con ConvNeXt-S → Resultados»; «Detección OOD con ConvNeXt-S → Comparación con el Swin-S» | Salidas de `f9_detectores.py` y de las fases A, B y C del Swin-S. |
| `f9_extra.py` | Sensibilidad pareada C − B y ViM 256 − 171; cifras para comprobar las conclusiones del Swin-S en los dos modelos. | «Detección OOD con ConvNeXt-S → Comparación con el Swin-S» (tabla de sensibilidad) | `results_ood_convnext/` y `results_ood/`. |
| `f9_mds_knn.py` | Diferencia de AUROC MDS − KNN pareada por imagen en cada modelo. | «Detección OOD con ConvNeXt-S → Comparación con el Swin-S» | Puntuaciones MDS y KNN de los dos modelos. |
| `f9_todo.py` | Lanza en orden `f9_extraer.py`, `f9_detectores.py` (B y C) y `f9_tablas.py`. | «Detección OOD con ConvNeXt-S» | — |
| `ep_preparar.py` | Espacio principal M: media espacial de los 768 canales de `stages.3.blocks.2`, comprobada con un *hook*. | «Detección OOD con ConvNeXt-S → Elección del vector de características» | Características C de `f9_extraer.py`. |
| `ep_detectores.py` | Detectores de características (KNN, MDS, RMDS, SHE, STOOD-X) en M. | «Detección OOD con ConvNeXt-S → Resultados»; «Detección OOD con ConvNeXt-S → Comparación con el Swin-S» (tabla de STOOD-X) | Salida de `ep_preparar.py`. Reutiliza `f9_detectores.py`. |
| `ep_tablas.py` | Tablas con M como espacio principal (CSV). | «Detección OOD con ConvNeXt-S → Resultados»; «Detección OOD con ConvNeXt-S → Comparación con el Swin-S» | Salidas de `ep_detectores.py` y `f9_detectores.py`. Reutiliza `f9_tablas.py`. |

## Dependencias entre ficheros y `PYTHONPATH`

Los scripts importan los módulos por su nombre, sin ruta. Desde la raíz del repositorio:

```bash
export PYTHONPATH=$PWD/OOD/openood:$PWD/OOD/stoodx:$PWD/OOD/corrupciones:$PWD/OOD/convnext:$PYTHONPATH
```

- `stoodx_paper`, `my_evauator_openood`, `comparar_resultados` y `f9_*` → `metricas_ood`.
- `fase8_extraer` → `corrupciones_zhang`, `my_evauator_openood`, `stoodx_paper`.
- `fase8_analizar` → `corrupciones_zhang` (y `fase8_figuras`, no incluido).
- `fase8b` → `fase8_analizar`, `corrupciones_zhang` (y `fase8b_figuras`, no incluido); `fase8b_empates_reales` → `fase8b`.
- `fase8_stoodx_k` → `fase8_analizar`, `fase8_extraer`, `stoodx_paper`, `corrupciones_zhang`.
- `f9_detectores` → `my_evauator_openood`, `stoodx_paper`; `ep_preparar` → `f9_extraer`, `stoodx_paper`; `ep_detectores` → `f9_detectores`; `ep_tablas` → `f9_tablas`.

Además, `my_evauator_openood.py` importa el paquete `openood` del repositorio OpenOOD y lee su carpeta `configs/`. Varios scripts añaden al `sys.path` la ruta absoluta del clon de OpenOOD donde se ejecutaron.

Entorno: `fastai_env_ood` (ver el README de la raíz).

## Fuentes

Este repositorio **no incluye** código de OpenOOD, de STOOD-X ni del banco de corrupciones; solo código propio que los importa o los llama.

- **OpenOOD**: <https://github.com/Jingkang50/OpenOOD>, commit `3c35632`, licencia MIT.
  - Yang, J. et al. (2022). *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection*. NeurIPS 2022, Datasets and Benchmarks Track. arXiv:2210.07242.
  - Zhang, J. et al. (2024). *OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection*. *Journal of Data-centric Machine Learning Research*. arXiv:2306.09301.
- **Banco de corrupciones**: <https://github.com/superjamessyx/robustness_benchmark>, commit `2b288c11f944192bea7ffae912687d27b3ad220f`. **El repositorio no tiene licencia, así que su código no se redistribuye aquí. Hay que descargarlo aparte** y apuntar `ZHANG` a la copia local.
  - Zhang, Y., Sun, Y., Li, H., Zheng, S., Zhu, C. y Yang, L. (2022). *Benchmarking the Robustness of Deep Neural Networks to Common Corruptions in Digital Pathology*. MICCAI 2022. arXiv:2206.14973.
- **STOOD-X**: Sevillano-García, I., Luengo, J. y Herrera, F. (2026). *STOOD-X: Explainable Out-of-Distribution Detection via Nonparametric Statistical Testing on Large-Scale Datasets*. *Pattern Recognition* 177, 113254. doi:10.1016/j.patcog.2026.113254. Repositorio oficial: <https://github.com/ari-dasci/S-STOOD-X> (licencia GPL-3.0). **`stoodx/stoodx_paper.py` es una implementación propia a partir del artículo**: no usa ni contiene código del repositorio oficial.
