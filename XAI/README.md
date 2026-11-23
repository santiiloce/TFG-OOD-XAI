# Explicabilidad

LRP y CRP sobre el ConvNeXt-S (timm `convnext_small` con la cabeza de fastai), tratamiento de la LayerNorm, *Expected Gradients* (EG), explicaciones de casos con sus vecinos al estilo de STOOD-X en el espacio M y atlas RelMax.

## Ficheros

### `lrp/`: configuración de LRP (variante A)

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `lrp_convnext_v2.py` | Base de LRP para ConvNeXt-S con zennit: canonizer del bloque ConvNeXt (suma residual como módulo `Sum`), reglas ε y z⁺ sin sesgo, regla de Ali et al. para la LayerNorm (`ReglaLN`), Pass en GELU y LayerNorm, y `make_composite`. | «Explicabilidad → LRP → Reglas de propagación»; «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | — |
| `lrp_fase2.py` | Variantes de composite, entre ellas `ReglaLNEstabilizada` (regla de Ali con ε) y el canonizer del Bottleneck de ResNet de timm. | «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | — |
| `comun2.py` | Carga del learner, preprocesado y selección de imágenes (28 de test). | «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | Learner exportado del ConvNeXt-S (`EJEC`) y datos del test. |
| `comun3.py` | Utilidades comunes (plegado de BatchNorm, IC por *bootstrap*). | «Explicabilidad y eliminación de sesgos» | — |
| `lrp_fase5.py` | Reglas LRP-γ por grupos de capas y tratamientos de la BatchNorm de la cabeza; ABPC y aleatorización del modelo. | «Explicabilidad y eliminación de sesgos» | — |
| `composite_final.py` | Configuración final (variante A): `preparar_modelo_final` y `composite_final`. La usa el resto del capítulo. | «Explicabilidad → LRP → Reglas de propagación»; «Explicabilidad y eliminación de sesgos» | — |
| `LICENSES/` | Textos íntegros de la LGPL-3.0, la GPL-3.0 y la Apache-2.0 (ver «Licencia»). | — | — |

### `layernorm/`: tratamiento de la LayerNorm

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `01_ablacion_ln.py` | Ablación en 28 imágenes: Pass (identidad), sin regla (jacobiano completo), regla de Ali et al. y regla de Ali con ε absoluto (10⁻²) y relativo; conservación por bloque y por LayerNorm. | «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | Learner del ConvNeXt-S y las 28 imágenes de `comun2`. Entorno `crp_env`. |
| `02_denominadores.py` | Distribución de los denominadores de la regla de Ali en cada LayerNorm (solo *forward*). | «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | Las mismas. |
| `comun_t4.py` | T4: fusión exacta del centrado y de γ de la LayerNorm en la capa lineal siguiente (la división por σ queda como identidad); EG, gradientes integrados y Gradiente × Entrada; aleatorización. | «Explicabilidad y eliminación de sesgos» (introducción del capítulo); «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | — |
| `01_t4_conservacion.py` | Conservación de A frente a T4 (fusión) en 28 imágenes, por imagen, etapa y bloque. | «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» | `--hilos`. |

### `eg/`: fidelidad de los métodos

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `02_eg_ig_fidelidad.py` | Calcula A, G×E, T4, EG (128 referencias de entrenamiento) e IG en 112 imágenes de test; fidelidad por ABPC (MoRF − LeRF, parcelas de 16 px), completitud de EG y sensibilidad. | «Explicabilidad y eliminación de sesgos» (introducción del capítulo: EG y ABPC); «Explicabilidad y eliminación de sesgos → Tratamiento de la LayerNorm» (ABPC de la fusión) | Learner del ConvNeXt-S, imágenes de test y de entrenamiento. Necesita GPU. |

### `crp/`: CRP, casos y atlas

| Fichero | Qué hace | Sección de la memoria | Entradas |
|---|---|---|---|
| `comun4.py` | Utilidades de CRP: concepto por canal, mapas condicionados, G×E condicionado y funciones de dibujo que usan los scripts de casos. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos»; «Explicabilidad y eliminación de sesgos → Atlas RelMax» | — |
| `02_caracteristicas.py` | Vector de características del ConvNeXt-S (media espacial de los 768 canales de `stages.3.blocks.2`), logits y predicción para entrenamiento, test, validación y NCT-CRC TUM y NORM. Solo cálculo. Lo usan los casos y el atlas. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos»; «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Learner del ConvNeXt-S y el orden de imágenes de `features/completo/*.npz` (de `OOD/openood/my_evauator_openood.py`). Escribe `fase4/features/convnext_<conjunto>.npz`. Necesita GPU. |
| `casos/casos_M.py` | Define los casos, sus 3 vecinos de entrenamiento y los canales con el STOOD-X del espacio M (media espacial de los 768 canales de `stages.3.blocks.2`). | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos» | Puntuaciones y características del ConvNeXt-S en M (`OOD/convnext/ep_*.py`). Solo CPU; entorno `fastai_env_ood`. |
| `casos/03_explicaciones.py` | Mapas de EG condicionado (φ^k, 128 referencias) de los casos y sus vecinos, cobertura, suelo de ruido entre semillas, aleatorización del modelo y controles. **Además dibuja figuras de comprobación** (PNG y PDF) con las funciones de `comun4`; no son las figuras de la memoria. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos» (mapas de EG); «Explicabilidad y eliminación de sesgos → Discusión» (aleatorización) | Los casos seleccionados, el learner del ConvNeXt-S e imágenes de entrenamiento y test. Escribe `fase10_eg/paso3/` (`mapas.npz`, `cobertura_explicaciones.csv`, `suelo_ruido_semillas.csv`…). Necesita GPU. |
| `casos/10_caso_lym.py` | Caso OOD aceptado (NCT-CRC LYM): la LYM aceptada más cercana a la mediana de las aceptadas, con sus vecinos y canales. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos» | Las mismas. Solo CPU. |
| `casos/01_mapas_M.py` | Mapas condicionados con la variante A y con EG condicionado para los casos, más el suelo de ruido y la aleatorización. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos»; «Explicabilidad y eliminación de sesgos → Discusión» (aleatorización) | Salida de `casos_M.py`. Reutiliza los mapas de EG de `03_explicaciones.py`. Necesita GPU. |
| `casos/11_mapas_lym.py` | Mapas A y EG del caso LYM. | «Explicabilidad y eliminación de sesgos → Explicaciones de casos con sus vecinos» | Salida de `10_caso_lym.py`. |
| `atlas/comun10.py` | Conductancia por canal con el camino de EG (la vía eficiente: *jvp* del cuerpo y gradiente solo en la cabeza). | «Explicabilidad y eliminación de sesgos» (EG condicionado); «Explicabilidad y eliminación de sesgos → Atlas RelMax» | — |
| `atlas/03_relmax.py` | RelMax con zennit-crp sobre todo el entrenamiento (90.800 imágenes), con la variante A o con G×E (`A\|GxE`); R = f_c de la clase real. Reanudable por tramos. | «Explicabilidad y eliminación de sesgos → Atlas RelMax» (comparación con el entrenamiento completo) | Argumento `A` o `GxE`; características de la fase 4 (`fase4/features/convnext_train.npz`, de `02_caracteristicas.py`). Importa `comun4`. Necesita GPU. |
| `atlas/04_conceptos.py` | 4 conceptos por clase (media de la relevancia condicional normalizada en el test de la clase real), 8 referencias RelMax, RelMax frente a ActMax y solapamiento. **Además dibuja figuras de comprobación** (PNG). | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Salidas de `02_caracteristicas.py` y `03_relmax.py` (RelMax de A y G×E, ActMax de A). Escribe `fase4/relmax/conceptos.csv`, `referencias.csv`… Necesita GPU. |
| `atlas/04_relmax_bloque.py` | Conductancia de EG (768 canales) para entrenamiento y test, por bloques de 2.000 imágenes. | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Argumentos `<bloque> [semilla]`. Necesita GPU. |
| `atlas/10_muestra_C_A.py` | Muestra estratificada de entrenamiento (4.654 imágenes): conductancia de EG y relevancia de canal de la variante A. | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Salida de `04_relmax_bloque.py`. Necesita GPU. |
| `atlas/11_relmax_muestra.py` | Atlas RelMax sobre la muestra: 4 conceptos por clase y 8 referencias por concepto, con A y con EG. | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Salidas de `10_muestra_C_A.py`, el RelMax de `03_relmax.py` y los conceptos de la fase 4 (`fase4/relmax/conceptos.csv`, de `04_conceptos.py`). |
| `atlas/05_relmax_post.py` | Une los bloques de conductancia, comprueba la matriz, elige 4 conceptos por clase, las 8 referencias RelMax por canal y las 50 candidatas para la estabilidad, y compara con A y G×E de la fase 4 (CSV y JSON, sin figuras). | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Salida de `04_relmax_bloque.py`, características de la fase 4 y los conceptos de la fase 4 (`fase4/relmax/conceptos.csv`, de `04_conceptos.py`). Solo CPU. |
| `atlas/06_estabilidad_mapas.py` | Conductancia con la semilla 1 para las 50 candidatas de cada concepto y mapas de las 8 referencias. | «Explicabilidad y eliminación de sesgos → Atlas RelMax»; «Explicabilidad y eliminación de sesgos → Discusión» (las referencias cambian con la semilla) | Una carpeta de resultados del RelMax (`sys.argv[1]`); su docstring la llama «dir de 05» (`05_relmax_post.py`). Necesita GPU. |
| `atlas/07_estabilidad_analisis.py` | Solape del top-8 de cada concepto entre semillas. | «Explicabilidad y eliminación de sesgos → Atlas RelMax»; «Explicabilidad y eliminación de sesgos → Discusión» | Salida de `06_estabilidad_mapas.py`. |
| `atlas/12_comprobaciones_finales.py` | Pureza del atlas de EG con la semilla 1 y mapas de EG por parcelas de 16 px. | «Explicabilidad y eliminación de sesgos → Atlas RelMax» | Resultados de la muestra. Solo CPU. |

## Dependencias entre módulos y `PYTHONPATH`

Los módulos de cada fase importan, sin modificarlos, los de fases anteriores. Cadena de dependencias:

```
lrp_convnext_v2                     (base)
comun2                              (base)
lrp_fase2      → comun2, lrp_convnext_v2
comun3         → comun2
lrp_fase5      → comun3, lrp_convnext_v2
composite_final→ lrp_fase5
comun4         → composite_final, lrp_fase5
comun_t4       → composite_final, lrp_convnext_v2, lrp_fase5
comun10        → comun_t4
```

Scripts: `01_ablacion_ln` → `comun2`, `lrp_convnext_v2`, `lrp_fase2`; `02_denominadores` → `comun2`; `01_t4_conservacion` y `02_eg_ig_fidelidad` → `comun_t4`; `01_mapas_M` y `11_mapas_lym` → `comun10`, `comun4`; `04_relmax_bloque` → `comun10`; `03_explicaciones` → `comun10`, `comun4`, `composite_final`; `02_caracteristicas`, `03_relmax` y `04_conceptos` → `comun4`; `06_estabilidad_mapas` y `10_muestra_C_A` → `comun10`, `composite_final`.

En la ejecución original, cada script añadía al `sys.path` las carpetas absolutas de la fase correspondiente. Para usarlos desde este repositorio, desde la raíz:

```bash
export PYTHONPATH=$PWD/XAI/lrp:$PWD/XAI/layernorm:$PWD/XAI/crp:$PWD/XAI/crp/atlas:$PYTHONPATH
```

Entorno: `crp_env` (zennit 0.4.6, zennit-crp 0.6.0; ver el README de la raíz), salvo `casos_M.py` y `10_caso_lym.py`, que se ejecutaron en `fastai_env_ood`. Todos necesitan el learner exportado del ConvNeXt-S (`entrenamiento_modelos/`) y los datos, que no se distribuyen.

## Licencia

Todo el código de esta carpeta es MIT (ver `LICENSE` en la raíz) **salvo tres clases derivadas de terceros**, que se distribuyen bajo **LGPL-3.0-or-later**:

| Clase | Fichero | Origen |
|---|---|---|
| `ConvNeXtBlockCanonizer` | `lrp/lrp_convnext_v2.py` | Estructura de `ResNetBottleneckCanonizer` de zennit 0.4.6 (LGPL-3.0-or-later, © 2019-2021 Christopher J. Anders) y `ConvNeXtBlock.forward` de timm 1.0.25 (Apache-2.0, © 2019 Ross Wightman) |
| `ZPlusSinSesgo` | `lrp/lrp_convnext_v2.py` | `ZPlus` de zennit 0.4.6 (LGPL-3.0-or-later) |
| `TimmBottleneckCanonizer` | `lrp/lrp_fase2.py` | Estructura de `ResNetBottleneckCanonizer` de zennit 0.4.6 y `Bottleneck.forward` de timm 1.0.25 (Apache-2.0) |

Cada clase lleva un comentario con su origen y lo que cambia respecto del original. Los textos de las licencias están en `lrp/LICENSES/`: la LGPL-3.0 y la GPL-3.0, porque la LGPL-3.0 se define como un conjunto de permisos adicionales sobre la GPL-3.0, y la Apache-2.0 tal como la distribuye timm.

## Fuentes

- **zennit**: <https://github.com/chr5tphr/zennit> (URL tomada de los metadatos del paquete instalado), versión 0.4.6, licencia LGPL-3.0-or-later. Anders, C. J., Neumann, D., Samek, W., Müller, K.-R. y Lapuschkin, S. (2021). *Software for Dataset-wide XAI: From Local Explanations to Global Insights with Zennit, CoRelAy, and ViRelAy*. arXiv:2106.13200.
- **zennit-crp**: <https://github.com/rachtibat/zennit-crp> (URL de los metadatos del paquete), versión 0.6.0, licencia BSD-3-Clause-Clear. Achtibat, R. et al. (2023). *From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation*. *Nature Machine Intelligence* 5, 1006-1019. doi:10.1038/s42256-023-00711-8.
- **Regla de la LayerNorm**: Ali, A., Schnake, T., Eberle, O., Montavon, G., Müller, K.-R. y Wolf, L. (2022). *XAI for Transformers: Better Explanations through Conservative Propagation*. ICML 2022, PMLR 162. arXiv:2202.07304.
- **Expected Gradients**: Erion, G., Janizek, J. D., Sturmfels, P., Lundberg, S. M. y Lee, S.-I. (2020). arXiv:1906.10670.
- **timm**: <https://github.com/huggingface/pytorch-image-models>, versión 1.0.25, licencia Apache-2.0.
