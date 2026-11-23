# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Santiago López Cerro
# Parte del TFG «Inteligencia Artificial Explicable en medicina: métodos para introducir al humano en la adquisición de conocimiento». Si usas este código, cita el TFG (ver README.md).
# Fase 9 completa en un solo trabajo (mismo nodo y núcleos): extracción -> detectores B -> sensibilidad C -> tablas.
# Cada paso es reanudable (la extracción y STOOD-X se saltan lo ya hecho).
import subprocess, sys, time
pasos = [['f9_extraer.py'],
         ['f9_detectores.py', '--espacio', 'B'],
         ['f9_detectores.py', '--espacio', 'C', '--metodos', 'knn', 'mds', 'rmds', '--stoodx_conjuntos'],
         ['f9_tablas.py'],
         ['f9_detectores.py', '--espacio', 'C', '--metodos', '--stoodx_conjuntos', 'id_test', 'near_nct_tum', 'near_nct_norm'],
         ['f9_tablas.py']]   # STOOD-X en C va al final ("si el tiempo lo permite"); se rehacen las tablas con él
for p in pasos:
    t = time.time(); print(f'==== {" ".join(p)}', flush=True)
    subprocess.run([sys.executable, '-u'] + p, check=True)
    print(f'==== hecho en {(time.time() - t) / 3600:.2f} h', flush=True)
