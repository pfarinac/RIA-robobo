#!/usr/bin/env python3
"""Uso: python3 p2_graficas.py evol_s42/evolucion.csv [evol_s43/evolucion.csv ...]

Dibuja la evolución a partir de uno o varios evolucion.csv (una evolución por
semilla) y guarda la figura como evolucion.png junto al primer CSV.
Con un solo CSV muestra la propia curva y, en la media de la población, una banda
de ± una desviación típica. Con varios CSV muestra la media entre semillas y una
banda entre el mínimo y el máximo.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Rutas de los CSV pasadas por la línea de comandos.
rutas_csv = sys.argv[1:]
# Se lee cada CSV como array estructurado: se accede a las columnas por su nombre de cabecera.
datos = [np.genfromtxt(r, delimiter=",", names=True) for r in rutas_csv]
# Número de generaciones a dibujar: el de la evolución más corta, para poder apilarlas.
# (atleast_1d evita errores si un CSV tuviera una sola fila.)
n = min(len(np.atleast_1d(d["generacion"])) for d in datos)
gen = np.arange(n)  # eje x común: 0 ... n-1


def apilar(campo):
    """Devuelve una matriz (semillas x generaciones) con la columna `campo` de cada CSV."""
    return np.array([np.atleast_1d(d[campo])[:n] for d in datos])


def curva(ax, campo, etiqueta, usar_desv=False):
    """Dibuja la curva de `campo` y su banda de sombreado en el eje `ax`.

    - media: media entre semillas (con un solo CSV es la propia columna).
    - banda: ± cal_desv si hay un solo CSV y usar_desv=True (desviación de la
      población en cada generación); en caso contrario, mínimo y máximo entre semillas.
    """
    v = apilar(campo)
    media = v.mean(axis=0)
    if len(datos) == 1 and usar_desv:
        s = apilar("cal_desv")[0]
        lo, hi = media - s, media + s
    else:
        lo, hi = v.min(axis=0), v.max(axis=0)
    ax.plot(gen, media, label=etiqueta)
    ax.fill_between(gen, lo, hi, alpha=0.25)


# Tres paneles en una fila: calidad, número de especies y complejidad del mejor.
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
# Panel 1: calidad del mejor individuo y media de la población (con su desviación).
curva(ax[0], "cal_mejor", "Mejor individuo")
curva(ax[0], "cal_media", "Media de la población", usar_desv=True)
ax[0].set_xlabel("Generación"); ax[0].set_ylabel("Calidad (escala 0-100)"); ax[0].legend()
# Panel 2: número de especies que mantiene NEAT en cada generación.
curva(ax[1], "especies", "Especies")
ax[1].set_xlabel("Generación"); ax[1].set_ylabel("Número de especies")
# Panel 3: complejidad de la red del mejor individuo (nodos y conexiones activas).
curva(ax[2], "nodos_mejor", "Nodos")
curva(ax[2], "conexiones_mejor", "Conexiones activas")
ax[2].set_xlabel("Generación"); ax[2].set_ylabel("Complejidad del mejor"); ax[2].legend()
# La imagen se guarda en la carpeta del primer CSV, con 200 ppp y sin márgenes sobrantes.
ruta_grafica = Path(rutas_csv[0]).resolve().parent / "evolucion.png"
fig.savefig(ruta_grafica, dpi=200, bbox_inches="tight")
print("Gráfica guardada en:", ruta_grafica)
# Resumen por consola: pasos y horas acumulados al final de cada evolución
# (última fila de las columnas 'pasos' y 'segundos').
for r, d in zip(rutas_csv, datos):
    print(r, "pasos:", int(np.atleast_1d(d["pasos"])[-1]),
          "horas:", round(float(np.atleast_1d(d["segundos"])[-1]) / 3600, 2))
