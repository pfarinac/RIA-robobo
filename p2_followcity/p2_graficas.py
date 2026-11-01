#!/usr/bin/env python3
"""Uso: python3 p2_graficas.py evol_s42/evolucion.csv [evol_s43/evolucion.csv ...]"""
import sys
import numpy as np
import matplotlib.pyplot as plt

datos = [np.genfromtxt(r, delimiter=",", names=True) for r in sys.argv[1:]]
n = min(len(np.atleast_1d(d["generacion"])) for d in datos)
gen = np.arange(n)

def apilar(campo):
    return np.array([np.atleast_1d(d[campo])[:n] for d in datos])

def curva(ax, campo, etiqueta, usar_desv=False):
    v = apilar(campo)
    media = v.mean(axis=0)
    if len(datos) == 1 and usar_desv:
        s = apilar("cal_desv")[0]
        lo, hi = media - s, media + s
    else:
        lo, hi = v.min(axis=0), v.max(axis=0)
    ax.plot(gen, media, label=etiqueta)
    ax.fill_between(gen, lo, hi, alpha=0.25)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
curva(ax[0], "cal_mejor", "Mejor individuo")
curva(ax[0], "cal_media", "Media de la población", usar_desv=True)
ax[0].set_xlabel("Generación"); ax[0].set_ylabel("Calidad (escala 0-100)"); ax[0].legend()
curva(ax[1], "especies", "Especies")
ax[1].set_xlabel("Generación"); ax[1].set_ylabel("Número de especies")
curva(ax[2], "nodos_mejor", "Nodos")
curva(ax[2], "conexiones_mejor", "Conexiones activas")
ax[2].set_xlabel("Generación"); ax[2].set_ylabel("Complejidad del mejor"); ax[2].legend()
fig.savefig("evolucion.png", dpi=200, bbox_inches="tight")
for r, d in zip(sys.argv[1:], datos):
    print(r, "pasos:", int(np.atleast_1d(d["pasos"])[-1]),
          "horas:", round(float(np.atleast_1d(d["segundos"])[-1]) / 3600, 2))