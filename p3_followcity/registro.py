"""Registros CSV de pasos y de episodios (enunciado 6.4).

Por cada paso: marca temporal, configuración (C1–C4), perfil, validez del
carril, detección del guía, observación normalizada, acción cmd_vel enviada,
recompensa y, en simulación, la verdad privilegiada. Por cada episodio: las
métricas de recorrido (completado sin intervención, salidas de carril,
contactos, paradas superadas...). Las mismas columnas en sim y en real, para
comparar antes y después de los ajustes.
"""

import csv
import time
from datetime import datetime
from pathlib import Path


class RegistroCSV:
    """CSV con cabecera fijada por la primera fila; las claves nuevas se ignoran
    con un aviso (las columnas deben ser estables dentro de un fichero)."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._f = None
        self._w = None
        self._avisadas = set()

    def escribir(self, fila):
        if self._w is None:
            self._f = open(self.ruta, 'w', newline='', encoding='utf-8')
            self._w = csv.DictWriter(self._f, fieldnames=list(fila), extrasaction='ignore')
            self._w.writeheader()
        nuevas = set(fila) - set(self._w.fieldnames) - self._avisadas
        if nuevas:
            self._avisadas |= nuevas
            print('Aviso registro {}: columnas no registradas {}'.format(self.ruta.name, sorted(nuevas)))
        self._w.writerow({k: _formato(v) for k, v in fila.items()})
        self._f.flush()

    def cerrar(self):
        if self._f:
            self._f.close()
            self._f = self._w = None


def _formato(v):
    if isinstance(v, float):
        return round(v, 5)
    return v


class Registros:
    """Pareja de registros (pasos y episodios) en una carpeta con marca de tiempo."""

    def __init__(self, carpeta, etiqueta):
        sello = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = Path(carpeta) / '{}_{}'.format(etiqueta, sello)
        self.carpeta = base
        self.pasos = RegistroCSV(base / 'pasos.csv')
        self.episodios = RegistroCSV(base / 'episodios.csv')
        self.t0 = time.monotonic()

    def cerrar(self):
        self.pasos.cerrar()
        self.episodios.cerrar()
