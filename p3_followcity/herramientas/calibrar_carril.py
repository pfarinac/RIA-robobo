#!/usr/bin/env python3
"""Calibración geométrica del detector de carril (enunciado 2.1, 3.1 y 3.3).

Estima, a partir de imágenes con posición conocida, los parámetros del modelo
de ``percepcion_carril.py``:

    horizonte_y_frac      fila del horizonte (fracción de la altura)
    ancho_carril_por_px   k: ancho del carril (px) por píxel bajo el horizonte
    centro_optico_x_frac  columna del punto de fuga con el robot alineado
    focal_frac            focal horizontal / ancho de imagen

Entrada: una carpeta con imágenes y un CSV ``etiquetas.csv`` con columnas
``fichero, lateral_m, rot_deg``:
    lateral_m > 0  robot a la DERECHA del centro del carril
    rot_deg   > 0  robot girado a la DERECHA respecto al carril
Hacen falta imágenes en tramo recto: varias centradas y alineadas
(lateral 0, rot 0), algunas desplazadas (±3 a ±6 cm) y algunas giradas (±10 a
±15°). En el robot real se colocan a mano midiendo con una regla.

    python3 herramientas/calibrar_carril.py --perfil sim --carpeta registros/carril_sim
    python3 herramientas/calibrar_carril.py --perfil real --carpeta registros/carril_real --escribir

Con --escribir guarda los valores en el perfil (sección carril). Con
--depuracion CARPETA guarda las imágenes con las detecciones dibujadas.
"""

import argparse
import csv
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402
from percepcion_carril import PercepcionCarril, ajustar_modelo, tramos  # noqa: E402


def parejas_por_fila(det, img, paso_px=4):
    """(y, ancho) de parejas de líneas compatibles con el modelo actual del perfil."""
    alto, ancho = img.shape[:2]
    mascara, y0 = det.mascara(img)
    salida = []
    for i in range(0, mascara.shape[0], paso_px):
        y = i + y0
        xs = tramos(mascara[i] > 0, det.ancho_min_linea)
        w = det.k * (y - det.horizonte * alto)
        if w <= 0:
            continue
        mejor = None
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                err = abs((xs[b] - xs[a]) - w) / w
                if err <= 0.35:
                    coste = err + abs((xs[a] + xs[b]) / 2 - det.cx * ancho) / ancho
                    if mejor is None or coste < mejor[0]:
                        mejor = (coste, xs[b] - xs[a])
        if mejor:
            salida.append((y, mejor[1]))
    return salida


def ajuste_horizonte(muestras, alto):
    """w = k·(y - y_h) por mínimos cuadrados robustos. Devuelve (k, y_h_frac)."""
    y, w = np.array(muestras, float).T
    dentro = np.ones(len(y), bool)
    for _ in range(3):
        k, b = np.polyfit(y[dentro], w[dentro], 1)
        res = np.abs(k * y + b - w)
        dentro = res <= max(3.0, 2.5 * np.median(res[dentro]))
    return k, (-b / k) / alto


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--carpeta', required=True)
    parser.add_argument('--iteraciones', type=int, default=3)
    parser.add_argument('--escribir', action='store_true', help='guardar en el perfil')
    parser.add_argument('--depuracion', default=None)
    args = parser.parse_args()

    perfil = cargar_perfil(args.perfil)
    carpeta = Path(args.carpeta)
    filas = list(csv.DictReader(open(carpeta / 'etiquetas.csv', encoding='utf-8')))
    imagenes = {f['fichero']: cv2.imread(str(carpeta / f['fichero'])) for f in filas}
    filas = [f for f in filas if imagenes[f['fichero']] is not None]
    alto, ancho = next(iter(imagenes.values())).shape[:2]
    centradas = [f for f in filas if float(f['lateral_m']) == 0 and float(f['rot_deg']) == 0]
    if len(centradas) < 3:
        sys.exit('Hacen falta al menos 3 imágenes con lateral_m = 0 y rot_deg = 0.')

    config = perfil.carril.como_dict()
    for it in range(args.iteraciones):
        det = PercepcionCarril(config)
        # 1) Horizonte y ancho por píxel con las imágenes centradas.
        muestras = [m for f in centradas for m in parejas_por_fila(det, imagenes[f['fichero']])]
        if len(muestras) < 10:
            sys.exit('Pocas parejas de líneas detectadas: revisa la ROI y los umbrales HSV.')
        k, yh = ajuste_horizonte(muestras, alto)
        config['ancho_carril_por_px'], config['horizonte_y_frac'] = float(k), float(yh)
        det = PercepcionCarril(config)
        # 2) (A, B) por imagen, sin modelo previo (caso más exigente).
        AB = []
        for f in filas:
            det.reiniciar(0.0)
            mascara, y0 = det.mascara(imagenes[f['fichero']])
            ind = np.linspace(0, mascara.shape[0] - 1, det.n_filas * 2).astype(int)
            fil = [(int(i) + y0, tramos(mascara[i] > 0, det.ancho_min_linea)) for i in ind]
            modelo = det.buscar_modelo(mascara, y0, ancho, alto)
            ajuste = (ajustar_modelo(det._seleccionar(fil, ancho, alto, modelo), det.k, max(4.0, 0.025 * ancho))
                      if modelo is not None else None)
            if ajuste is not None:
                AB.append((float(f['lateral_m']), float(f['rot_deg']), ajuste[0], ajuste[1]))
        AB = np.array(AB)
        # 3) A = c_x + f·tan(θ), con θ = -rot (robot girado a la derecha -> carril hacia la izquierda).
        tan_t = np.tan(np.radians(-AB[:, 1]))
        if np.ptp(tan_t) > 0:
            fpx, cx = np.polyfit(tan_t, AB[:, 2], 1)
        else:
            fpx, cx = config['focal_frac'] * ancho, AB[:, 2].mean()
        config['centro_optico_x_frac'], config['focal_frac'] = float(cx / ancho), float(fpx / ancho)
        print('Iteración {}: horizonte={:.3f} k={:.3f} cx={:.3f} focal={:.3f} ({} imágenes ajustadas de {})'.format(
            it + 1, yh, k, cx / ancho, fpx / ancho, len(AB), len(filas)))

    # Comprobación con el detector completo (sin seguimiento entre fotogramas).
    det = PercepcionCarril(config, perfil.seguridad)
    print('\nlateral_m  rot_deg   n  validas  desplazamiento   orientacion(°)')
    grupos = OrderedDict()
    for f in filas:
        det.reiniciar(0.0)
        o = det.procesar(imagenes[f['fichero']], t=0.0)
        grupos.setdefault((float(f['lateral_m']), float(f['rot_deg'])), []).append(o)
        if args.depuracion:
            Path(args.depuracion).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.depuracion) / f['fichero']), PercepcionCarril.dibujar(imagenes[f['fichero']], o))
    pares = []
    for (lat, rot), obs in sorted(grupos.items()):
        v = [o for o in obs if o.valido]
        if v:
            d = np.array([o.desplazamiento for o in v])
            th = np.degrees([o.orientacion_rad for o in v])
            pares += [(lat, x) for x in d]
            print('{:+8.3f} {:+8.1f} {:3d} {:7d}   {:+.3f} ± {:.3f}   {:+6.1f} ± {:4.1f}'.format(
                lat, rot, len(obs), len(v), d.mean(), d.std(), th.mean(), th.std()))
        else:
            print('{:+8.3f} {:+8.1f} {:3d} {:7d}'.format(lat, rot, len(obs), 0))
    lat, desp = np.array(pares).T if pares else (np.zeros(1), np.zeros(1))
    if np.ptp(lat) > 0:
        pendiente = np.polyfit(lat, desp, 1)[0]
        print('\nAncho de carril estimado: {:.3f} m (desplazamiento = lateral / ancho)'.format(1 / pendiente))

    claves = ('horizonte_y_frac', 'ancho_carril_por_px', 'centro_optico_x_frac', 'focal_frac')
    print('\nValores calibrados:\n' + '\n'.join('  carril.{} = {:.4f}'.format(c, config[c]) for c in claves))
    if args.escribir:
        datos = json.load(open(perfil.ruta, encoding='utf-8'), object_pairs_hook=OrderedDict)
        for c in claves:
            datos['carril'][c] = round(config[c], 4)
        json.dump(datos, open(perfil.ruta, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        print('Guardado en', perfil.ruta)


if __name__ == '__main__':
    main()
