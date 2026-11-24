#!/usr/bin/env python3
"""Inspección y calibración del detector del guía (enunciado 2.2, 3.1 y 3.3).

Modos:

  Monitor (por defecto): muestra en vivo lo que entrega detected_object y la
  observación normalizada, junto con los valores máximos de x, y, width y
  height vistos. Sirve para comprobar la resolución real del detector
  (guia.ancho_imagen_px / alto_imagen_px) y las etiquetas que aparecen
  (guia.etiquetas_validas).

      python3 herramientas/inspeccionar_guia.py --perfil sim
      python3 herramientas/inspeccionar_guia.py --perfil real

  Calibración a distancia conocida: con el guía parado a una distancia
  medida delante del seguidor (por ejemplo 0,25, 0,35 y 0,50 m), recoge N
  detecciones nuevas y las añade a un CSV. Repetir en simulación y en real
  y comparar con analizar_calibracion_guia.py.

      python3 herramientas/inspeccionar_guia.py --perfil sim \
          --calibrar --distancia 0.25 --muestras 40 \
          --salida registros/calibracion_guia.csv

  Los mensajes se registran sin filtrar por etiqueta ni confianza en las
  columnas brutas, y la observación filtrada en las columnas norm_*, para
  poder ver qué descarta el filtro.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402
from ros_guia import SuscriptorGuia  # noqa: E402

CAMPOS_CSV = ['fecha', 'perfil', 'distancia_m', 'lateral_m', 'muestra',
              'label', 'confidence', 'x_px', 'y_px', 'width_px', 'height_px',
              'timestamp', 'aceptada', 'norm_visible', 'norm_x', 'norm_tamano',
              'escala_tamano', 'metrica_tamano']


def monitor(nodo, sub, periodo_s):
    maximos = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    etiquetas = {}
    ultimo_print = 0.0
    n_previo = 0
    print('Escuchando {} (Ctrl+C para salir)...'.format(
        sub.perfil.topic('topic_objeto')))
    try:
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.05)
            msg = sub.ultimo_mensaje
            if msg is not None and sub.n_mensajes != n_previo:
                n_previo = sub.n_mensajes
                for clave in maximos:
                    maximos[clave] = max(maximos[clave], int(getattr(msg, clave)))
                etiquetas[msg.label] = etiquetas.get(msg.label, 0) + 1
            if time.monotonic() - ultimo_print < periodo_s:
                continue
            ultimo_print = time.monotonic()
            obs = sub.observar()
            if msg is not None:
                bruto = ('label={!r:<12} conf={:.2f} x={:4d} y={:4d} w={:4d} h={:4d} ts={}'
                         .format(msg.label, msg.confidence, msg.x, msg.y,
                                 msg.width, msg.height, msg.timestamp))
            else:
                bruto = 'sin mensajes todavía'
            print('{} | vis={:.0f} x={:+.2f} tam={:.3f} perd={:.2f} {} | max={} | {}'.format(
                bruto, obs.visible, obs.x, obs.tamano, obs.tiempo_perdido,
                obs.motivo_invalida or 'ok', maximos,
                sub.percepcion.contadores))
    except KeyboardInterrupt:
        pass
    print('\nResumen')
    print('  Mensajes recibidos:', sub.n_mensajes)
    print('  Contadores del filtro:', sub.percepcion.contadores)
    print('  Máximos vistos (px):', maximos)
    print('  Etiquetas vistas:', etiquetas)
    g = sub.perfil.guia
    if maximos['x'] > g.ancho_imagen_px or maximos['y'] > g.alto_imagen_px:
        print('  AVISO: hay coordenadas mayores que guia.ancho/alto_imagen_px del perfil '
              '({} x {}). Corrige el perfil.'.format(g.ancho_imagen_px, g.alto_imagen_px))


def calibrar(nodo, sub, args):
    ruta = Path(args.salida)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not ruta.exists()
    g = sub.perfil.guia
    recogidas, vistos_ts = 0, set()
    limite = time.monotonic() + args.timeout
    print('Guía a {:.3f} m (lateral {:+.3f} m). Recogiendo {} detecciones...'.format(
        args.distancia, args.lateral, args.muestras))
    with open(ruta, 'a', newline='', encoding='utf-8') as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=CAMPOS_CSV)
        if nuevo:
            escritor.writeheader()
        n_previo = sub.n_mensajes
        while recogidas < args.muestras and time.monotonic() < limite:
            rclpy.spin_once(nodo, timeout_sec=0.05)
            if sub.n_mensajes == n_previo:
                continue
            n_previo = sub.n_mensajes
            msg = sub.ultimo_mensaje
            clave = (msg.timestamp, msg.x, msg.y, msg.width, msg.height)
            if msg.timestamp != 0 and clave in vistos_ts:
                continue  # mensaje repetido por el puente
            vistos_ts.add(clave)
            obs = sub.observar()
            escritor.writerow({
                'fecha': datetime.now().isoformat(timespec='seconds'),
                'perfil': sub.perfil.nombre, 'distancia_m': args.distancia,
                'lateral_m': args.lateral, 'muestra': recogidas,
                'label': msg.label, 'confidence': round(float(msg.confidence), 4),
                'x_px': msg.x, 'y_px': msg.y, 'width_px': msg.width,
                'height_px': msg.height, 'timestamp': msg.timestamp,
                'aceptada': int(obs.visible > 0 and obs.motivo_invalida == ''),
                'norm_visible': obs.visible, 'norm_x': round(obs.x, 4),
                'norm_tamano': round(obs.tamano, 4),
                'escala_tamano': g.escala_tamano, 'metrica_tamano': g.metrica_tamano})
            recogidas += 1
    print('Guardadas {} muestras en {}'.format(recogidas, ruta))
    if recogidas < args.muestras:
        print('AVISO: se agotó el tiempo ({:.0f} s). ¿Se ve al guía?'.format(args.timeout))


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--calibrar', action='store_true')
    parser.add_argument('--distancia', type=float, help='m, distancia frontal al guía')
    parser.add_argument('--lateral', type=float, default=0.0,
                        help='m, desplazamiento lateral del guía (+ derecha)')
    parser.add_argument('--muestras', type=int, default=40)
    parser.add_argument('--timeout', type=float, default=60.0, help='s')
    parser.add_argument('--periodo', type=float, default=0.5,
                        help='s, periodo de impresión del monitor')
    parser.add_argument('--salida', default='registros/calibracion_guia.csv')
    args = parser.parse_args()
    if args.calibrar and args.distancia is None:
        parser.error('--calibrar necesita --distancia')

    perfil = cargar_perfil(args.perfil)
    print('Perfil: {} ({})'.format(perfil.nombre, perfil.ruta))
    rclpy.init()
    nodo = Node('p3_inspeccionar_guia')
    sub = SuscriptorGuia(nodo, perfil)
    try:
        if args.calibrar:
            calibrar(nodo, sub, args)
        else:
            monitor(nodo, sub, args.periodo)
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
