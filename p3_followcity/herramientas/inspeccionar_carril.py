#!/usr/bin/env python3
"""Inspección en vivo del detector de carril y captura de imágenes de calibración.

Monitor (por defecto): procesa camera/image_raw, imprime desplazamiento,
ángulo, orientación y validez, y guarda cada --cada fotogramas la imagen con
las detecciones dibujadas en --salida (para la memoria y para ajustar umbrales).

    python3 herramientas/inspeccionar_carril.py --perfil sim --salida registros/carril_vivo

Captura para calibrar_carril.py: guarda N fotogramas crudos con una posición
conocida y añade sus etiquetas a <carpeta>/etiquetas.csv. Repetir con el
robot centrado y alineado, desplazado y girado, en un tramo recto.

    python3 herramientas/inspeccionar_carril.py --perfil real --capturar registros/carril_real \
        --lateral 0.0 --rot 0 --muestras 3
    (convención: lateral_m > 0 robot a la derecha del centro; rot_deg > 0 girado a la derecha)

Con --tilt se fija la inclinación del perfil antes de empezar.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402
from percepcion_carril import PercepcionCarril  # noqa: E402
from ros_carril import SuscriptorCarril  # noqa: E402


def mover_tilt(nodo, perfil):
    try:
        from robobo_ros2_interfaces.srv import MoveTilt
    except ImportError:
        return
    cliente = nodo.create_client(MoveTilt, perfil.topic('accion_tilt'))
    if not cliente.wait_for_service(timeout_sec=3.0):
        print('Aviso: no responde', perfil.topic('accion_tilt'))
        return
    peticion = MoveTilt.Request()
    peticion.angle = float(perfil.camara.tilt_deg)
    peticion.speed = float(perfil.camara.tilt_velocidad_deg_s)
    futuro = cliente.call_async(peticion)
    rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=perfil.camara.timeout_movimiento_s)


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--salida', default='registros/carril_vivo')
    parser.add_argument('--cada', type=int, default=5, help='guardar 1 de cada N fotogramas')
    parser.add_argument('--capturar', default=None, help='carpeta de captura para calibrar')
    parser.add_argument('--lateral', type=float, default=0.0, help='m (+ derecha)')
    parser.add_argument('--rot', type=float, default=0.0, help='grados (+ girado a la derecha)')
    parser.add_argument('--muestras', type=int, default=3)
    parser.add_argument('--tilt', action='store_true', help='mover el tilt al valor del perfil')
    args = parser.parse_args()

    perfil = cargar_perfil(args.perfil)
    rclpy.init()
    nodo = Node('p3_inspeccionar_carril')
    sub = SuscriptorCarril(nodo, perfil)
    if args.tilt:
        mover_tilt(nodo, perfil)
    try:
        if args.capturar:
            carpeta = Path(args.capturar)
            carpeta.mkdir(parents=True, exist_ok=True)
            etiquetas = carpeta / 'etiquetas.csv'
            nuevo = not etiquetas.exists()
            with open(etiquetas, 'a', newline='', encoding='utf-8') as f:
                escritor = csv.writer(f)
                if nuevo:
                    escritor.writerow(['fichero', 'lateral_m', 'rot_deg'])
                for i in range(args.muestras):
                    if not sub.esperar_imagen(10.0):
                        sys.exit('No llegan imágenes de ' + perfil.topic('topic_imagen'))
                    img, _ = sub.ultima_imagen()
                    nombre = 'cal_{}_lat{:+.3f}_rot{:+.0f}_{}.png'.format(
                        time.strftime('%H%M%S'), args.lateral, args.rot, i)
                    cv2.imwrite(str(carpeta / nombre), img)
                    escritor.writerow([nombre, args.lateral, args.rot])
                    print('guardada', nombre)
            return
        salida = Path(args.salida)
        salida.mkdir(parents=True, exist_ok=True)
        print('Procesando {} (Ctrl+C para salir)...'.format(perfil.topic('topic_imagen')))
        n_previo = 0
        while rclpy.ok():
            rclpy.spin_once(nodo, timeout_sec=0.05)
            if sub.n_imagenes == n_previo:
                continue
            n_previo = sub.n_imagenes
            t0 = time.perf_counter()
            obs = sub.observar()
            ms = 1000 * (time.perf_counter() - t0)
            print('img {:5d} | val={:.0f} validez={:.2f} desp={:+.2f} ang={:+5.1f}° or={:+5.1f}° {:<9} '
                  'pts={:2d} edad={:.2f}s {:.1f} ms {}'.format(
                      sub.n_imagenes, obs.valido, obs.validez, obs.desplazamiento,
                      obs.angulo_rad * 57.2958, obs.orientacion_rad * 57.2958, obs.lados,
                      obs.puntos, obs.edad_imagen_s, ms, obs.motivo_invalida))
            if sub.n_imagenes % args.cada == 0:
                img, _ = sub.ultima_imagen()
                cv2.imwrite(str(salida / 'carril_{:05d}.png'.format(sub.n_imagenes)),
                            PercepcionCarril.dibujar(img, obs))
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
