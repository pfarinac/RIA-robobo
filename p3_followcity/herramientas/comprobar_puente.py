#!/usr/bin/env python3
"""Comprueba que el puente robobo_ros2 ofrece lo que necesita la P3.

Durante unos segundos mide la frecuencia real y el retardo de
camera/image_raw y detected_object, compara la resolución de la imagen con
el perfil y lista los servicios y acciones esperados.

    python3 herramientas/comprobar_puente.py --perfil sim --segundos 10

Lanzar el puente con los módulos necesarios, por ejemplo:
    ros2 launch robobo_ros2 ... modules:="['base','camera','object','sim']"
(en real, sin 'sim').
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from robobo_ros2_interfaces.msg import DetectedObject  # noqa: E402

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402


class Medidor:
    def __init__(self, nodo):
        self.nodo = nodo
        self.llegadas = []
        self.retardos = []
        self.ultimo = None
        self.timestamps = set()

    def __call__(self, msg):
        self.llegadas.append(time.monotonic())
        self.ultimo = msg
        if hasattr(msg, 'timestamp'):
            self.timestamps.add(msg.timestamp)
        cabecera = getattr(msg, 'header', None)
        if cabecera is not None and cabecera.stamp.sec:
            ahora = self.nodo.get_clock().now().nanoseconds * 1e-9
            sello = cabecera.stamp.sec + cabecera.stamp.nanosec * 1e-9
            self.retardos.append(ahora - sello)

    def frecuencia(self):
        if len(self.llegadas) < 2:
            return 0.0
        return (len(self.llegadas) - 1) / (self.llegadas[-1] - self.llegadas[0])


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--segundos', type=float, default=10.0)
    args = parser.parse_args()
    perfil = cargar_perfil(args.perfil)

    rclpy.init()
    nodo = Node('p3_comprobar_puente')
    imagen, objeto = Medidor(nodo), Medidor(nodo)
    nodo.create_subscription(Image, perfil.topic('topic_imagen'), imagen, 10)
    nodo.create_subscription(DetectedObject, perfil.topic('topic_objeto'), objeto, 10)

    print('Perfil {}: escuchando {:.0f} s...'.format(perfil.nombre, args.segundos))
    fin = time.monotonic() + args.segundos
    while time.monotonic() < fin:
        rclpy.spin_once(nodo, timeout_sec=0.05)

    print('\n== Topics ==')
    cam = perfil.camara
    print('{}: {:.1f} Hz (esperado {} Hz), {} mensajes'.format(
        perfil.topic('topic_imagen'), imagen.frecuencia(), cam.frecuencia_esperada_hz,
        len(imagen.llegadas)))
    if imagen.retardos:
        print('  retardo cabecera->recepción: medio {:.3f} s, máx {:.3f} s '
              '(antigüedad máx del perfil {} s)'.format(
                  sum(imagen.retardos) / len(imagen.retardos), max(imagen.retardos),
                  cam.antiguedad_max_imagen_s))
    if imagen.ultimo is not None:
        w, h = imagen.ultimo.width, imagen.ultimo.height
        ok = (w, h) == (cam.resolucion_ancho_px, cam.resolucion_alto_px)
        print('  resolución {}x{} px, codificación {} -> {}'.format(
            w, h, imagen.ultimo.encoding,
            'coincide con el perfil' if ok else
            'NO coincide con camara.resolucion_* ({}x{})'.format(
                cam.resolucion_ancho_px, cam.resolucion_alto_px)))
    print('{}: {:.1f} Hz, {} mensajes'.format(
        perfil.topic('topic_objeto'), objeto.frecuencia(), len(objeto.llegadas)))
    if objeto.ultimo is not None:
        m = objeto.ultimo
        print('  último: label={!r} conf={:.2f} x={} y={} w={} h={} ts={}'.format(
            m.label, m.confidence, m.x, m.y, m.width, m.height, m.timestamp))
        print('  timestamps distintos: {} de {} mensajes (los repetidos son '
              'detecciones antiguas reenviadas por el puente)'.format(
                  len(objeto.timestamps), len(objeto.llegadas)))

    print('\n== Servicios y acciones ==')
    servicios = {n for n, _ in nodo.get_service_names_and_types()}
    esperados = [perfil.topic('srv_iniciar_reconocimiento')]
    if perfil.simulacion is not None:
        esperados.append(perfil.simulacion.srv_reset)
    for nombre in esperados:
        print('  {:<55} {}'.format(nombre, 'OK' if nombre in servicios else 'NO ENCONTRADO'))
    acciones = {n for n, _ in nodo.get_topic_names_and_types()}
    for clave in ('accion_tilt', 'accion_pan'):
        nombre = perfil.topic(clave)
        presente = any(t.startswith(nombre + '/_action') for t in acciones)
        print('  {:<55} {}'.format(nombre, 'OK' if presente else 'NO ENCONTRADO'))

    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
