#!/usr/bin/env python3
"""Mide la latencia orden -> imagen (enunciado 3.3: frecuencia y retardo).

Con el robot parado, ordena un giro corto sobre sí mismo por cmd_vel y mide
cuánto tarda en cambiar la imagen de camera/image_raw respecto a la de
referencia. Incluye el retardo de actuación más el de la cámara: es el
retardo que "ve" la política entre actuar y observar el efecto. Repite N
veces y da la mediana.

    python3 herramientas/medir_latencia_camara.py --perfil sim --repeticiones 3
    python3 herramientas/medir_latencia_camara.py --perfil real

Si la latencia es comparable al periodo de control, la política debe
recibirla en la observación (acciones anteriores) o el periodo debe crecer.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402
from ros_carril import imagen_a_numpy  # noqa: E402


def reducida(msg):
    gris = cv2.cvtColor(imagen_a_numpy(msg), cv2.COLOR_BGR2GRAY)
    return cv2.resize(gris, (60, 80), interpolation=cv2.INTER_AREA).astype(np.float32)


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--repeticiones', type=int, default=3)
    parser.add_argument('--w', type=float, default=0.3, help='rad/s del giro de prueba')
    parser.add_argument('--duracion', type=float, default=1.0, help='s de giro')
    parser.add_argument('--umbral', type=float, default=6.0,
                        help='diferencia media de gris (0-255) que cuenta como cambio')
    parser.add_argument('--timeout', type=float, default=10.0)
    args = parser.parse_args()

    perfil = cargar_perfil(args.perfil)
    rclpy.init()
    nodo = Node('p3_medir_latencia')
    estado = {'n': 0}

    def cb(msg):
        estado.update(n=estado['n'] + 1, msg=msg, t=time.monotonic())

    nodo.create_subscription(Image, perfil.topic('topic_imagen'), cb, 1)
    pub = nodo.create_publisher(Twist, perfil.topic('topic_cmd_vel'), 10)
    periodo = float(perfil.seguridad.periodo_reenvio_cmd_vel_s)

    def spin(s):
        fin = time.monotonic() + s
        while time.monotonic() < fin:
            rclpy.spin_once(nodo, timeout_sec=0.02)

    def publicar(w):
        t = Twist()
        t.angular.z = float(w)
        pub.publish(t)

    latencias = []
    signo = 1.0
    try:
        for r in range(args.repeticiones):
            publicar(0.0)
            spin(max(3.0, args.timeout / 3))           # que se asiente la imagen
            if 'msg' not in estado:
                sys.exit('No llegan imágenes de ' + perfil.topic('topic_imagen'))
            ref = reducida(estado['msg'])
            t_orden = time.monotonic()
            cambio = None
            while time.monotonic() - t_orden < args.timeout:
                if time.monotonic() - t_orden < args.duracion:
                    publicar(signo * args.w)
                else:
                    publicar(0.0)
                spin(periodo)
                if estado.get('t', 0) > t_orden:
                    if float(np.abs(reducida(estado['msg']) - ref).mean()) > args.umbral:
                        cambio = estado['t'] - t_orden
                        break
            publicar(0.0)
            signo = -signo                             # alternar para volver a la posición
            print('Repetición {}: {}'.format(
                r + 1, 'latencia {:.2f} s'.format(cambio) if cambio else 'sin cambio detectado'))
            if cambio:
                latencias.append(cambio)
            spin(args.duracion + 1.0)
            # deshacer el giro
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.duracion:
                publicar(signo * args.w)
                spin(periodo)
            publicar(0.0)
            signo = -signo
    finally:
        publicar(0.0)
    if latencias:
        print('\nLatencia orden -> imagen: mediana {:.2f} s (mín {:.2f}, máx {:.2f}); '
              'periodo de control del perfil {:.2f} s'.format(
                  statistics.median(latencias), min(latencias), max(latencias),
                  perfil.seguridad.periodo_control_s))
    nodo.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
