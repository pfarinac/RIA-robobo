#!/usr/bin/env python3
"""Mueve el Robobo hacia delante en RoboboSim durante un tiempo limitado.

No usa la percepción del guía ni del carril. Publica cmd_vel a la frecuencia
del perfil y envía una orden de parada al terminar o al pulsar Ctrl+C.

    python3 herramientas/avanzar_recto.py --perfil sim --velocidad 0.05 --duracion 10
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from perfil import anadir_argumento_perfil, cargar_perfil


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--velocidad', type=float, default=0.05,
                        help='velocidad lineal en m/s (por defecto: 0.05)')
    parser.add_argument('--duracion', type=float, default=10.0,
                        help='duración en segundos, máximo 30 (por defecto: 10)')
    args = parser.parse_args()

    perfil = cargar_perfil(args.perfil)
    if not perfil.es_simulacion:
        parser.error('este script solo permite el perfil sim')
    if not 0.0 < args.velocidad <= perfil.politica.v_max_m_s:
        parser.error('--velocidad debe ser mayor que 0 y no superar {:.3f} m/s'
                     .format(perfil.politica.v_max_m_s))
    if not 0.0 < args.duracion <= 30.0:
        parser.error('--duracion debe estar entre 0 y 30 segundos')

    rclpy.init()
    nodo = Node('p3_avanzar_recto')
    publicador = nodo.create_publisher(
        Twist, perfil.topic('topic_cmd_vel'), 10)
    periodo = float(perfil.seguridad.periodo_reenvio_cmd_vel_s)

    def publicar(lineal):
        mensaje = Twist()
        mensaje.linear.x = float(lineal * perfil.politica.escala_v)
        publicador.publish(mensaje)

    try:
        limite_conexion = time.monotonic() + 5.0
        while (rclpy.ok()
               and publicador.get_subscription_count() == 0
               and time.monotonic() < limite_conexion):
            rclpy.spin_once(nodo, timeout_sec=0.1)
        if publicador.get_subscription_count() == 0:
            raise RuntimeError(
                'No hay suscriptor ROS para {}. Comprueba que el puente está activo.'
                .format(perfil.topic('topic_cmd_vel')))

        print('Avance recto a {:.3f} m/s durante {:.1f} s. Ctrl+C para parar.'
              .format(args.velocidad, args.duracion), flush=True)
        fin = time.monotonic() + args.duracion
        while rclpy.ok() and time.monotonic() < fin:
            publicar(args.velocidad)
            restante = fin - time.monotonic()
            if restante > 0:
                rclpy.spin_once(nodo, timeout_sec=min(periodo, restante))
    except KeyboardInterrupt:
        print('\nParada solicitada.', flush=True)
    finally:
        publicar(0.0)
        rclpy.spin_once(nodo, timeout_sec=0.05)
        publicar(0.0)
        rclpy.spin_once(nodo, timeout_sec=0.05)
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
