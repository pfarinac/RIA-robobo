#!/usr/bin/env python3
"""
Versión "plana" de la demostración de robobo_ros2.

Reproduce exactamente la misma secuencia de nueve pasos que sample_demo.py,
pero SIN encapsular ninguna llamada en funciones ni en una clase. Todo el
código está escrito en línea, en el orden en que se ejecuta.

El objetivo es didáctico: ver el trabajo real que hay detrás de cada
orden que se le da al robot, y entender por qué sample_demo.py agrupa
ese trabajo en métodos.

Requisitos: el nodo robobo_container debe estar en ejecución en otra
terminal, conectado al simulador o al robot:

    ros2 run robobo_ros2 robobo_container --ros-args \
         -p ip:=host.docker.internal -p modules:="['blob','emotion']"

Ejecución:

    python3 demo_plano.py
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from std_msgs.msg import Int32MultiArray

from robobo_ros2_interfaces.srv import SetLed
from robobo_ros2_interfaces.action import MovePan, MoveTilt, MoveWheelsTime


# ---------------------------------------------------------------------
# Preparación: nodo y clientes
# ---------------------------------------------------------------------

rclpy.init()
nodo = Node('demo_plano')

# Espacio de nombres del nodo puente. Con los valores por defecto de
# robobo_container, robot_name vale "0".
base = '/robobo/robot_0/base'

# Los clientes se crean UNA sola vez, antes de usarlos. Crear un cliente
# justo antes de cada llamada es un error habitual: el descubrimiento de
# ROS 2 tarda unos instantes y el cliente todavía no estaría listo.
cliente_led = nodo.create_client(SetLed, base + '/set_led')
accion_ruedas = ActionClient(nodo, MoveWheelsTime, base + '/move_wheels_time')
accion_pan = ActionClient(nodo, MovePan, base + '/move_pan')
accion_tilt = ActionClient(nodo, MoveTilt, base + '/move_tilt')

# Un topic no se "pide": el robot publica cuando quiere y nosotros
# declaramos una suscripción con una función de retorno (callback) que se
# ejecutará con cada mensaje.
#
# Aquí aparece el primer límite de este estilo plano: la suscripción EXIGE
# una función. No hay forma de evitarlo. Se usa una lambda que va dejando
# las lecturas en una lista.
lecturas_ir = []
suscripcion_ir = nodo.create_subscription(
    Int32MultiArray,
    base + '/ir',
    lambda mensaje: lecturas_ir.append(list(mensaje.data)),
    10)


# ---------------------------------------------------------------------
# Espera a que el nodo puente esté disponible
# ---------------------------------------------------------------------

nodo.get_logger().info('Esperando a robobo_container...')

if not cliente_led.wait_for_service(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra el servicio set_led. '
                            '¿Está robobo_container en ejecución?')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

if not accion_ruedas.wait_for_server(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra la acción move_wheels_time.')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

if not accion_pan.wait_for_server(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra la acción move_pan.')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

if not accion_tilt.wait_for_server(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra la acción move_tilt.')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

nodo.get_logger().info('Nodo puente localizado. Comienza la secuencia.')


# ---------------------------------------------------------------------
# Paso 1 de 9: leer los sensores infrarrojos   [TOPIC]
# ---------------------------------------------------------------------

nodo.get_logger().info('[1/9] Lectura inicial de los sensores infrarrojos')

# El callback sólo se ejecuta mientras se hace girar el nodo. Por eso hay
# que vaciar la lista, girar, y comprobar si ha llegado algo.
lecturas_ir.clear()
instante_inicial = time.time()

while not lecturas_ir and (time.time() - instante_inicial) < 2.0:
    rclpy.spin_once(nodo, timeout_sec=0.1)

if lecturas_ir:
    # Orden de los ocho valores:
    # FrontLL, FrontL, FrontC, FrontR, FrontRR, BackL, BackC, BackR
    nodo.get_logger().info('  IR: ' + str(lecturas_ir[-1]))
else:
    nodo.get_logger().warning('  No se ha recibido ninguna lectura en 2 s')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 2 de 9: encender todos los LED en rojo   [SERVICIO]
# ---------------------------------------------------------------------

nodo.get_logger().info('[2/9] LED en rojo')

peticion = SetLed.Request()
peticion.led = 'All'
peticion.color = 'RED'

futuro = cliente_led.call_async(peticion)              # no bloquea
rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=5.0)

if not futuro.done():
    nodo.get_logger().error('  Tiempo agotado en set_led')
else:
    respuesta = futuro.result()
    if respuesta is None or not respuesta.success:
        nodo.get_logger().error('  set_led ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 3 de 9: avanzar durante 2 segundos   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[3/9] Avanzar 2 s')

meta = MoveWheelsTime.Goal()
meta.right_speed = 50.0
meta.left_speed = 50.0
meta.time = 2.0

# Primer futuro: el servidor acepta o rechaza la meta.
futuro_meta = accion_ruedas.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de ruedas')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de ruedas rechazada')
    else:
        # Segundo futuro: el resultado, cuando el movimiento termina.
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=12.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de ruedas ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 4 de 9: LED en azul   [SERVICIO]
# ---------------------------------------------------------------------

nodo.get_logger().info('[4/9] LED en azul')

peticion = SetLed.Request()
peticion.led = 'All'
peticion.color = 'BLUE'

futuro = cliente_led.call_async(peticion)
rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=5.0)

if not futuro.done():
    nodo.get_logger().error('  Tiempo agotado en set_led')
else:
    respuesta = futuro.result()
    if respuesta is None or not respuesta.success:
        nodo.get_logger().error('  set_led ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 5 de 9: girar el soporte del móvil a 45 grados   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[5/9] Pan a 45 grados')

meta = MovePan.Goal()
meta.angle = 45.0
meta.speed = 30.0

futuro_meta = accion_pan.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de pan')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de pan rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=10.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado de pan')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de pan ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 6 de 9: inclinar el soporte a 30 grados   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[6/9] Tilt a 30 grados')

meta = MoveTilt.Goal()
meta.angle = 30.0
meta.speed = 20.0

futuro_meta = accion_tilt.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de tilt')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de tilt rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=10.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado de tilt')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de tilt ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 7 de 9: devolver pan y tilt a su posición neutra   [DOS ACCIONES]
# ---------------------------------------------------------------------

nodo.get_logger().info('[7/9] Pan a 0 grados y tilt a 5 grados')

meta = MovePan.Goal()
meta.angle = 0.0
meta.speed = 30.0

futuro_meta = accion_pan.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de pan')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de pan rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=10.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado de pan')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de pan ha fallado')

meta = MoveTilt.Goal()
meta.angle = 5.0
meta.speed = 20.0

futuro_meta = accion_tilt.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de tilt')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de tilt rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=10.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado de tilt')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de tilt ha fallado')

time.sleep(0.5)


# ---------------------------------------------------------------------
# Paso 8 de 9: LED en verde   [SERVICIO]
# ---------------------------------------------------------------------

nodo.get_logger().info('[8/9] LED en verde')

peticion = SetLed.Request()
peticion.led = 'All'
peticion.color = 'GREEN'

futuro = cliente_led.call_async(peticion)
rclpy.spin_until_future_complete(nodo, futuro, timeout_sec=5.0)

if not futuro.done():
    nodo.get_logger().error('  Tiempo agotado en set_led')
else:
    respuesta = futuro.result()
    if respuesta is None or not respuesta.success:
        nodo.get_logger().error('  set_led ha fallado')


# ---------------------------------------------------------------------
# Paso 9 de 9: volver a leer los sensores infrarrojos   [TOPIC]
# ---------------------------------------------------------------------

nodo.get_logger().info('[9/9] Lectura final de los sensores infrarrojos')

lecturas_ir.clear()
instante_inicial = time.time()

while not lecturas_ir and (time.time() - instante_inicial) < 2.0:
    rclpy.spin_once(nodo, timeout_sec=0.1)

if lecturas_ir:
    nodo.get_logger().info('  IR: ' + str(lecturas_ir[-1]))
else:
    nodo.get_logger().warning('  No se ha recibido ninguna lectura en 2 s')


# ---------------------------------------------------------------------
# Cierre
# ---------------------------------------------------------------------

nodo.get_logger().info('Secuencia finalizada.')

nodo.destroy_node()
rclpy.shutdown()
