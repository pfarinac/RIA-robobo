#!/usr/bin/env python3
"""
Versión "plana" en ROS 2 del comportamiento.py (robobopy).

El robot hace lo siguiente:

    1. Inclina el soporte del móvil a 90 grados para que la cámara mire
       en vertical.
    2. Gira sobre sí mismo, despacio, hasta que la bola verde queda en el
       centro de la imagen (posición horizontal entre el 45 % y el 55 %).
    3. Se para un segundo.
    4. Avanza hacia la pared con un control proporcional: cuanto más lejos
       está, más rápido va. Cuando el error baja de 40 se detiene.

Igual que basic_demo.py, todo el código está escrito en línea, sin
funciones ni clases, en el orden en que se ejecuta.

Requisitos: el nodo robobo_container debe estar en ejecución en otra
terminal, con el módulo 'blob' cargado:

    ros2 run robobo_ros2 robobo_container --ros-args \
         -p ip:=host.docker.internal -p modules:="['blob']"

Ejecución:

    python3 comportamiento_plano.py
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rosidl_runtime_py.utilities import get_message

from std_msgs.msg import Int32MultiArray

from robobo_ros2_interfaces.action import MoveTilt, MoveWheelsTime


# ---------------------------------------------------------------------
# Parámetros del comportamiento 
# ---------------------------------------------------------------------

kp = 0.4                  # Ganancia del controlador proporcional
goal = 100                # Valor de IR objetivo
umbral_error = 40         # Por debajo de este error, hemos llegado
velocidad_giro = 5.0      # Velocidad de las ruedas mientras se busca la bola
centro_min = 45           # Límites del "centro" de la imagen (en %)
centro_max = 55

# Color de la bola a buscar. Se compara en minúsculas.
color_objetivo = 'green'

# Duración de cada empujón de ruedas. La acción move_wheels_time se para
# sola al cumplirse el tiempo, así que si el script se interrumpe el robot
# no se queda corriendo.
paso_giro = 0.4
paso_avance = 0.5


# ---------------------------------------------------------------------
# Preparación: nodo y clientes
# ---------------------------------------------------------------------

rclpy.init()
nodo = Node('comportamiento_plano')

base = '/robobo/robot_0/base'

accion_ruedas = ActionClient(nodo, MoveWheelsTime, base + '/move_wheels_time')
accion_tilt = ActionClient(nodo, MoveTilt, base + '/move_tilt')

# Sensores infrarrojos. Orden de los ocho valores:
# FrontLL, FrontL, FrontC, FrontR, FrontRR, BackL, BackC, BackR
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

if not accion_ruedas.wait_for_server(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra la acción move_wheels_time. '
                            '¿Está robobo_container en ejecución?')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

if not accion_tilt.wait_for_server(timeout_sec=15.0):
    nodo.get_logger().error('No se encuentra la acción move_tilt.')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)


# ---------------------------------------------------------------------
# Suscripción a los blobs de color   [TOPIC]
# ---------------------------------------------------------------------
#
# El topic de blobs pertenece al módulo del smartphone, no a la base, y su
# nombre y su tipo de mensaje dependen de la versión de robobo_ros2. En
# lugar de fijarlos a mano, se busca en el grafo de ROS 2 un topic cuyo
# nombre contenga "blob" y se lee su tipo de mensaje.

nodo.get_logger().info('Buscando el topic de blobs...')

topic_blob = None
tipo_blob = None
instante_inicial = time.time()

while topic_blob is None and (time.time() - instante_inicial) < 15.0:
    for nombre, tipos in nodo.get_topic_names_and_types():
        if 'blob' in nombre.lower():
            topic_blob = nombre
            tipo_blob = tipos[0]
            break
    rclpy.spin_once(nodo, timeout_sec=0.5)

if topic_blob is None:
    nodo.get_logger().error('No se encuentra ningún topic de blobs. '
                            '¿Está cargado el módulo "blob" en robobo_container?')
    nodo.destroy_node()
    rclpy.shutdown()
    raise SystemExit(1)

clase_blob = get_message(tipo_blob)

# Se muestran los campos del mensaje (BlobArray) a modo de comprobación.
nodo.get_logger().info('Topic de blobs: ' + topic_blob + ' (' + tipo_blob + ')')
nodo.get_logger().info('  Campos: ' + str(clase_blob.get_fields_and_field_types()))

ultimos_blobs = []
suscripcion_blob = nodo.create_subscription(
    clase_blob,
    topic_blob,
    lambda mensaje: ultimos_blobs.append(mensaje),
    10)

nodo.get_logger().info('Nodo puente localizado. Comienza el comportamiento.')


# ---------------------------------------------------------------------
# Paso 1 de 4: inclinar la cámara a 90 grados   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[1/4] Tilt a 90 grados')

meta = MoveTilt.Goal()
meta.angle = 90.0
meta.speed = 30.0

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
# Paso 2 de 4: girar hasta centrar la bola verde   [ACCIÓN + TOPIC]
# ---------------------------------------------------------------------
#
# En robobopy esto lo hacía un callback (whenANewColorBlobIsDetected).
# Aquí no hay callback que dispare la lógica: la suscripción solo va
# guardando mensajes y es el bucle principal quien los revisa después de
# cada pequeño giro. Los mensajes llegan mientras se espera el resultado
# de la acción, porque spin_until_future_complete también atiende la
# suscripción.

nodo.get_logger().info('[2/4] Buscando la bola ' + color_objetivo)

centrada = False

while not centrada:
    ultimos_blobs.clear()

    # Un giro corto sobre sí mismo: rueda izquierda hacia atrás y derecha
    # hacia delante (equivale a moveWheels(-5, 5)).
    meta = MoveWheelsTime.Goal()
    meta.left_speed = -velocidad_giro
    meta.right_speed = velocidad_giro
    meta.time = paso_giro

    futuro_meta = accion_ruedas.send_goal_async(meta)
    rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

    if not futuro_meta.done():
        nodo.get_logger().error('  Tiempo agotado al enviar la meta de ruedas')
    else:
        gestor = futuro_meta.result()
        if gestor is None or not gestor.accepted:
            nodo.get_logger().error('  Meta de ruedas rechazada')
        else:
            futuro_resultado = gestor.get_result_async()
            rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=5.0)

            if not futuro_resultado.done():
                nodo.get_logger().error('  Tiempo agotado esperando el resultado')
            else:
                resultado = futuro_resultado.result()
                if resultado is None or not resultado.result.success:
                    nodo.get_logger().error('  El movimiento de ruedas ha fallado')

    # El mensaje es un BlobArray: su campo "blobs" es una lista con un
    # blob por cada color detectado. Cada blob tiene "color" (texto),
    # "x" e "y" (posición en la imagen, de 0 a 100) y "size".
    x_bola = None
    for mensaje_blob in ultimos_blobs:
        for blob in mensaje_blob.blobs:
            if blob.color.lower() == color_objetivo:
                x_bola = blob.x

    if x_bola is not None:
        nodo.get_logger().info('  Bola vista en x = ' + str(round(x_bola, 1)))
        if centro_min <= x_bola <= centro_max:
            centrada = True


# ---------------------------------------------------------------------
# Paso 3 de 4: parar un segundo   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[3/4] Bola centrada: parada de 1 s')

meta = MoveWheelsTime.Goal()
meta.left_speed = 0.0
meta.right_speed = 0.0
meta.time = 0.1

futuro_meta = accion_ruedas.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de ruedas')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de ruedas rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=5.0)

        if not futuro_resultado.done():
            nodo.get_logger().error('  Tiempo agotado esperando el resultado')
        else:
            resultado = futuro_resultado.result()
            if resultado is None or not resultado.result.success:
                nodo.get_logger().error('  El movimiento de ruedas ha fallado')

time.sleep(1.0)


# ---------------------------------------------------------------------
# Paso 4 de 4: avanzar hacia la pared con control proporcional
#              [TOPIC + ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('[4/4] Avanzando hacia la pared')

llegado = False

while not llegado:
    # Se descarta lo antiguo y se espera una lectura de IR reciente.
    lecturas_ir.clear()
    instante_inicial = time.time()

    while not lecturas_ir and (time.time() - instante_inicial) < 2.0:
        rclpy.spin_once(nodo, timeout_sec=0.1)

    if not lecturas_ir:
        # Sin sensores no es seguro seguir avanzando.
        nodo.get_logger().error('  No se ha recibido ninguna lectura de IR en 2 s')
        break

    ir = lecturas_ir[-1]

    # Sensores frontales: FrontC (índice 2), FrontLL (0) y FrontRR (4).
    closest = max(ir[2], ir[0], ir[4])
    error = goal - closest
    velocidad = error * kp

    nodo.get_logger().info('  IR más cercano: ' + str(closest)
                           + '  error: ' + str(error)
                           + '  velocidad: ' + str(round(velocidad, 1)))

    if error < umbral_error:
        llegado = True
    else:
        meta = MoveWheelsTime.Goal()
        meta.left_speed = float(velocidad)
        meta.right_speed = float(velocidad)
        meta.time = paso_avance

        futuro_meta = accion_ruedas.send_goal_async(meta)
        rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

        if not futuro_meta.done():
            nodo.get_logger().error('  Tiempo agotado al enviar la meta de ruedas')
        else:
            gestor = futuro_meta.result()
            if gestor is None or not gestor.accepted:
                nodo.get_logger().error('  Meta de ruedas rechazada')
            else:
                futuro_resultado = gestor.get_result_async()
                rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=5.0)

                if not futuro_resultado.done():
                    nodo.get_logger().error('  Tiempo agotado esperando el resultado')
                else:
                    resultado = futuro_resultado.result()
                    if resultado is None or not resultado.result.success:
                        nodo.get_logger().error('  El movimiento de ruedas ha fallado')


# ---------------------------------------------------------------------
# Parada final   [ACCIÓN]
# ---------------------------------------------------------------------

nodo.get_logger().info('Parada final de motores')

meta = MoveWheelsTime.Goal()
meta.left_speed = 0.0
meta.right_speed = 0.0
meta.time = 0.1

futuro_meta = accion_ruedas.send_goal_async(meta)
rclpy.spin_until_future_complete(nodo, futuro_meta, timeout_sec=5.0)

if not futuro_meta.done():
    nodo.get_logger().error('  Tiempo agotado al enviar la meta de ruedas')
else:
    gestor = futuro_meta.result()
    if gestor is None or not gestor.accepted:
        nodo.get_logger().error('  Meta de ruedas rechazada')
    else:
        futuro_resultado = gestor.get_result_async()
        rclpy.spin_until_future_complete(nodo, futuro_resultado, timeout_sec=5.0)


# ---------------------------------------------------------------------
# Cierre
# ---------------------------------------------------------------------

nodo.get_logger().info('Comportamiento finalizado.')

nodo.destroy_node()
rclpy.shutdown()
