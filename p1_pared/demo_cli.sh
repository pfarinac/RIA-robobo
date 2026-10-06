#!/usr/bin/env bash
#
# Versión por línea de comandos de la demostración de robobo_ros2.
#
# Reproduce la misma secuencia de nueve pasos que sample_demo.py, pero
# empleando únicamente las herramientas de línea de comandos de ROS 2.
# No hay control de errores: cada orden se envía y se da por buena.
#
# Requisito previo: el nodo puente debe estar en ejecución en otra
# terminal del contenedor, conectado al simulador o al robot:
#
#     ros2 run robobo_ros2 robobo_container --ros-args \
#          -p ip:=host.docker.internal -p modules:="['blob','emotion']"
#
# Ejecución:
#
#     bash demo_cli.sh
#
# Nota sobre los números: todos los campos son de tipo float32, así que
# deben escribirse con punto decimal. Si se escribe 50 en lugar de 50.0,
# YAML lo interpreta como entero y la llamada falla.

# Espacio de nombres del nodo puente. Si robobo_container se lanzó con
# otro robot_name, hay que cambiar el 0 por ese valor.
NS=/robobo/robot_0/base

SRV=robobo_ros2_interfaces/srv
ACT=robobo_ros2_interfaces/action


echo "[1/9] Lectura inicial de los sensores infrarrojos   [topic]"
# A un topic no se le pide nada: se escucha. Con --once se toma una sola
# publicación y el comando termina; sin --once se queda escuchando hasta
# que se interrumpa con Ctrl-C.
# Orden de los ocho valores del array:
#   FrontLL, FrontL, FrontC, FrontR, FrontRR, BackL, BackC, BackR
ros2 topic echo --once $NS/ir
sleep 0.5


echo "[2/9] Encender todos los LED en rojo   [servicio]"
ros2 service call $NS/set_led $SRV/SetLed "{led: 'All', color: 'RED'}"
sleep 0.5


echo "[3/9] Avanzar 2 s a velocidad 50   [acción]"
ros2 action send_goal $NS/move_wheels_time $ACT/MoveWheelsTime \
    "{right_speed: 50.0, left_speed: 50.0, time: 2.0}"
sleep 0.5


echo "[4/9] Encender todos los LED en azul   [servicio]"
ros2 service call $NS/set_led $SRV/SetLed "{led: 'All', color: 'BLUE'}"
sleep 0.5


echo "[5/9] Girar el soporte del móvil a 45 grados   [acción]"
ros2 action send_goal $NS/move_pan $ACT/MovePan \
    "{angle: 45.0, speed: 30.0}"
sleep 0.5


echo "[6/9] Inclinar el soporte a 30 grados   [acción]"
ros2 action send_goal $NS/move_tilt $ACT/MoveTilt \
    "{angle: 30.0, speed: 20.0}"
sleep 0.5


echo "[7/9] Devolver el soporte a la posición neutra   [dos acciones]"
ros2 action send_goal $NS/move_pan $ACT/MovePan \
    "{angle: 0.0, speed: 30.0}"
ros2 action send_goal $NS/move_tilt $ACT/MoveTilt \
    "{angle: 5.0, speed: 20.0}"
sleep 0.5


echo "[8/9] Encender todos los LED en verde   [servicio]"
ros2 service call $NS/set_led $SRV/SetLed "{led: 'All', color: 'GREEN'}"

sleep 0.5


echo "[9/9] Lectura final de los sensores infrarrojos   [topic]"
ros2 topic echo --once $NS/ir

echo "Secuencia finalizada."


# ---------------------------------------------------------------------
# EJERCICIO: servicio frente a acción
# ---------------------------------------------------------------------
#
# move_wheels_time, move_pan y move_tilt están publicados a la vez como
# servicio y como acción, con el mismo nombre. Ejecutar las dos órdenes
# siguientes, una detrás de otra, y observar la diferencia:
#
#   # El servicio devuelve el prompt de inmediato: el robot sigue en
#   # movimiento cuando la terminal ya está libre.
#   ros2 service call $NS/move_wheels_time $SRV/MoveWheelsTime \
#       "{right_speed: 50.0, left_speed: 50.0, time: 3.0}"
#
#   # La acción no devuelve hasta que el movimiento ha terminado.
#   # Con -f se muestra además el avance mientras se ejecuta.
#   ros2 action send_goal -f $NS/move_wheels_time $ACT/MoveWheelsTime \
#       "{right_speed: 50.0, left_speed: 50.0, time: 3.0}"
#
# Ésa es la razón por la que un bucle de control emplea acciones: el
# servicio no informa de cuándo ha concluido la orden.
#
#
# ---------------------------------------------------------------------
# EJERCICIO: observar los sensores mientras el robot se mueve
# ---------------------------------------------------------------------
#
# En una terminal, dejar los infrarrojos publicando de forma continua:
#
#   ros2 topic echo $NS/ir
#
# En otra, acercar el robot a un obstáculo y observar cómo cambian los
# valores. Ésa es la diferencia de fondo con los servicios y las acciones:
# el topic no se pide, llega solo.
#
# Cada sensor está publicado además por separado, lo que evita depender
# del orden del array:
#
#   ros2 topic echo $NS/ir/frontc
#   ros2 topic echo $NS/irs/frontc/range     # como sensor_msgs/Range
#
# Frecuencia de publicación y número de publicadores:
#
#   ros2 topic hz $NS/ir
#   ros2 topic info $NS/ir
#
#
# Para consultar los campos de cualquier interfaz:
#
#   ros2 interface show robobo_ros2_interfaces/srv/SetLed
#   ros2 interface show robobo_ros2_interfaces/action/MoveWheelsTime
#   ros2 interface show std_msgs/msg/Int32MultiArray
#
# Para ver qué ofrece el robot:
#
#   ros2 topic list | grep robobo
#   ros2 service list | grep robobo
#   ros2 action list
