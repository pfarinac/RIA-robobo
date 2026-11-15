#!/usr/bin/env python3
"""Entorno Gymnasium para seguir al vehículo guía en Follow City (Adaptado para NEAT).

El entorno envuelve la comunicación ROS 2 con RoboboSim:
  - Observación (8 valores): visible, err_x, err_y, tam, last_x, t_perdido,
    delta_tam, delta_err_x (todos derivados del blob verde de la cámara).
  - Acción (2 valores en [-1, 1]): velocidad lineal y velocidad angular.
  - No calcula la recompensa (devuelve 0.0): en P2 la calidad se calcula fuera,
    en p2_entrenar.py, a partir del diccionario `info` que devuelve cada paso.
"""

import time
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from rclpy.node import Node
from robobo_ros2_interfaces.action import MoveTilt
from robobo_ros2_interfaces.srv import ResetSimulation, SetActiveColorBlobs
from robobo_ros2_interfaces.msg import RobotLocation
from std_msgs.msg import Int32MultiArray

# Algunos tipos de mensaje pueden no existir según la versión de la interfaz:
# se importan con try/except y se comprueban en __init__ para dar un error claro.
try:
    from robobo_ros2_interfaces.msg import BlobArray
except ImportError:
    BlobArray = None

try:
    from robobo_ros2_interfaces.msg import SimObjectArray
except ImportError:
    SimObjectArray = None

import follow_city_config as config

# Namespaces ROS 2 del robot seguidor: base (motores, IR), smartphone (cámara) y sim (simulador).
NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'

class FollowCityEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, verbose=False, episode_seconds=None):
        super().__init__()
        if BlobArray is None:
            raise ImportError('No se encuentra robobo_ros2_interfaces/BlobArray.')
        if SimObjectArray is None:
            raise ImportError('No se encuentra robobo_ros2_interfaces/SimObjectArray.')

        # Un único nodo ROS 2 por proceso: si rclpy ya estaba iniciado no se reinicia.
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('follow_city_rl')
        self.verbose = verbose

        # Publicador de órdenes de velocidad y cliente para reiniciar el simulador.
        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)
        self.cli_reset = self.nodo.create_client(ResetSimulation, NS_SIM + '/reset_simulation')
        if not self.cli_reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'No responde {NS_SIM}/reset_simulation.')

        # Acción ROS 2 para inclinar la cámara (tilt).
        self.accion_tilt = ActionClient(self.nodo, MoveTilt, NS_BASE + '/move_tilt')
        if not self.accion_tilt.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'No responde la acción {NS_BASE}/move_tilt.')

        # Suscripción a los blobs de color. Los contadores _n_* permiten saber si
        # ha llegado un mensaje nuevo desde que se miró por última vez.
        self._blobs = []
        self._n_blobs = 0
        self.nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)

        # Suscripción a los sensores infrarrojos de la base.
        self._ir = None
        self._n_ir = 0
        self.nodo.create_subscription(Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

        # Poses privilegiadas del simulador para medir la separación real:
        # la del seguidor viene de robot_location y la del guía de la lista de objetos.
        self._pose_seguidor = None
        self._pose_guia = None
        self._n_pose = 0
        if config.USE_SIM_SEPARATION:
            self.nodo.create_subscription(
                RobotLocation, config.FOLLOWER_SIM_NAMESPACE + '/robot_location',
                self._cb_pose_seguidor, 1)
            self.nodo.create_subscription(
                SimObjectArray, config.GUIDE_SIM_NAMESPACE + '/objects',
                self._cb_objects, 1)

        # Se activa solo el detector de color verde (la pelota que lleva el guía).
        self.cli_colores = self.nodo.create_client(SetActiveColorBlobs, NS_SMARTPHONE + '/set_active_color_blobs')
        if not self.cli_colores.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('No responde set_active_color_blobs.')

        solicitud = SetActiveColorBlobs.Request()
        solicitud.red = False
        solicitud.green = True
        solicitud.blue = False
        solicitud.custom = False
        futuro = self.cli_colores.call_async(solicitud)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)

        # Espacios. Orden de la observación:
        # [visible, err_x, err_y, tam, last_x, t_perdido, delta_tam, delta_err_x]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -2.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32)
        # Acción: [velocidad lineal, velocidad angular], ambas normalizadas a [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Duración máxima del episodio en pasos (30 s -> 120 pasos; 120 s -> 480).
        if episode_seconds is None:
            episode_seconds = config.EPISODE_SECONDS
        self.pasos_max = max(1, int(episode_seconds / config.REAL_STEP_S))

        # Estado interno entre pasos.
        self.pasos = 0
        self.inicio_perdida_blob = None             # instante en que se dejó de ver la pelota
        self.ultima_posicion_blob_x = None          # última x conocida (se usa si se pierde)
        self.instante_ultima_deteccion_blob = None
        self.giro_anterior = 0.0                    # velocidad angular aplicada en el paso previo
        self.velocidad_lineal_anterior = 0.0
        self.tamano_anterior = None                 # para calcular delta_tam
        self.error_x_anterior = None                # para calcular delta_err_x
        self.frame_ts = -1
        self._guia_ini = None                       # pose inicial del guía, para detectar su arranque
        self.guia_arrancado = False

    # ---------------- Callbacks de suscripción ----------------
    def _cb_blobs(self, msg):
        self._blobs = list(msg.blobs)
        self._n_blobs += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_ir += 1

    def _cb_pose_seguidor(self, msg):
        # Posición en el plano (x, z) del simulador; las unidades son milímetros.
        self._pose_seguidor = (float(msg.position.x), float(msg.position.z))
        self._n_pose += 1

    def _cb_pose_guia(self, msg):
        # No se suscribe en este archivo (la pose del guía llega por _cb_objects);
        # se conserva por si se usase una suscripción directa.
        self._pose_guia = (float(msg.position.x), float(msg.position.z))
        self._n_pose += 1

    def _cb_objects(self, msg):
        # Busca entre los objetos del simulador el que corresponde al guía.
        for objeto in msg.objects:
            if str(objeto.object_id) == config.GUIDE_SIM_OBJECT_ID:
                self._pose_guia = (
                    float(objeto.position.x),
                    float(objeto.position.z),
                )
                self._n_pose += 1
                return

    def _blob_verde(self):
        """Devuelve el blob verde más grande y válido, o None si no se ve ninguno."""
        verdes = [blob for blob in self._blobs if str(blob.color).lower() == 'green'
                  and float(blob.size) > 0.0 and int(blob.frame_timestamp) >= 0]
        return max(verdes, key=lambda blob: blob.size) if verdes else None

    # ---------------- Esperas activas de mensajes ----------------
    def _esperar_mensaje(self, despues_de, timeout):
        """Espera un mensaje de blobs nuevo (contador > despues_de) o lanza RuntimeError."""
        limite = time.monotonic() + timeout
        while self._n_blobs <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(f'No llegan mensajes de {NS_SMARTPHONE}/color_blobs.')
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _esperar_ir(self, despues_de, timeout):
        """Igual que _esperar_mensaje, pero para la lectura de infrarrojos."""
        limite = time.monotonic() + timeout
        while self._n_ir <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(f'No llegan lecturas de {NS_BASE}/ir.')
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _esperar_poses(self, timeout):
        """Espera a tener la pose del seguidor y la del guía. Devuelve False si se agota el tiempo."""
        limite = time.monotonic() + timeout
        while self._pose_seguidor is None or self._pose_guia is None:
            restante = limite - time.monotonic()
            if restante <= 0:
                return False
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))
        return True

    def _separacion(self, tamano_norm):
        """Devuelve (separación en metros, ¿procede del simulador?).

        Con poses del simulador: distancia euclídea entre ambos centros (mm -> m).
        Sin ellas: fallback visual que interpola la separación desde el tamaño del blob.
        """
        if self._pose_seguidor is not None and self._pose_guia is not None:
            dx = self._pose_seguidor[0] - self._pose_guia[0]
            dz = self._pose_seguidor[1] - self._pose_guia[1]
            return float(np.hypot(dx, dz) / 1000.0), True

        # Fallback para una ejecución sin el segundo contenedor: calibrar los
        # tamaños de blob de 0,20, 0,45 y 0,90 m en el escenario utilizado.
        puntos_tamano = np.array([
            config.TOO_FAR_BLOB_SIZE_NORM,
            config.TARGET_BLOB_SIZE_NORM,
            config.TOO_CLOSE_BLOB_SIZE_NORM,
        ])
        puntos_distancia = np.array([
            config.LOST_GUIDE_SEPARATION_M,
            config.MAX_SEPARATION_M,
            config.MIN_SEPARATION_M,
        ])
        return float(np.interp(tamano_norm, puntos_tamano, puntos_distancia)), False

    def _vaciar_cola(self):
        """Procesa los mensajes pendientes para quedarse con los más recientes (hasta 100 iteraciones)."""
        for _ in range(100):
            antes = (self._n_blobs, self._n_ir)
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if (self._n_blobs, self._n_ir) == antes:
                return

    def _proximidad_frontal(self):
        """Máxima proximidad normalizada [0, 1] entre los sensores IR frontales."""
        if self._ir is None:
            raise RuntimeError('Sin lectura IR.')
        return max(min(max(int(self._ir[i]), 0) / config.IR_SATURATION, 1.0) for i in config.FRONT_IR_INDICES)

    def _esperar_blob_verde(self, timeout=None):
        """Espera a ver la pelota verde (tras un reset) y actualiza la observación interna."""
        if timeout is None: timeout = config.SENSOR_TIMEOUT_S
        limite = time.monotonic() + timeout
        while self._blob_verde() is None:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError('No se ve la pelota verde.')
            self._esperar_mensaje(self._n_blobs, restante)
        self._observacion()

    def _observacion(self):
        """Construye el vector de 8 entradas de la red a partir del blob verde actual."""
        blob = self._blob_verde()
        if blob is None:
            # Pelota no visible: todo a 0 salvo last_x (última posición vista),
            # t_perdido (tiempo sin verla / LOST_BLOB_SECONDS, recortado a [0, 1])
            # y los deltas, que reflejan que el tamaño cae a 0.
            ahora = time.monotonic()
            if self.inicio_perdida_blob is None and self.instante_ultima_deteccion_blob is not None:
                self.inicio_perdida_blob = self.instante_ultima_deteccion_blob
            tiempo_perdido = 0.0
            if self.inicio_perdida_blob is not None:
                tiempo_perdido = np.clip((ahora - self.inicio_perdida_blob) / config.LOST_BLOB_SECONDS, 0.0, 1.0)
            ultima_posicion_x = self.ultima_posicion_blob_x if self.ultima_posicion_blob_x is not None else 0.0
            delta_tamano = 0.0 if self.tamano_anterior is None else -self.tamano_anterior
            delta_error_x = 0.0 if self.error_x_anterior is None else ultima_posicion_x - self.error_x_anterior
            self.tamano_anterior = 0.0
            self.error_x_anterior = ultima_posicion_x
            return np.array(
                [0.0, 0.0, 0.0, 0.0, ultima_posicion_x, tiempo_perdido,
                 np.clip(delta_tamano, -1.0, 1.0),
                 np.clip(delta_error_x, -2.0, 2.0)],
                dtype=np.float32)

        # Pelota visible: coordenadas del blob (0-100) centradas y normalizadas a [-1, 1];
        # el tamaño se normaliza a [0, 1] con BLOB_SIZE_MAX.
        error_x = np.clip((float(blob.x) - config.BLOB_COORD_MAX / 2) / (config.BLOB_COORD_MAX / 2), -1.0, 1.0)
        error_y = np.clip((float(blob.y) - config.BLOB_COORD_MAX / 2) / (config.BLOB_COORD_MAX / 2), -1.0, 1.0)
        tamano = np.clip(float(blob.size) / config.BLOB_SIZE_MAX, 0.0, 1.0)

        self.frame_ts = int(blob.frame_timestamp)
        self.ultima_posicion_blob_x = float(error_x)
        self.instante_ultima_deteccion_blob = time.monotonic()
        self.inicio_perdida_blob = None   # se ve la pelota: se cancela la cuenta de pérdida

        # Variaciones respecto al paso anterior (velocidad de acercamiento y de desplazamiento lateral).
        delta_tamano = 0.0 if self.tamano_anterior is None else tamano - self.tamano_anterior
        delta_error_x = 0.0 if self.error_x_anterior is None else error_x - self.error_x_anterior
        self.tamano_anterior = tamano
        self.error_x_anterior = error_x
        return np.array(
            [1.0, error_x, error_y, tamano, error_x, 0.0,
             np.clip(delta_tamano, -1.0, 1.0),
             np.clip(delta_error_x, -2.0, 2.0)],
            dtype=np.float32)

    # ---------------- Actuación y reinicio ----------------
    def _publicar_velocidad(self, lineal, angular):
        """Publica una orden de velocidad (m/s y rad/s) en cmd_vel."""
        mensaje = Twist()
        mensaje.linear.x = float(lineal)
        mensaje.angular.z = float(angular)
        self.pub_vel.publish(mensaje)

    def _detener(self):
        """Para el robot publicando velocidad 0 tres veces (para asegurar que llega)."""
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _reiniciar_simulacion(self):
        """Llama al servicio de reinicio del simulador y espera medio segundo."""
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=config.SENSOR_TIMEOUT_S)
        time.sleep(0.5)

    def _poner_tilt_inicial(self):
        """Inclina la cámara al ángulo inicial configurado."""
        objetivo = MoveTilt.Goal()
        objetivo.angle = float(config.TILT_ANGLE_DEG)
        objetivo.speed = float(config.TILT_SPEED_DEG_S)
        futuro = self.accion_tilt.send_goal_async(objetivo)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)

    def reset(self, seed=None, options=None):
        """Reinicia el simulador y devuelve la primera observación.

        Reintenta hasta RESET_ATTEMPTS veces si no llegan sensores, no se ve la
        pelota o no llegan las poses del simulador; en el último intento relanza el error.
        """
        super().reset(seed=seed)
        for intento in range(config.RESET_ATTEMPTS):
            self._detener()
            self._reiniciar_simulacion()
            self._poner_tilt_inicial()
            # Se borra el estado del episodio anterior.
            self.pasos = 0
            self.inicio_perdida_blob = None
            self.ultima_posicion_blob_x = None
            self.instante_ultima_deteccion_blob = None
            self.tamano_anterior = None
            self.error_x_anterior = None
            self.giro_anterior = 0.0
            self.velocidad_lineal_anterior = 0.0
            self._blobs = []
            self._pose_seguidor = None
            self._pose_guia = None

            secuencia = self._n_blobs
            secuencia_ir = self._n_ir

            try:
                # Esperar datos frescos de cámara e IR y ver la pelota verde.
                self._esperar_mensaje(secuencia, config.RESET_SENSOR_TIMEOUT_S)
                self._esperar_ir(secuencia_ir, config.SENSOR_TIMEOUT_S)
                self._esperar_blob_verde(config.RESET_SENSOR_TIMEOUT_S)
                # Dar tiempo al guía para ponerse en marcha y descartar mensajes antiguos.
                time.sleep(config.GUIDE_START_DELAY_S)
                self._vaciar_cola()
                self._esperar_mensaje(self._n_blobs, config.RESET_SENSOR_TIMEOUT_S)
                self._esperar_ir(self._n_ir, config.SENSOR_TIMEOUT_S)
                if not self._esperar_poses(config.RESET_SENSOR_TIMEOUT_S):
                    raise RuntimeError(
                        'No llegan robot_location: ¿está el contenedor sim del guía?')
                self._guia_ini = self._pose_guia   # pose de referencia para detectar el arranque del guía
                self.guia_arrancado = False
                self.instante_ultima_deteccion_blob = time.monotonic()
                self.tamano_anterior = None
                self.error_x_anterior = None
            except RuntimeError as error:
                if intento + 1 == config.RESET_ATTEMPTS:
                    raise
                continue
            return self._observacion(), {}
        raise RuntimeError('Fallo total al reiniciar.')

    def step(self, accion):
        """Aplica una acción durante un paso y devuelve (obs, 0.0, terminated, truncated, info)."""
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)

        # Arranque inicial recto, igual que en P1: los primeros pasos ignoran la red.
        if self.pasos < config.AUTO_FORWARD_STEPS:
            accion = np.array([1.0, 0.0], dtype=np.float32)

        # Velocidad lineal: la acción [-1, 1] se reescala a [MIN, MAX]. En los primeros pasos el tope es menor.
        velocidad_maxima = config.INITIAL_LINEAR_SPEED_M_S if self.pasos < config.INITIAL_SLOW_STEPS else config.MAX_LINEAR_SPEED_M_S
        fraccion_avance = (float(accion[0]) + 1.0) / 2.0
        velocidad_lineal = config.MIN_LINEAR_SPEED_M_S + fraccion_avance * (velocidad_maxima - config.MIN_LINEAR_SPEED_M_S)
        self.velocidad_lineal_anterior = velocidad_lineal

        # Velocidad angular: se limita el cambio entre pasos (rampa) para que el giro sea suave.
        giro_solicitado = float(accion[1]) * config.MAX_ANGULAR_SPEED_RAD_S
        cambio_giro = np.clip(giro_solicitado - self.giro_anterior, -config.MAX_ANGULAR_CHANGE_RAD_S, config.MAX_ANGULAR_CHANGE_RAD_S)
        giro_previo = self.giro_anterior
        self.giro_anterior += float(cambio_giro)

        # Se publica la orden y se deja actuar ACTION_SETTLE_S procesando mensajes.
        self._publicar_velocidad(velocidad_lineal, self.giro_anterior)
        siguiente_publicacion = time.monotonic() + config.CONTROL_PERIOD_S
        fin_espera = time.monotonic() + config.ACTION_SETTLE_S

        while time.monotonic() < fin_espera:
            rclpy.spin_once(self.nodo, timeout_sec=0.01)

        # Se descartan mensajes viejos y se espera una lectura nueva de cámara e IR,
        # republicando la orden cada CONTROL_PERIOD_S para que el robot no se pare.
        self._vaciar_cola()
        secuencia = self._n_blobs
        secuencia_ir = self._n_ir
        plazo = time.monotonic() + config.SENSOR_TIMEOUT_S

        while self._n_blobs <= secuencia or self._n_ir <= secuencia_ir:
            ahora = time.monotonic()
            if ahora >= plazo:
                self._detener()
                raise RuntimeError('Timeout de sensores.')
            if ahora >= siguiente_publicacion:
                self._publicar_velocidad(velocidad_lineal, self.giro_anterior)
                siguiente_publicacion = ahora + config.CONTROL_PERIOD_S
            rclpy.spin_once(self.nodo, timeout_sec=0.01)

        self.pasos += 1
        obs = self._observacion()
        # Detecta si el guía ha empezado a moverse (más de ARRANQUE_UMBRAL_GUIA_M desde su pose inicial).
        if not self.guia_arrancado and self._guia_ini and self._pose_guia:
            desplazamiento = np.hypot(
                self._pose_guia[0] - self._guia_ini[0],
                self._pose_guia[1] - self._guia_ini[1],
            ) / 1000.0
            self.guia_arrancado = desplazamiento > config.ARRANQUE_UMBRAL_GUIA_M
        proximidad_frontal = self._proximidad_frontal()

        # ---- Condiciones de terminación ----
        contacto_aproximado = (proximidad_frontal >= config.FRONT_IR_CONTACT_THRESHOLD)
        separacion_m, separacion_sim = self._separacion(float(obs[3]))
        demasiado_cerca = separacion_m < config.MIN_SEPARATION_M
        separacion_demasiado_grande = bool(separacion_sim and separacion_m > config.LOST_GUIDE_SEPARATION_M)
        blob_perdido_demasiado_tiempo = (self.inicio_perdida_blob is not None and time.monotonic() - self.inicio_perdida_blob >= config.LOST_BLOB_SECONDS)
        # La separación excesiva se conserva como diagnóstico, pero no termina
        # ni reinicia el episodio.
        fallo = (contacto_aproximado or demasiado_cerca or blob_perdido_demasiado_tiempo)

        # Motivo de fin, por orden de prioridad (se registra en el CSV de episodios).
        motivo_fin = None
        if contacto_aproximado: motivo_fin = 'proximidad frontal (choque)'
        elif demasiado_cerca: motivo_fin = 'blob demasiado grande'
        # elif separacion_demasiado_grande: motivo_fin = 'separación real > 0.9 m'
        elif blob_perdido_demasiado_tiempo: motivo_fin = 'blob perdido'
        elif self.pasos >= self.pasos_max: motivo_fin = 'duración máxima'

        # terminated: fallo; truncated: se agotó el tiempo. En ambos casos se detiene el robot.
        terminated = fallo
        truncated = self.pasos >= self.pasos_max
        if terminated or truncated:
            self._detener()

        # Componentes de calidad para la función externa.
        # g_centro: 1 si la pelota está centrada, cae con |err_x|.
        # g_separacion: campana gaussiana alrededor de la separación objetivo.
        g_centro = np.exp(-(float(obs[1]) / config.CENTER_SIGMA) ** 2)
        g_separacion = np.exp(-0.5 * (
            (separacion_m - config.TARGET_SEPARATION_M) / config.SEPARATION_SIGMA_M
        ) ** 2)
        r_separacion = g_separacion * (0.7 + 0.3 * g_centro)
        # Penalizaciones por girar mucho y por cambiar bruscamente de giro.
        r_suavidad = -(
            config.ANGULAR_PENALTY_WEIGHT
            * abs(float(self.giro_anterior))
            / config.MAX_ANGULAR_SPEED_RAD_S
        )
        delta_angular = self.giro_anterior - giro_previo
        r_cambio_giro = -(
            config.ANGULAR_CHANGE_PENALTY_WEIGHT
            * abs(delta_angular)
            / config.MAX_ANGULAR_CHANGE_RAD_S
        )

        # Todo lo que p2_entrenar.calcular_calidad necesita para puntuar el episodio.
        info = {
            'visible': bool(obs[0]),
            'guia_arrancado': self.guia_arrancado,
            'demasiado_cerca': demasiado_cerca,
            'separacion_demasiado_grande': separacion_demasiado_grande,
            'contacto_aproximado': contacto_aproximado,
            'proximidad_frontal': proximidad_frontal,
            'separacion_m': float(separacion_m),
            'separacion_sim': separacion_sim,
            'motivo_fin': motivo_fin,
            'error_x': float(obs[1]),
            'tamano_norm': float(obs[3]),
            'g_separacion': float(g_separacion),
            'r_separacion': float(r_separacion),
            'r_suavidad': float(r_suavidad),
            'r_cambio_giro': float(r_cambio_giro),
            'velocidad_angular': float(self.giro_anterior),
            'delta_angular': float(delta_angular),
            'delta_tamano': float(obs[6]),
            'delta_error_x': float(obs[7]),
        }

        # Ignoramos la recompensa numérica de SB3 devolviendo 0.0
        return obs, 0.0, terminated, truncated, info

    def close(self):
        """Detiene el robot (si es posible) y apaga el nodo y rclpy."""
        try:
            self._detener()
        except Exception:
            pass
        self.nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
