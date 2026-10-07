#!/usr/bin/env python3
"""Entorno Gymnasium mínimo para seguir la pelota verde en Follow City."""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from rclpy.node import Node
from robobo_ros2_interfaces.action import MoveTilt
from robobo_ros2_interfaces.srv import (
    ResetSimulation,
    SetActiveColorBlobs,
)
from std_msgs.msg import Int32MultiArray

try:
    from robobo_ros2_interfaces.msg import BlobArray
except ImportError:
    BlobArray = None

import follow_city_config as config


NS_BASE = '/robobo/robot_0/base'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'
NS_SIM = '/robobo/robot_0/sim'


class FollowCityEnv(gym.Env):
    """El seguidor controla avance y giro; el guía lo mueve el simulador.

    La acción es [avance, giro], normalizada cada componente en [-1, 1].

    Observación: [visible, error_x, error_y, tamaño, último_error_x,
    tiempo_perdido]. El tiempo perdido se normaliza con el límite de
    pérdida del episodio; si no hay detección actual, error_x, error_y
    y tamaño son cero.
    """

    metadata = {'render_modes': []}

    def __init__(self, verbose=False, usar_demostracion_inicial=True):
        super().__init__()
        if BlobArray is None:
            raise ImportError('No se encuentra robobo_ros2_interfaces/BlobArray.')
        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('follow_city_rl')
        self.verbose = verbose
        self.usar_demostracion_inicial = usar_demostracion_inicial
        self.pub_vel = self.nodo.create_publisher(
            Twist, NS_BASE + '/cmd_vel', 10)
        self.cli_reset = self.nodo.create_client(
            ResetSimulation, NS_SIM + '/reset_simulation')
        if not self.cli_reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'No responde {}/reset_simulation. Lanza el puente con '
                "los módulos ['sim', 'blob'].".format(NS_SIM))
        self.accion_tilt = ActionClient(
            self.nodo, MoveTilt, NS_BASE + '/move_tilt')
        if not self.accion_tilt.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(
                'No responde la acción {}/move_tilt.'.format(NS_BASE))
        self._blobs = []
        self._n_blobs = 0
        self.nodo.create_subscription(
            BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)
        self._ir = None
        self._n_ir = 0
        self.nodo.create_subscription(
            Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

        self.cli_colores = self.nodo.create_client(
            SetActiveColorBlobs,
            NS_SMARTPHONE + '/set_active_color_blobs')
        if not self.cli_colores.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(
                'No responde {}/set_active_color_blobs. Comprueba el módulo blob.'
                .format(NS_SMARTPHONE))
        solicitud = SetActiveColorBlobs.Request()
        solicitud.red = False
        solicitud.green = True
        solicitud.blue = False
        solicitud.custom = False
        futuro = self.cli_colores.call_async(solicitud)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        respuesta = futuro.result()
        if respuesta is None or not respuesta.success:
            raise RuntimeError('No se pudo activar la detección de blobs verdes.')

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, -1.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.pasos_max = max(1, int(
            config.EPISODE_SECONDS / config.BLOB_PERIOD_S))
        self.pasos = 0
        self.inicio_perdida_blob = None
        self.ultima_posicion_blob_x = None
        self.instante_ultima_deteccion_blob = None
        self.giro_anterior = 0.0
        self.error_x_anterior = None
        self.error_tamano_anterior = None

    def _cb_blobs(self, msg):
        self._blobs = list(msg.blobs)
        self._n_blobs += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_ir += 1

    def _blob_verde(self):
        verdes = [blob for blob in self._blobs
                  if str(blob.color).lower() == 'green'
                  and float(blob.size) > 0.0
                  and int(blob.frame_timestamp) >= 0]
        return max(verdes, key=lambda blob: blob.size) if verdes else None

    def _esperar_mensaje(self, despues_de, timeout):
        limite = time.monotonic() + timeout
        while self._n_blobs <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(
                    'No llegan mensajes de {}/color_blobs.'
                    .format(NS_SMARTPHONE))
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _esperar_ir(self, despues_de, timeout):
        limite = time.monotonic() + timeout
        while self._n_ir <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(
                    'No llegan lecturas del topic {}/ir.'
                    .format(NS_BASE))
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _proximidad_frontal(self):
        if self._ir is None:
            raise RuntimeError('Todavía no se ha recibido ninguna lectura IR.')
        if len(self._ir) <= max(config.FRONT_IR_INDICES):
            raise RuntimeError(
                'El topic {}/ir publicó {} sensores; se necesitan al menos {}.'
                .format(NS_BASE, len(self._ir),
                        max(config.FRONT_IR_INDICES) + 1))
        return max(
            min(max(int(self._ir[indice]), 0) / config.IR_SATURATION, 1.0)
            for indice in config.FRONT_IR_INDICES)

    def calibracion(self):
        """Avanza ROS una vez y devuelve las lecturas útiles para calibrar."""
        rclpy.spin_once(self.nodo, timeout_sec=0.1)
        ir_frontal = None
        proximidad = None
        if self._ir is not None:
            ir_frontal = [
                int(self._ir[indice])
                for indice in config.FRONT_IR_INDICES
            ]
            proximidad = self._proximidad_frontal()
        obs = self._observacion()
        return {
            'ir_frontal': ir_frontal,
            'proximidad_frontal': proximidad,
            'blob_visible': bool(obs[0]),
            'error_x': float(obs[1]),
            'error_y': float(obs[2]),
            'tamano_norm': float(obs[3]),
        }

    def _esperar_blob_verde(self, timeout=None):
        if timeout is None:
            timeout = config.SENSOR_TIMEOUT_S
        limite = time.monotonic() + timeout
        while self._blob_verde() is None:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(
                    'No se ve la pelota verde. Revisa el TILT y Follow City.')
            secuencia = self._n_blobs
            self._esperar_mensaje(secuencia, restante)
        self._observacion()

    def _observacion(self):
        blob = self._blob_verde()
        if blob is None:
            ahora = time.monotonic()
            if (
                    self.inicio_perdida_blob is None
                    and self.instante_ultima_deteccion_blob is not None):
                self.inicio_perdida_blob = (
                    self.instante_ultima_deteccion_blob)
            tiempo_perdido = 0.0
            if self.inicio_perdida_blob is not None:
                tiempo_perdido = np.clip(
                    (ahora - self.inicio_perdida_blob)
                    / config.LOST_BLOB_SECONDS,
                    0.0, 1.0)
            ultima_posicion_x = (
                self.ultima_posicion_blob_x
                if self.ultima_posicion_blob_x is not None else 0.0)
            return np.array(
                [0.0, 0.0, 0.0, 0.0, ultima_posicion_x, tiempo_perdido],
                dtype=np.float32)
        error_x = np.clip(
            (float(blob.x) - config.IMAGE_WIDTH_PX / 2) /
            (config.IMAGE_WIDTH_PX / 2), -1.0, 1.0)
        error_y = np.clip(
            (float(blob.y) - config.IMAGE_HEIGHT_PX / 2) /
            (config.IMAGE_HEIGHT_PX / 2), -1.0, 1.0)
        tamano = np.clip(float(blob.size) / config.BLOB_SIZE_MAX, 0.0, 1.0)
        self.ultima_posicion_blob_x = float(error_x)
        self.instante_ultima_deteccion_blob = time.monotonic()
        self.inicio_perdida_blob = None
        return np.array(
            [1.0, error_x, error_y, tamano, error_x, 0.0],
            dtype=np.float32)

    def _potencial_visual(self, obs):
        error_tamano = abs(
            float(obs[3]) - config.TARGET_BLOB_SIZE_NORM)
        return (0.7 * abs(float(obs[1]))
                + 0.3 * error_tamano)

    def _publicar_velocidad(self, lineal, angular):
        mensaje = Twist()
        mensaje.linear.x = float(lineal)
        mensaje.angular.z = float(angular)
        self.pub_vel.publish(mensaje)

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _reiniciar_simulacion(self):
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(
            self.nodo, futuro, timeout_sec=config.SENSOR_TIMEOUT_S)
        respuesta = futuro.result()
        if respuesta is None or not respuesta.success:
            raise RuntimeError('El reinicio de la simulación ha fallado.')
        time.sleep(0.5)

    def _poner_tilt_inicial(self):
        objetivo = MoveTilt.Goal()
        objetivo.angle = float(config.TILT_ANGLE_DEG)
        objetivo.speed = float(config.TILT_SPEED_DEG_S)
        futuro = self.accion_tilt.send_goal_async(objetivo)
        rclpy.spin_until_future_complete(
            self.nodo, futuro, timeout_sec=5.0)
        if not futuro.done():
            raise RuntimeError(
                'Se agotó el tiempo enviando la orden del TILT.')
        gestor = futuro.result()
        if gestor is None or not gestor.accepted:
            raise RuntimeError('El Robobo rechazó la orden del TILT.')

    def preparar_calibracion(self):
        """Reinicia escena y TILT, sin exigir que el blob ya sea visible."""
        self._detener()
        self._reiniciar_simulacion()
        self._poner_tilt_inicial()
        self._blobs = []
        self.inicio_perdida_blob = None
        self.ultima_posicion_blob_x = None
        self.instante_ultima_deteccion_blob = None
        self._esperar_ir(self._n_ir, config.SENSOR_TIMEOUT_S)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for intento in range(config.RESET_ATTEMPTS):
            self._detener()
            self._reiniciar_simulacion()
            self._poner_tilt_inicial()
            self.pasos = 0
            self.inicio_perdida_blob = None
            self.ultima_posicion_blob_x = None
            self.instante_ultima_deteccion_blob = None
            self.giro_anterior = 0.0
            self._blobs = []
            secuencia = self._n_blobs
            secuencia_ir = self._n_ir
            try:
                self._esperar_mensaje(
                    secuencia, config.RESET_SENSOR_TIMEOUT_S)
                self._esperar_ir(secuencia_ir, config.SENSOR_TIMEOUT_S)
                self._esperar_blob_verde(config.RESET_SENSOR_TIMEOUT_S)
                time.sleep(config.GUIDE_START_DELAY_S)
            except RuntimeError as error:
                error_topic_blob = (
                    'No llegan mensajes de {}/color_blobs.'
                    .format(NS_SMARTPHONE))
                error_avance_blob = (
                    'No llegan lecturas nuevas de {}/color_blobs'
                    .format(NS_SMARTPHONE))
                if (
                        error_topic_blob not in str(error)
                        and error_avance_blob not in str(error)):
                    raise
                if intento + 1 == config.RESET_ATTEMPTS:
                    raise
                continue

            obs = self._observacion()
            self.error_x_anterior = abs(float(obs[1]))
            self.error_tamano_anterior = abs(
                float(obs[3]) - config.TARGET_BLOB_SIZE_NORM)
            return obs, {}

        raise RuntimeError(
            'No se pudo reiniciar el episodio tras perder el topic de blobs.')

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        accion_demostracion = (
            self.usar_demostracion_inicial
            and self.pasos < config.AUTO_FORWARD_STEPS)
        if accion_demostracion:
            accion = np.array([1.0, 0.0], dtype=np.float32)
        velocidad_maxima = (
            config.INITIAL_LINEAR_SPEED_M_S
            if self.pasos < config.INITIAL_SLOW_STEPS
            else config.MAX_LINEAR_SPEED_M_S)
        fraccion_avance = (float(accion[0]) + 1.0) / 2.0
        velocidad_lineal = (
            config.MIN_LINEAR_SPEED_M_S
            + fraccion_avance
            * (velocidad_maxima - config.MIN_LINEAR_SPEED_M_S))
        giro_solicitado = float(accion[1]) * config.MAX_ANGULAR_SPEED_RAD_S
        cambio_giro = np.clip(
            giro_solicitado - self.giro_anterior,
            -config.MAX_ANGULAR_CHANGE_RAD_S,
            config.MAX_ANGULAR_CHANGE_RAD_S)
        self.giro_anterior += float(cambio_giro)

        secuencia = self._n_blobs
        secuencia_ir = self._n_ir
        plazo = time.monotonic() + config.SENSOR_TIMEOUT_S
        siguiente_publicacion = time.monotonic()
        while self._n_blobs <= secuencia or self._n_ir <= secuencia_ir:
            ahora = time.monotonic()
            if ahora >= plazo:
                self._detener()
                if self._n_blobs <= secuencia:
                    obs = self._observacion()
                    self.error_x_anterior = None
                    info = {
                        'visible': bool(obs[0]),
                        'demasiado_cerca': False,
                        'contacto_aproximado': False,
                        'proximidad_frontal': self._proximidad_frontal(),
                        'motivo_fin': 'sin mensajes del topic de blobs',
                        'error_x': float(obs[1]),
                        'error_y': float(obs[2]),
                        'tamano_norm': float(obs[3]),
                        'velocidad_lineal': velocidad_lineal,
                        'velocidad_angular': self.giro_anterior,
                        '_accion_ejecutada': (
                            accion.tolist() if accion_demostracion else None),
                    }
                    return obs, -1.0, True, False, info
                faltan = []
                if self._n_ir <= secuencia_ir:
                    faltan.append(NS_BASE + '/ir')
                raise RuntimeError(
                    'No llegan lecturas nuevas de {}; se detuvo el seguidor.'
                    .format(', '.join(faltan)))
            if ahora >= siguiente_publicacion:
                self._publicar_velocidad(
                    velocidad_lineal, self.giro_anterior)
                siguiente_publicacion = ahora + config.CONTROL_PERIOD_S
            rclpy.spin_once(self.nodo, timeout_sec=0.01)

        estaba_perdido = self.inicio_perdida_blob is not None
        self.pasos += 1
        obs = self._observacion()
        proximidad_frontal = self._proximidad_frontal()
        contacto_aproximado = (
            proximidad_frontal >= config.FRONT_IR_CONTACT_THRESHOLD)
        if obs[0]:
            blob_recuperado = estaba_perdido
            recompensa = -self._potencial_visual(obs)
            if blob_recuperado:
                recompensa += config.BLOB_RECOVERY_REWARD
            if self.error_x_anterior is not None:
                mejora_horizontal = (
                    self.error_x_anterior - abs(float(obs[1])))
                recompensa += (
                    config.HORIZONTAL_PROGRESS_WEIGHT * mejora_horizontal)
            self.error_x_anterior = abs(float(obs[1]))
            error_tamano = abs(
                float(obs[3]) - config.TARGET_BLOB_SIZE_NORM)
            if self.error_tamano_anterior is not None:
                mejora_tamano = self.error_tamano_anterior - error_tamano
                recompensa += config.SIZE_PROGRESS_WEIGHT * mejora_tamano
            self.error_tamano_anterior = error_tamano
            blob_centrado = (
                abs(float(obs[1])) <= config.STABILITY_ERROR_X_TOLERANCE)
            distancia_correcta = (
                abs(float(obs[3]) - config.TARGET_BLOB_SIZE_NORM)
                <= config.STABILITY_BLOB_SIZE_TOLERANCE)
            if blob_centrado and distancia_correcta:
                recompensa += config.STABILITY_REWARD
        else:
            recompensa = config.LOST_BLOB_REWARD
            self.error_x_anterior = None
            self.error_tamano_anterior = None
        if contacto_aproximado:
            recompensa -= config.CONTACT_PENALTY

        demasiado_cerca = bool(
            obs[0] and obs[3] >= config.TOO_CLOSE_BLOB_SIZE_NORM)
        blob_perdido_demasiado_tiempo = (
            self.inicio_perdida_blob is not None
            and time.monotonic() - self.inicio_perdida_blob
            >= config.LOST_BLOB_SECONDS)
        motivo_fin = None
        if contacto_aproximado:
            motivo_fin = 'proximidad frontal (choque aproximado)'
        elif demasiado_cerca:
            motivo_fin = 'blob demasiado grande'
        elif blob_perdido_demasiado_tiempo:
            motivo_fin = 'blob perdido'
        elif self.pasos >= self.pasos_max:
            motivo_fin = 'duración máxima'
        terminated = (
            contacto_aproximado
            or demasiado_cerca
            or blob_perdido_demasiado_tiempo)
        truncated = self.pasos >= self.pasos_max
        if terminated or truncated:
            self._detener()
        info = {
            'visible': bool(obs[0]),
            'demasiado_cerca': demasiado_cerca,
            'contacto_aproximado': contacto_aproximado,
            'proximidad_frontal': proximidad_frontal,
            'motivo_fin': motivo_fin,
            'error_x': float(obs[1]),
            'error_y': float(obs[2]),
            'tamano_norm': float(obs[3]),
            'velocidad_lineal': velocidad_lineal,
            'velocidad_angular': self.giro_anterior,
            '_accion_ejecutada': (
                accion.tolist() if accion_demostracion else None),
        }
        if self.verbose:
            print('paso={} visible={} ex={:+.2f} ey={:+.2f} '
              'v={:.2f} w={:+.2f} r={:+.2f} contacto_aproximado={}'
              .format(self.pasos, info['visible'], info['error_x'],
                          info['error_y'], velocidad_lineal,
                          self.giro_anterior, recompensa,
                          contacto_aproximado))
        return obs, float(recompensa), terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        self.nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()