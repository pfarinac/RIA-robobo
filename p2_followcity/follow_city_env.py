#!/usr/bin/env python3
"""Entorno Gymnasium para seguir al vehículo guía en Follow City (Adaptado para NEAT)."""

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
    metadata = {'render_modes': []}

    def __init__(self, verbose=False, episode_seconds=None):
        super().__init__()
        if BlobArray is None:
            raise ImportError('No se encuentra robobo_ros2_interfaces/BlobArray.')

        if not rclpy.ok():
            rclpy.init()

        self.nodo = Node('follow_city_rl')
        self.verbose = verbose

        self.pub_vel = self.nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)
        self.cli_reset = self.nodo.create_client(ResetSimulation, NS_SIM + '/reset_simulation')
        if not self.cli_reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'No responde {NS_SIM}/reset_simulation.')

        self.accion_tilt = ActionClient(self.nodo, MoveTilt, NS_BASE + '/move_tilt')
        if not self.accion_tilt.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f'No responde la acción {NS_BASE}/move_tilt.')

        self._blobs = []
        self._n_blobs = 0
        self.nodo.create_subscription(BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)

        self._ir = None
        self._n_ir = 0
        self.nodo.create_subscription(Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)

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

        # Espacios
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if episode_seconds is None:
            episode_seconds = config.EPISODE_SECONDS
        self.pasos_max = max(1, int(episode_seconds / config.REAL_STEP_S))

        # Estado interno
        self.pasos = 0
        self.inicio_perdida_blob = None
        self.ultima_posicion_blob_x = None
        self.instante_ultima_deteccion_blob = None
        self.giro_anterior = 0.0
        self.velocidad_lineal_anterior = 0.0
        self.frame_ts = -1

    def _cb_blobs(self, msg):
        self._blobs = list(msg.blobs)
        self._n_blobs += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_ir += 1

    def _blob_verde(self):
        verdes = [blob for blob in self._blobs if str(blob.color).lower() == 'green'
                  and float(blob.size) > 0.0 and int(blob.frame_timestamp) >= 0]
        return max(verdes, key=lambda blob: blob.size) if verdes else None

    def _esperar_mensaje(self, despues_de, timeout):
        limite = time.monotonic() + timeout
        while self._n_blobs <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(f'No llegan mensajes de {NS_SMARTPHONE}/color_blobs.')
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _esperar_ir(self, despues_de, timeout):
        limite = time.monotonic() + timeout
        while self._n_ir <= despues_de:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError(f'No llegan lecturas de {NS_BASE}/ir.')
            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _vaciar_cola(self):
        for _ in range(100):
            antes = (self._n_blobs, self._n_ir)
            rclpy.spin_once(self.nodo, timeout_sec=0.0)
            if (self._n_blobs, self._n_ir) == antes:
                return

    def _proximidad_frontal(self):
        if self._ir is None:
            raise RuntimeError('Sin lectura IR.')
        return max(min(max(int(self._ir[i]), 0) / config.IR_SATURATION, 1.0) for i in config.FRONT_IR_INDICES)

    def _esperar_blob_verde(self, timeout=None):
        if timeout is None: timeout = config.SENSOR_TIMEOUT_S
        limite = time.monotonic() + timeout
        while self._blob_verde() is None:
            restante = limite - time.monotonic()
            if restante <= 0:
                raise RuntimeError('No se ve la pelota verde.')
            self._esperar_mensaje(self._n_blobs, restante)
        self._observacion()

    def _observacion(self):
        w_norm = np.clip(self.giro_anterior / config.MAX_ANGULAR_SPEED_RAD_S, -1.0, 1.0)
        v_norm = 0.0
        rango_v = config.MAX_LINEAR_SPEED_M_S - config.MIN_LINEAR_SPEED_M_S
        if rango_v > 0:
            v_norm = np.clip((self.velocidad_lineal_anterior - config.MIN_LINEAR_SPEED_M_S) / rango_v, 0.0, 1.0)

        blob = self._blob_verde()
        if blob is None:
            ahora = time.monotonic()
            if self.inicio_perdida_blob is None and self.instante_ultima_deteccion_blob is not None:
                self.inicio_perdida_blob = self.instante_ultima_deteccion_blob
            tiempo_perdido = 0.0
            if self.inicio_perdida_blob is not None:
                tiempo_perdido = np.clip((ahora - self.inicio_perdida_blob) / config.LOST_BLOB_SECONDS, 0.0, 1.0)
            ultima_posicion_x = self.ultima_posicion_blob_x if self.ultima_posicion_blob_x is not None else 0.0
            return np.array([0.0, 0.0, 0.0, 0.0, ultima_posicion_x, tiempo_perdido, v_norm, w_norm], dtype=np.float32)

        error_x = np.clip((float(blob.x) - config.BLOB_COORD_MAX / 2) / (config.BLOB_COORD_MAX / 2), -1.0, 1.0)
        error_y = np.clip((float(blob.y) - config.BLOB_COORD_MAX / 2) / (config.BLOB_COORD_MAX / 2), -1.0, 1.0)
        tamano = np.clip(float(blob.size) / config.BLOB_SIZE_MAX, 0.0, 1.0)

        self.frame_ts = int(blob.frame_timestamp)
        self.ultima_posicion_blob_x = float(error_x)
        self.instante_ultima_deteccion_blob = time.monotonic()
        self.inicio_perdida_blob = None

        return np.array([1.0, error_x, error_y, tamano, error_x, 0.0, v_norm, w_norm], dtype=np.float32)

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
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=config.SENSOR_TIMEOUT_S)
        time.sleep(0.5)

    def _poner_tilt_inicial(self):
        objetivo = MoveTilt.Goal()
        objetivo.angle = float(config.TILT_ANGLE_DEG)
        objetivo.speed = float(config.TILT_SPEED_DEG_S)
        futuro = self.accion_tilt.send_goal_async(objetivo)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)

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
            self.velocidad_lineal_anterior = 0.0
            self._blobs = []

            secuencia = self._n_blobs
            secuencia_ir = self._n_ir

            try:
                self._esperar_mensaje(secuencia, config.RESET_SENSOR_TIMEOUT_S)
                self._esperar_ir(secuencia_ir, config.SENSOR_TIMEOUT_S)
                self._esperar_blob_verde(config.RESET_SENSOR_TIMEOUT_S)
                time.sleep(config.GUIDE_START_DELAY_S)
                self._vaciar_cola()
                self._esperar_mensaje(self._n_blobs, config.RESET_SENSOR_TIMEOUT_S)
                self._esperar_ir(self._n_ir, config.SENSOR_TIMEOUT_S)
                self.instante_ultima_deteccion_blob = time.monotonic()
            except RuntimeError as error:
                if intento + 1 == config.RESET_ATTEMPTS:
                    raise
                continue
            return self._observacion(), {}
        raise RuntimeError('Fallo total al reiniciar.')

    def step(self, accion):
        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)
        
        velocidad_maxima = config.INITIAL_LINEAR_SPEED_M_S if self.pasos < config.INITIAL_SLOW_STEPS else config.MAX_LINEAR_SPEED_M_S
        fraccion_avance = (float(accion[0]) + 1.0) / 2.0
        velocidad_lineal = config.MIN_LINEAR_SPEED_M_S + fraccion_avance * (velocidad_maxima - config.MIN_LINEAR_SPEED_M_S)
        self.velocidad_lineal_anterior = velocidad_lineal

        giro_solicitado = float(accion[1]) * config.MAX_ANGULAR_SPEED_RAD_S
        cambio_giro = np.clip(giro_solicitado - self.giro_anterior, -config.MAX_ANGULAR_CHANGE_RAD_S, config.MAX_ANGULAR_CHANGE_RAD_S)
        giro_previo = self.giro_anterior
        self.giro_anterior += float(cambio_giro)

        self._publicar_velocidad(velocidad_lineal, self.giro_anterior)
        siguiente_publicacion = time.monotonic() + config.CONTROL_PERIOD_S
        fin_espera = time.monotonic() + config.ACTION_SETTLE_S

        while time.monotonic() < fin_espera:
            rclpy.spin_once(self.nodo, timeout_sec=0.01)

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
        proximidad_frontal = self._proximidad_frontal()

        contacto_aproximado = (proximidad_frontal >= config.FRONT_IR_CONTACT_THRESHOLD)
        demasiado_cerca = bool(obs[0] and obs[3] >= config.TOO_CLOSE_BLOB_SIZE_NORM)
        blob_perdido_demasiado_tiempo = (self.inicio_perdida_blob is not None and time.monotonic() - self.inicio_perdida_blob >= config.LOST_BLOB_SECONDS)
        fallo = (contacto_aproximado or demasiado_cerca or blob_perdido_demasiado_tiempo)

        motivo_fin = None
        if contacto_aproximado: motivo_fin = 'proximidad frontal (choque)'
        elif demasiado_cerca: motivo_fin = 'blob demasiado grande'
        elif blob_perdido_demasiado_tiempo: motivo_fin = 'blob perdido'
        elif self.pasos >= self.pasos_max: motivo_fin = 'duración máxima'

        terminated = fallo
        truncated = self.pasos >= self.pasos_max
        if terminated or truncated:
            self._detener()

        # Componentes de calidad para la función externa
        r_centro = config.CENTER_WEIGHT * np.exp(-(float(obs[1]) / config.CENTER_SIGMA) ** 2) if obs[0] else 0.0
        r_tamano = config.SIZE_WEIGHT * np.exp(-((float(obs[3]) - config.TARGET_BLOB_SIZE_NORM) / config.SIZE_SIGMA) ** 2) if obs[0] else 0.0
        r_suavidad = -(config.SMOOTHNESS_WEIGHT * abs(float(cambio_giro)) / config.MAX_ANGULAR_CHANGE_RAD_S)

        info = {
            'visible': bool(obs[0]),
            'demasiado_cerca': demasiado_cerca,
            'contacto_aproximado': contacto_aproximado,
            'proximidad_frontal': proximidad_frontal,
            'motivo_fin': motivo_fin,
            'error_x': float(obs[1]),
            'tamano_norm': float(obs[3]),
            'r_centro': float(r_centro),
            'r_tamano': float(r_tamano),
            'r_suavidad': float(r_suavidad),
            'velocidad_angular': float(self.giro_anterior)
        }
        
        # Ignoramos la recompensa numérica de SB3 devolviendo 0.0 
        return obs, 0.0, terminated, truncated, info

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        self.nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()