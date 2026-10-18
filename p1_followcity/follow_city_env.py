#!/usr/bin/env python3

"""Entorno Gymnasium para seguir al vehículo guía en Follow City.
 
El seguidor es el robot conectado al puente robobo_ros2; el guía lo

mueve el simulador y no reacciona al seguidor. Toda la percepción y la

actuación pasan por ROS 2:
 
  - smartphone/color_blobs   detección de la pelota verde del guía.

  - base/ir                  infrarrojos, para detectar choques.

  - base/cmd_vel             actuación en velocidad lineal y angular.

  - base/move_tilt           inclinación de la cámara al empezar.

  - sim/reset_simulation     reinicio de la escena (solo simulador).
 
La información privilegiada del simulador solo se usa para reiniciar

los episodios; nunca forma parte de la observación ni de la recompensa,

para que la política pueda ejecutarse en el robot real sin reentrenar.

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
 
 
# Espacios de nombres del puente para el robot 0 (el seguidor).

NS_BASE = '/robobo/robot_0/base'

NS_SMARTPHONE = '/robobo/robot_0/smartphone'

NS_SIM = '/robobo/robot_0/sim'
 
 
class FollowCityEnv(gym.Env):

    """El seguidor controla avance y giro; el guía lo mueve el simulador.
 
    Acción (Box, dimensión 2, cada componente en [-1, 1]):

        [0] avance, reescalado a [MIN_LINEAR_SPEED, velocidad máxima].

        [1] giro, reescalado a [-MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED]

            y limitado a un cambio de MAX_ANGULAR_CHANGE por paso.
 
    Observación (Box, dimensión 8):

        [0] visible: 1 si se detecta la pelota verde, 0 si no.

        [1] error_x: posición horizontal del blob normalizada.

        [2] error_y: posición vertical del blob normalizada.

        [3] tamaño del blob normalizado en [0, 1].

        [4] última posición horizontal conocida del blob.

        [5] tiempo sin ver el blob, normalizado con LOST_BLOB_SECONDS.

        [6] velocidad lineal del paso anterior, normalizada en [0, 1].

        [7] velocidad angular del paso anterior, normalizada en [-1, 1].

    Si no hay detección, las componentes 1 a 3 valen cero.

    """
 
    metadata = {'render_modes': []}
 
    def __init__(

            self, verbose=False, usar_demostracion_inicial=True,

            episode_seconds=None):

        """Crea el nodo ROS 2, las suscripciones, los clientes y los espacios.
 
        usar_demostracion_inicial: si es True, los primeros

            AUTO_FORWARD_STEPS pasos de cada episodio ejecutan una acción

            fija de avance recto (solo durante el entrenamiento de SAC).

        episode_seconds: duración nominal del episodio; por defecto,

            EPISODE_SECONDS. Se convierte en pasos con BLOB_PERIOD_S.

        """

        super().__init__()

        if BlobArray is None:

            raise ImportError('No se encuentra robobo_ros2_interfaces/BlobArray.')

        # rclpy solo puede inicializarse una vez por proceso.

        if not rclpy.ok():

            rclpy.init()
 
        self.nodo = Node('follow_city_rl')

        self.verbose = verbose

        self.usar_demostracion_inicial = usar_demostracion_inicial

        # Publicador de velocidades: se crea una sola vez, aquí, porque el

        # descubrimiento de ROS 2 tarda y se perderían las primeras órdenes.

        self.pub_vel = self.nodo.create_publisher(

            Twist, NS_BASE + '/cmd_vel', 10)

        # Servicio de reinicio de la escena (canal privilegiado del simulador).

        self.cli_reset = self.nodo.create_client(

            ResetSimulation, NS_SIM + '/reset_simulation')

        if not self.cli_reset.wait_for_service(timeout_sec=5.0):

            raise RuntimeError(

                'No responde {}/reset_simulation. Lanza el puente con '

                "los módulos ['sim', 'blob'].".format(NS_SIM))

        # Acción para inclinar la cámara al inicio de cada episodio.

        self.accion_tilt = ActionClient(

            self.nodo, MoveTilt, NS_BASE + '/move_tilt')

        if not self.accion_tilt.wait_for_server(timeout_sec=5.0):

            raise RuntimeError(

                'No responde la acción {}/move_tilt.'.format(NS_BASE))

        # Última lista de blobs recibida y contador de mensajes, que

        # permite esperar a una lectura posterior a un instante dado.

        self._blobs = []

        self._n_blobs = 0

        self.nodo.create_subscription(

            BlobArray, NS_SMARTPHONE + '/color_blobs', self._cb_blobs, 1)

        # Última lectura de infrarrojos y contador de mensajes.

        self._ir = None

        self._n_ir = 0

        self.nodo.create_subscription(

            Int32MultiArray, NS_BASE + '/ir', self._cb_ir, 1)
 
        # Activa solo la detección de blobs verdes (la pelota del guía).

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
 
        # Espacio de observación de 8 componentes (véase el docstring).

        self.observation_space = spaces.Box(

            low=np.array(

                [0.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0], dtype=np.float32),

            high=np.ones(8, dtype=np.float32),

            dtype=np.float32)

        # Espacio de acción continuo: [avance, giro] en [-1, 1].

        self.action_space = spaces.Box(

            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Límite de pasos del episodio (truncamiento).

        if episode_seconds is None:

            episode_seconds = config.EPISODE_SECONDS

        self.pasos_max = max(1, int(

            episode_seconds / config.BLOB_PERIOD_S))

        # Estado interno del episodio.

        self.pasos = 0

        self.inicio_perdida_blob = None

        self.ultima_posicion_blob_x = None

        self.instante_ultima_deteccion_blob = None

        self.giro_anterior = 0.0

        self.frame_ts = -1
 
 
    # ----------------------------------------------------------------

    # Comunicación con ROS 2

    # ----------------------------------------------------------------
 
    def _cb_blobs(self, msg):

        """Guarda la última lista de blobs y cuenta los mensajes."""

        self._blobs = list(msg.blobs)

        self._n_blobs += 1
 
    def _cb_ir(self, msg):

        """Guarda la última lectura de infrarrojos y cuenta los mensajes."""

        self._ir = list(msg.data)

        self._n_ir += 1
 
    def _blob_verde(self):

        """Devuelve el blob verde válido de mayor tamaño, o None."""

        verdes = [blob for blob in self._blobs

                  if str(blob.color).lower() == 'green'

                  and float(blob.size) > 0.0

                  and int(blob.frame_timestamp) >= 0]

        return max(verdes, key=lambda blob: blob.size) if verdes else None
 
    def _esperar_mensaje(self, despues_de, timeout):

        """Bloquea hasta recibir un mensaje de blobs posterior al número

        de secuencia indicado, o lanza un error si se agota el plazo."""

        limite = time.monotonic() + timeout

        while self._n_blobs <= despues_de:

            restante = limite - time.monotonic()

            if restante <= 0:

                raise RuntimeError(

                    'No llegan mensajes de {}/color_blobs.'

                    .format(NS_SMARTPHONE))

            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))
 
    def _esperar_ir(self, despues_de, timeout):

        """Igual que _esperar_mensaje, para los infrarrojos."""

        limite = time.monotonic() + timeout

        while self._n_ir <= despues_de:

            restante = limite - time.monotonic()

            if restante <= 0:

                raise RuntimeError(

                    'No llegan lecturas del topic {}/ir.'

                    .format(NS_BASE))

            rclpy.spin_once(self.nodo, timeout_sec=min(0.05, restante))

    def _vaciar_cola(self):

        """Procesa y descarta los mensajes pendientes, que pueden ser

        anteriores a la última acción aplicada."""

        for _ in range(100):

            antes = (self._n_blobs, self._n_ir)

            rclpy.spin_once(self.nodo, timeout_sec=0.0)

            if (self._n_blobs, self._n_ir) == antes:

                return
 
    def _proximidad_frontal(self):

        """Proximidad frontal normalizada en [0, 1]: el máximo de los

        infrarrojos frontales, ya que basta con que uno vea el obstáculo."""

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

        """Bloquea hasta que la pelota verde sea visible, o lanza un error."""

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
 
    # ----------------------------------------------------------------

    # Observación

    # ----------------------------------------------------------------
 
    def _observacion(self):

        """Construye el vector de observación de 8 componentes y actualiza

        el registro de la última detección del blob."""

        # Velocidades del paso anterior, normalizadas: hacen observable la

        # inercia del propio robot.

        w_norm = np.clip(self.giro_anterior / config.MAX_ANGULAR_SPEED_RAD_S, -1.0, 1.0)

        v_norm = 0.0

        rango_v = config.MAX_LINEAR_SPEED_M_S - config.MIN_LINEAR_SPEED_M_S

        if rango_v > 0 and hasattr(self, 'velocidad_lineal_anterior'):

            v_norm = np.clip((self.velocidad_lineal_anterior - config.MIN_LINEAR_SPEED_M_S) / rango_v, 0.0, 1.0)

        blob = self._blob_verde()

        if blob is None:

            # Sin detección: se cuenta el tiempo desde la última vez que se

            # vio la pelota y se conserva su última posición horizontal,

            # para que la política sepa hacia dónde girar para recuperarla.

            ahora = time.monotonic()

            if (self.inicio_perdida_blob is None and self.instante_ultima_deteccion_blob is not None):

                self.inicio_perdida_blob = self.instante_ultima_deteccion_blob

            tiempo_perdido = 0.0

            if self.inicio_perdida_blob is not None:

                tiempo_perdido = np.clip((ahora - self.inicio_perdida_blob) / config.LOST_BLOB_SECONDS, 0.0, 1.0)

            ultima_posicion_x = self.ultima_posicion_blob_x if self.ultima_posicion_blob_x is not None else 0.0

            return np.array([0.0, 0.0, 0.0, 0.0, ultima_posicion_x, tiempo_perdido, v_norm, w_norm], dtype=np.float32)

        # Con detección: posición y tamaño del blob normalizados.

        error_x = np.clip((float(blob.x) - config.IMAGE_WIDTH_PX / 2) / (config.IMAGE_WIDTH_PX / 2), -1.0, 1.0)

        error_y = np.clip((float(blob.y) - config.IMAGE_HEIGHT_PX / 2) / (config.IMAGE_HEIGHT_PX / 2), -1.0, 1.0)

        tamano = np.clip(float(blob.size) / config.BLOB_SIZE_MAX, 0.0, 1.0)

        self.frame_ts = int(blob.frame_timestamp)

        self.ultima_posicion_blob_x = float(error_x)

        self.instante_ultima_deteccion_blob = time.monotonic()

        self.inicio_perdida_blob = None

        return np.array([1.0, error_x, error_y, tamano, error_x, 0.0, v_norm, w_norm], dtype=np.float32)
 
 
    # ----------------------------------------------------------------

    # Actuación y escena

    # ----------------------------------------------------------------
 
    def _publicar_velocidad(self, lineal, angular):

        """Publica una orden de velocidad en cmd_vel (m/s y rad/s)."""

        mensaje = Twist()

        mensaje.linear.x = float(lineal)

        mensaje.angular.z = float(angular)

        self.pub_vel.publish(mensaje)
 
    def _detener(self):

        """Detiene el seguidor; la orden se repite para asegurar que llega."""

        for _ in range(3):

            self._publicar_velocidad(0.0, 0.0)

            time.sleep(0.05)
 
    def _reiniciar_simulacion(self):

        """Reinicia la escena mediante el servicio del módulo sim."""

        futuro = self.cli_reset.call_async(ResetSimulation.Request())

        rclpy.spin_until_future_complete(

            self.nodo, futuro, timeout_sec=config.SENSOR_TIMEOUT_S)

        respuesta = futuro.result()

        if respuesta is None or not respuesta.success:

            raise RuntimeError('El reinicio de la simulación ha fallado.')

        time.sleep(0.5)
 
    def _poner_tilt_inicial(self):

        """Inclina la cámara al ángulo de configuración para ver la pelota."""

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
 
    # ----------------------------------------------------------------

    # Interfaz de Gymnasium

    # ----------------------------------------------------------------
 
    def reset(self, seed=None, options=None):

        """Reinicia la escena, coloca la cámara, espera a ver la pelota y

        deja arrancar al guía antes de devolver la primera observación.
 
        Si se pierde el topic de blobs durante el reinicio, se reintenta

        hasta RESET_ATTEMPTS veces.

        """

        super().reset(seed=seed)

        for intento in range(config.RESET_ATTEMPTS):

            self._detener()

            self._reiniciar_simulacion()

            self._poner_tilt_inicial()

            # Estado del episodio a cero.

            self.pasos = 0

            self.inicio_perdida_blob = None

            self.ultima_posicion_blob_x = None

            self.instante_ultima_deteccion_blob = None

            self.giro_anterior = 0.0

            self.instante_ultima_deteccion_blob = None

            self.giro_anterior = 0.0

            # Velocidad lineal del paso anterior, usada en la observación.

            self.velocidad_lineal_anterior = 0.0

            self._blobs = []

            self._blobs = []

            secuencia = self._n_blobs

            secuencia_ir = self._n_ir

            try:

                # Esperar lecturas nuevas de ambos sensores y la pelota.

                self._esperar_mensaje(

                    secuencia, config.RESET_SENSOR_TIMEOUT_S)

                self._esperar_ir(secuencia_ir, config.SENSOR_TIMEOUT_S)

                self._esperar_blob_verde(config.RESET_SENSOR_TIMEOUT_S)

                # El seguidor espera parado a que el guía arranque.

                time.sleep(config.GUIDE_START_DELAY_S)

                # Descartar lo acumulado durante la espera y leer de nuevo.

                self._vaciar_cola()

                self._esperar_mensaje(self._n_blobs, config.RESET_SENSOR_TIMEOUT_S)

                self._esperar_ir(self._n_ir, config.SENSOR_TIMEOUT_S)

                # Reinicia el reloj de "último blob visto": si no, un blob ausente tras

                # la espera contaría como perdido desde hace 7 s y cerraría el episodio.

                self.instante_ultima_deteccion_blob = time.monotonic()

            except RuntimeError as error:

                # Solo se reintenta si el fallo es la pérdida del topic de

                # blobs; cualquier otro error se propaga.

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
 
            return self._observacion(), {}
 
        raise RuntimeError(

            'No se pudo reiniciar el episodio tras perder el topic de blobs.')
 
    def step(self, accion):

        """Aplica la acción, espera lecturas posteriores a ella y devuelve

        la observación, la recompensa y las condiciones de fin."""

        accion = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)

        # Demostración inicial: los primeros pasos avanzan en línea recta.

        accion_demostracion = (

            self.usar_demostracion_inicial

            and self.pasos < config.AUTO_FORWARD_STEPS)

        if accion_demostracion:

            accion = np.array([1.0, 0.0], dtype=np.float32)

        # Arranque suave: velocidad máxima reducida en los primeros pasos.

        velocidad_maxima = (

            config.INITIAL_LINEAR_SPEED_M_S

            if self.pasos < config.INITIAL_SLOW_STEPS

            else config.MAX_LINEAR_SPEED_M_S)

        # Reescalado del avance de [-1, 1] a [MIN, velocidad_maxima].

        fraccion_avance = (float(accion[0]) + 1.0) / 2.0

        velocidad_lineal = (

            config.MIN_LINEAR_SPEED_M_S

            + fraccion_avance

            * (velocidad_maxima - config.MIN_LINEAR_SPEED_M_S))

        # Se guarda para incluirla en la siguiente observación.

        self.velocidad_lineal_anterior = velocidad_lineal

        # Reescalado del giro y limitación de su cambio entre pasos, que

        # impide giros bruscos aunque la política los pida.

        giro_solicitado = float(accion[1]) * config.MAX_ANGULAR_SPEED_RAD_S

        cambio_giro = np.clip(

            giro_solicitado - self.giro_anterior,

            -config.MAX_ANGULAR_CHANGE_RAD_S,

            config.MAX_ANGULAR_CHANGE_RAD_S)

        giro_previo = self.giro_anterior

        self.giro_anterior += float(cambio_giro)
 
        # 1. Aplicar la acción y dejar que surta efecto.

        self._publicar_velocidad(velocidad_lineal, self.giro_anterior)

        siguiente_publicacion = time.monotonic() + config.CONTROL_PERIOD_S

        fin_espera = time.monotonic() + config.ACTION_SETTLE_S

        while time.monotonic() < fin_espera:

            rclpy.spin_once(self.nodo, timeout_sec=0.01)
 
        # 2. Descartar todo lo publicado antes (puede ser anterior a la acción).

        self._vaciar_cola()

        secuencia = self._n_blobs

        secuencia_ir = self._n_ir
 
        # 3. Esperar un blob y un IR publicados DESPUÉS de ese instante,

        #    reenviando la orden de velocidad para que no salte el

        #    temporizador de seguridad de cmd_vel (0,5 s).

        plazo = time.monotonic() + config.SENSOR_TIMEOUT_S

        while self._n_blobs <= secuencia or self._n_ir <= secuencia_ir:

            ahora = time.monotonic()

            if ahora >= plazo:

                self._detener()

                if self._n_blobs <= secuencia:

                    raise RuntimeError(

                        'No llegan mensajes de {}/color_blobs.'

                        .format(NS_SMARTPHONE))

                raise RuntimeError(

                    'No llegan lecturas nuevas de {}/ir; '

                    'se detuvo el seguidor.'.format(NS_BASE))

                ...

            if ahora >= siguiente_publicacion:

                self._publicar_velocidad(

                    velocidad_lineal, self.giro_anterior)

                siguiente_publicacion = ahora + config.CONTROL_PERIOD_S

            rclpy.spin_once(self.nodo, timeout_sec=0.01)
 
        # 4. Observación resultante y condiciones de fallo.

        self.pasos += 1

        obs = self._observacion()

        proximidad_frontal = self._proximidad_frontal()

        # Choque inminente con un obstáculo según los infrarrojos.

        contacto_aproximado = (

            proximidad_frontal >= config.FRONT_IR_CONTACT_THRESHOLD)

        # Pelota demasiado grande: el seguidor está pegado al guía.

        demasiado_cerca = bool(

            obs[0] and obs[3] >= config.TOO_CLOSE_BLOB_SIZE_NORM)

        # Pelota sin detectar durante LOST_BLOB_SECONDS: guía perdido.

        blob_perdido_demasiado_tiempo = (

            self.inicio_perdida_blob is not None

            and time.monotonic() - self.inicio_perdida_blob
>= config.LOST_BLOB_SECONDS)

        fallo = (contacto_aproximado or demasiado_cerca

                 or blob_perdido_demasiado_tiempo)
 
        # 5. Recompensa, con cada término por separado.

        if obs[0]:

            # Pelota visible: premio por supervivencia, por centrado

            # (campana en error_x) y por tamaño cercano al objetivo.

            r_vivo = config.ALIVE_REWARD

            r_centro = config.CENTER_WEIGHT * np.exp(

                -(float(obs[1]) / config.CENTER_SIGMA) ** 2)

            r_tamano = config.SIZE_WEIGHT * np.exp(

                -((float(obs[3]) - config.TARGET_BLOB_SIZE_NORM)

                  / config.SIZE_SIGMA) ** 2)

            r_perdido = 0.0

        else:

            # Pelota perdida: penalización por cada paso sin verla.

            r_vivo = r_centro = r_tamano = 0.0

            r_perdido = -config.LOST_PENALTY

        # Penaliza los cambios bruscos de velocidad angular.

        r_suavidad = -(config.SMOOTHNESS_WEIGHT

                       * abs(float(cambio_giro))

                       / config.MAX_ANGULAR_CHANGE_RAD_S)

        # Penaliza el módulo de la velocidad angular, para que el robot no

        # gire más de lo necesario y se mantenga en su carril.

        r_regularizacion_giro = -config.ANGULAR_PENALTY_WEIGHT * abs(self.giro_anterior) / config.MAX_ANGULAR_SPEED_RAD_S

        # Penaliza invertir el sentido del giro (zigzag).

        r_zigzag = (-config.ZIGZAG_PENALTY

                    if giro_previo * self.giro_anterior < 0 else 0.0)

        # Penalización terminal por cualquier fallo.

        r_fallo = -config.FAILURE_PENALTY if fallo else 0.0

        recompensa = (r_vivo + r_centro + r_tamano + r_perdido

                      + r_suavidad + r_zigzag + r_fallo + r_regularizacion_giro)
 
        # 6. Motivo de fin, terminación (fallo) y truncamiento (duración).

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

        # Información de diagnóstico; '_accion_ejecutada' indica la acción

        # de demostración realmente aplicada, si la hubo.

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

            'r_centro': float(r_centro),

            'r_tamano': float(r_tamano),

            'r_suavidad': float(r_suavidad),

        }

        if self.verbose:

            print('paso={} vis={} ex={:+.2f} tam={:.3f} w={:+.3f} ts={} | '

                  'centro={:.2f} tam={:.2f} zig={:+.1f} | r={:+.2f}'

                  .format(self.pasos, int(info['visible']),

                          info['error_x'], info['tamano_norm'],

                          self.giro_anterior, self.frame_ts,

                          r_centro, r_tamano, r_zigzag, recompensa))

        return obs, float(recompensa), terminated, truncated, info
 
    def close(self):

        """Detiene el robot, destruye el nodo y cierra rclpy."""

        try:

            self._detener()

        except Exception:

            pass

        self.nodo.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()
 