"""Adaptador ROS 2 de la percepción del carril.

Se suscribe a ``smartphone/camera/image_raw`` (sensor_msgs/Image, bgr8),
guarda solo el último fotograma con su instante de llegada y lo procesa bajo
demanda (en cada paso de control), para no gastar CPU en fotogramas que la
política nunca usará. Mismo código en simulación y en real.
"""

import time

import numpy as np
import rclpy
from sensor_msgs.msg import Image

from percepcion_carril import ObservacionCarril, PercepcionCarril


def imagen_a_numpy(msg):
    """sensor_msgs/Image (bgr8 o rgb8) -> array BGR (alto, ancho, 3) sin cv_bridge."""
    datos = np.frombuffer(msg.data, np.uint8)
    paso = msg.step if msg.step else msg.width * 3
    img = datos.reshape(msg.height, paso)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == 'rgb8':
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


class SuscriptorCarril:
    """Suscripción a image_raw que alimenta una PercepcionCarril."""

    def __init__(self, nodo, perfil):
        self.nodo = nodo
        self.perfil = perfil
        self.percepcion = PercepcionCarril(perfil.carril, perfil.seguridad)
        self.antiguedad_max = float(perfil.camara.antiguedad_max_imagen_s)
        self.n_imagenes = 0
        self._msg = None
        self._t_llegada = None
        self._procesada = None          # n_imagenes del último fotograma procesado
        nodo.create_subscription(Image, perfil.topic('topic_imagen'), self._callback, 1)

    def _callback(self, msg):
        self._msg = msg
        self._t_llegada = time.monotonic()
        self.n_imagenes += 1

    def reiniciar(self):
        self.percepcion.reiniciar(time.monotonic())
        self._procesada = None

    def ultima_imagen(self):
        """(imagen BGR, instante de llegada) o (None, None)."""
        if self._msg is None:
            return None, None
        return imagen_a_numpy(self._msg), self._t_llegada

    def observar(self):
        """Procesa el último fotograma si es nuevo y está dentro de la antigüedad máxima."""
        ahora = time.monotonic()
        if self._msg is None or ahora - self._t_llegada > self.antiguedad_max:
            # Sin imagen reciente: cuenta como carril no visto.
            obs = ObservacionCarril(motivo_invalida='imagen_antigua' if self._msg else 'sin_imagen')
            obs.edad_imagen_s = (ahora - self._t_llegada) if self._t_llegada else float('inf')
            obs.tiempo_sin = min(1.0, self.percepcion.segundos_sin_carril(ahora)
                                 / self.percepcion.tiempo_max_sin)
            return obs
        if self._procesada == self.n_imagenes:
            return self.percepcion.ultima       # mismo fotograma: no se reprocesa
        self._procesada = self.n_imagenes
        img, t_img = self.ultima_imagen()
        return self.percepcion.procesar(img, t_imagen=t_img, t=ahora)

    def esperar_imagen(self, timeout_s, spin=None):
        spin = spin or (lambda: rclpy.spin_once(self.nodo, timeout_sec=0.05))
        n0, limite = self.n_imagenes, time.monotonic() + timeout_s
        while time.monotonic() < limite:
            spin()
            if self.n_imagenes > n0:
                return True
        return False
