"""Adaptador ROS 2 de la percepción del guía.

Conecta ``smartphone/detected_object`` con ``PercepcionGuia``. Es el mismo
código en simulación y en el robot real: los nombres de topics y todos los
umbrales salen del perfil.
"""

import time

import rclpy
from robobo_ros2_interfaces.msg import DetectedObject

from percepcion_guia import PercepcionGuia

try:
    from robobo_ros2_interfaces.srv import StartObjectRecognition
except ImportError:  # versiones del puente sin el servicio
    StartObjectRecognition = None


class SuscriptorGuia:
    """Suscripción a detected_object que alimenta una PercepcionGuia."""

    def __init__(self, nodo, perfil, iniciar_reconocimiento=True, log=None):
        self.nodo = nodo
        self.perfil = perfil
        self.percepcion = PercepcionGuia(perfil.guia)
        self.n_mensajes = 0
        self.ultimo_mensaje = None  # último DetectedObject sin filtrar
        self._log = log or (lambda texto: nodo.get_logger().info(texto))
        topic = perfil.topic('topic_objeto')
        nodo.create_subscription(DetectedObject, topic, self._callback, 10)
        if iniciar_reconocimiento:
            self.iniciar_reconocimiento()

    def _callback(self, msg):
        self.n_mensajes += 1
        self.ultimo_mensaje = msg
        self.percepcion.actualizar(msg, t_llegada=time.monotonic())

    def iniciar_reconocimiento(self, timeout_s=3.0):
        """Llama a start_object_recognition si existe.

        El nodo del puente lo arranca solo al crearse, pero tras un reinicio
        de la simulación o del teléfono conviene pedirlo de nuevo.
        """
        if StartObjectRecognition is None or 'srv_iniciar_reconocimiento' not in self.perfil.ros:
            return False
        cliente = self.nodo.create_client(
            StartObjectRecognition, self.perfil.topic('srv_iniciar_reconocimiento'))
        if not cliente.wait_for_service(timeout_sec=timeout_s):
            self._log('Aviso: no responde start_object_recognition; '
                      "¿se lanzó el puente con el módulo 'object'?")
            return False
        futuro = cliente.call_async(StartObjectRecognition.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=timeout_s)
        respuesta = futuro.result()
        return bool(respuesta is not None and respuesta.success)

    def reiniciar(self):
        self.percepcion.reiniciar(time.monotonic())

    def observar(self):
        return self.percepcion.observar(time.monotonic())

    def esperar_guia(self, timeout_s, spin=None):
        """Bloquea hasta ver al guía o agotar ``timeout_s``. Devuelve bool."""
        spin = spin or (lambda: rclpy.spin_once(self.nodo, timeout_sec=0.05))
        limite = time.monotonic() + timeout_s
        while time.monotonic() < limite:
            spin()
            if self.observar().visible:
                return True
        return False
