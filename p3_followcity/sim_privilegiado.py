"""Información privilegiada de RoboboSim (SOLO simulación).

Nada de este módulo llega a la observación de la política. Se usa para:

- reiniciar episodios con ``reset_simulation`` (ambos robots vuelven a la
  posición inicial de la escena, protocolo del anexo B de la P2),
- (opcional, solo depuración) colocar al seguidor detrás del guía con
  ``set_robot_location``,
- la recompensa de entrenamiento (si ``recompensa.usar_privilegiada``),
- las métricas de verdad en simulación: distancia al guía, desplazamiento
  lateral respecto al centro del carril y paradas del guía.

Centro del carril: el guía de RoboboSim recorre el carril por su centro, así
que su rastro (posiciones pasadas) es la línea central. El seguidor siempre
va detrás, por lo que siempre hay rastro donde está él.

Coordenadas de RoboboSim (mm, grados): con rumbo 0 el robot mira a +z y +x
queda a su IZQUIERDA; el ángulo ``rotation.y`` crece al girar a la izquierda
y el rumbo de un desplazamiento (dx, dz) es atan2(dx, dz).
"""

import math
import random
import time
from collections import deque

import rclpy
from robobo_ros2_interfaces.msg import RobotLocation, SimObjectArray
from robobo_ros2_interfaces.srv import ResetSimulation, SetRobotLocation

from recompensa_metricas import Privilegiada


class SimPrivilegiado:
    def __init__(self, nodo, perfil):
        self.nodo = nodo
        cfg = perfil.simulacion
        self.cfg = cfg
        self.ep = perfil.episodio
        self.id_guia = cfg.id_objeto_guia
        self.guia = None            # (x, z) mm
        self.guia_rumbo = None      # grados
        self.seg = None
        self.seg_rumbo = None
        self.y_seguidor = None
        self.n_poses = 0
        self.rastro = deque(maxlen=int(cfg.rastro_max_puntos))
        self._hist_guia = deque()   # (t, x, z) para la velocidad del guía
        self._desde_lento = None    # instante desde el que el guía va por debajo del umbral
        self.guia_arrancado = False
        nodo.create_subscription(SimObjectArray, cfg.topic_objetos, self._cb_objetos, 1)
        nodo.create_subscription(RobotLocation, cfg.topic_pose_seguidor, self._cb_seguidor, 1)
        self.cli_colocar = nodo.create_client(SetRobotLocation, cfg.srv_colocar)
        self.cli_reset = nodo.create_client(ResetSimulation, cfg.srv_reset)

    def olvidar(self):
        """Descarta poses, rastro e historial (tras reiniciar la escena)."""
        self.guia = self.guia_rumbo = self.seg = self.seg_rumbo = None
        self.rastro.clear()
        self._hist_guia.clear()
        self._desde_lento = None
        self.guia_arrancado = False

    def reiniciar_simulacion(self, timeout_s=15.0):
        """Llama a ``reset_simulation``: guía y seguidor a la posición inicial."""
        if not self.cli_reset.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('No responde {} (¿módulo sim del puente?)'.format(self.cfg.srv_reset))
        fut = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, fut, timeout_sec=timeout_s)
        ok = fut.done() and fut.result() is not None and fut.result().success
        self.olvidar()
        return bool(ok)

    # ------------------------------------------------------------------
    def _cb_objetos(self, msg):
        for o in msg.objects:
            if str(o.object_id) == self.id_guia:
                p = (float(o.position.x), float(o.position.z))
                self.guia, self.guia_rumbo = p, float(o.rotation.y)
                ahora = time.monotonic()
                self._hist_guia.append((ahora,) + p)
                while self._hist_guia and ahora - self._hist_guia[0][0] > self.cfg.ventana_velocidad_s:
                    self._hist_guia.popleft()
                if not self.rastro or math.dist(p, self.rastro[-1]) >= self.cfg.rastro_paso_mm:
                    self.rastro.append(p)
                self.n_poses += 1
                return

    def _cb_seguidor(self, msg):
        self.seg = (float(msg.position.x), float(msg.position.z))
        self.seg_rumbo = float(msg.rotation.y)
        self.y_seguidor = float(msg.position.y)
        self.n_poses += 1

    def listo(self):
        return self.guia is not None and self.seg is not None

    def esperar(self, timeout_s, spin):
        limite = time.monotonic() + timeout_s
        while not self.listo() and time.monotonic() < limite:
            spin()
        return self.listo()

    # ------------------------------------------------------------------
    def velocidad_guia_m_s(self):
        """Velocidad media del guía en la ventana (None si la ventana no está llena).

        Se usa la ventana completa y no dos mensajes seguidos porque el
        simulador republica la misma pose varias veces antes de actualizarla.
        """
        h = self._hist_guia
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.8 * self.cfg.ventana_velocidad_s:
            return None
        return math.dist(h[0][1:], h[-1][1:]) / 1000.0 / (h[-1][0] - h[0][0])

    def guia_parado(self):
        """Parado si la velocidad lleva >= parada_min_s por debajo del umbral (histéresis)."""
        vel = self.velocidad_guia_m_s()
        if vel is None:
            return None, None
        if not self.guia_arrancado:
            # Al empezar, el guía espera quieto a que el seguidor se acerque: eso no
            # es una parada de C4. Las paradas cuentan desde que arranca por primera vez.
            self.guia_arrancado = vel >= 2.0 * self.cfg.v_guia_parado_m_s
            return (False if self.guia_arrancado else None), vel
        ahora = time.monotonic()
        if vel < self.cfg.v_guia_parado_m_s:
            if self._desde_lento is None:
                self._desde_lento = ahora
            return ahora - self._desde_lento >= self.cfg.parada_min_s, vel
        self._desde_lento = None
        return False, vel

    def lateral_m(self):
        """Distancia con signo del seguidor a la trayectoria del guía (> 0: a la derecha)."""
        if self.seg is None or len(self.rastro) < 2:
            return None
        pts = list(self.rastro)
        mejor = None
        n = len(pts) - 1
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            ax, az = a
            dx, dz = b[0] - ax, b[1] - az
            l2 = dx * dx + dz * dz
            if l2 < 1e-9:
                continue
            u_bruto = ((self.seg[0] - ax) * dx + (self.seg[1] - az) * dz) / l2
            u = max(0.0, min(1.0, u_bruto))
            px, pz = ax + u * dx, az + u * dz
            d = math.hypot(self.seg[0] - px, self.seg[1] - pz)
            if mejor is None or d < mejor[0]:
                # Antes del inicio o después del final del rastro, la distancia al
                # extremo no es un desplazamiento lateral.
                extremo = (i == 0 and u_bruto < 0.0) or (i == n - 1 and u_bruto > 1.0)
                # Signo: +x es la IZQUIERDA del avance; producto vectorial en (x, z).
                cruz = dx * (self.seg[1] - az) - dz * (self.seg[0] - ax)
                mejor = (d, 1.0 if cruz > 0 else -1.0, extremo)
        if mejor is not None and mejor[2]:
            return None
        if mejor is None or mejor[0] > self.cfg.distancia_max_rastro_mm:
            return None     # sin rastro cerca (aún no se ha recorrido ese tramo)
        return mejor[0] * mejor[1] / 1000.0

    def punto_referencia(self):
        """Punto ``punto_referencia_detras_m`` detrás del centro del guía (anexo B de
        la P2: la pelota verde estaba 0,08 m detrás; sin pelota se usa ese punto
        sobre la pose del guía, como permite la nota del anexo)."""
        d = self.cfg.punto_referencia_detras_m * 1000.0
        th = math.radians(self.guia_rumbo or 0.0)
        return self.guia[0] - d * math.sin(th), self.guia[1] - d * math.cos(th)

    def medir(self):
        if not self.listo():
            return Privilegiada()
        parado, vel = self.guia_parado()
        ref = self.punto_referencia()
        dx, dz = ref[0] - self.seg[0], ref[1] - self.seg[1]
        desv = None
        if self.seg_rumbo is not None and math.hypot(dx, dz) > 1.0:
            desv = (math.degrees(math.atan2(dx, dz)) - self.seg_rumbo + 180.0) % 360.0 - 180.0
        return Privilegiada(
            distancia_m=math.hypot(dx, dz) / 1000.0,
            desviacion_deg=desv,
            lateral_m=self.lateral_m(),
            guia_parado=parado,
            guia_velocidad_m_s=vel)

    # ------------------------------------------------------------------
    def colocar_detras_del_guia(self, rng=random, spin=None):
        """Coloca al seguidor detrás del guía, en su carril, con variaciones.

        Distancia, desplazamiento lateral y giro se sortean en los rangos de la
        sección ``episodio`` del perfil. La dirección del carril se toma del
        rumbo del guía (en curva es una aproximación de cuerda).
        """
        if self.guia is None:
            raise RuntimeError('Sin pose del guía (¿topic {}?)'.format(self.cfg.topic_objetos))
        d = rng.uniform(*self.ep.distancia_inicial_m) * 1000.0
        lat = rng.uniform(-1, 1) * self.ep.lateral_inicial_m * 1000.0
        giro = rng.uniform(-1, 1) * self.ep.giro_inicial_deg
        th = math.radians(self.guia_rumbo)
        fx, fz = math.sin(th), math.cos(th)          # avance del guía
        lx, lz = fz, -fx                              # hacia la izquierda del avance (+x a rumbo 0)
        x = self.guia[0] - d * fx - lat * lx          # lat > 0 = a la derecha
        z = self.guia[1] - d * fz - lat * lz
        pet = SetRobotLocation.Request()
        pet.position.x, pet.position.z = float(x), float(z)
        pet.position.y = float(self.y_seguidor if self.y_seguidor is not None else self.cfg.altura_robot_mm)
        pet.rotation.y = float((self.guia_rumbo + giro) % 360.0)
        if not self.cli_colocar.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('No responde {}'.format(self.cfg.srv_colocar))
        fut = self.cli_colocar.call_async(pet)
        rclpy.spin_until_future_complete(self.nodo, fut, timeout_sec=10.0)
        return {'distancia_inicial_m': d / 1000.0, 'lateral_inicial_m': lat / 1000.0,
                'giro_inicial_deg': giro}
