#!/usr/bin/env python3
"""Entorno Gymnasium de la P3: seguir al guía por la RoboboCity (C4).

El mismo código se ejecuta en RoboboSim y en el Robobo real; solo cambia el
perfil (``--perfil sim|real``). Toda la percepción y la actuación pasan por el
puente robobo_ros2:

  smartphone/detected_object  -> percepción del guía (percepcion_guia.py)
  smartphone/camera/image_raw -> percepción del carril (percepcion_carril.py)
  base/ir                     -> proximidad frontal (contactos)
  base/cmd_vel                <- acción (v, w)
  base/move_tilt              <- tilt fijo del perfil al arrancar

Observación (Box float32, 12 componentes, todas normalizadas):
   0 guia_visible          {0,1}   detección nueva, fiable y reciente
   1 guia_x                [-1,1]  posición horizontal (0 = centrado)
   2 guia_tamano           [0,1]   tamaño aparente (crece al acercarse)
   3 guia_x_ultima         [-1,1]  última x válida (hacia dónde se perdió)
   4 guia_tiempo_perdido   [0,1]   s sin verlo / tiempo_perdido_max_s
   5 carril_validez        [0,1]   indicador continuo de validez del carril
   6 carril_desplazamiento [-1,1]  anchos de carril (> 0: robot a la derecha)
   7 carril_angulo         [-1,1]  ángulo al punto adelantado / escala
   8 carril_tiempo_sin     [0,1]   s sin carril válido / tiempo_max_sin_carril_s
   9 ir_frontal            [0,1]   máxima proximidad IR frontal
  10 v_anterior            [0,1]   velocidad lineal enviada en el paso anterior
  11 w_anterior            [-1,1]  velocidad angular enviada en el paso anterior
Cuando el guía o el carril no son válidos, sus componentes de posición valen
0 y la política dispone del indicador de validez y del tiempo sin dato. Las
acciones anteriores (10, 11) compensan en parte el retardo de la cámara.

Acción (Box [-1,1]², continua):
   0 -> v = v_min + (a0+1)/2·(v_max - v_min)   (v_min = 0: puede detenerse en C4)
   1 -> w_objetivo = a1·w_max, limitado a ±dw_max de cambio por paso
Se envía (v·escala_v, w·escala_w) por cmd_vel, reenviada cada
``periodo_reenvio_cmd_vel_s`` durante ``periodo_control_s``.

PAN/TILT: fuera del espacio de acción. El tilt se fija al valor del perfil
(105° en sim) porque la geometría del carril (horizonte, ancho por píxel) está
calibrada para esa inclinación; moverlo cambiaría el significado de las
observaciones del carril y del tamaño del guía.

Parada operativa (igual en sim y real): contacto (IR o, en sim, distancia),
guía sin ver durante ``tiempo_max_sin_guia_s`` o carril sin validar durante
``tiempo_max_sin_carril_s``. Se detiene el robot y termina el episodio. En
evaluación cuenta como intervención (recorrido no completado).

Fin de episodio (protocolo del anexo B de la P2): colisión (separación
< 0,12 m), guía perdido (separación > 0,90 m o sin verlo > 3 s), parada
operativa por carril, o fin del tiempo (60 s entrenamiento, 120 s evaluación).

Reinicio: en simulación SIEMPRE con ``reset_simulation`` (modo_reinicio
"reset_simulation"): guía y seguidor vuelven a la posición inicial de la
escena y el seguidor parte parado; el guía no arranca hasta que el seguidor se
le acerca. Tras el reinicio se vuelve a fijar el tilt y se comprueba que el
robot responde y que llegan imágenes nuevas; si el reinicio ha roto las
conexiones del puente, el propio entorno lo relanza (puente.py,
``simulacion.gestionar_puente``), sin intervención. Después se espera a ver
guía y carril. (modo "colocar": solo depuración.)
En real se detiene el robot y se espera a que el operador lo coloque detrás
del guía (guía visible y carril válido).
"""

import math
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

from percepcion_carril import ObservacionCarril
from recompensa_metricas import Evaluador, Paso, Privilegiada
from registro import Registros
from ros_carril import SuscriptorCarril
from ros_guia import SuscriptorGuia

NOMBRES_OBS = ('guia_visible', 'guia_x', 'guia_tamano', 'guia_x_ultima',
               'guia_tiempo_perdido', 'carril_validez', 'carril_desplazamiento',
               'carril_angulo', 'carril_tiempo_sin', 'ir_frontal', 'v_anterior',
               'w_anterior')
OBS_BAJO = np.array([0, -1, 0, -1, 0, 0, -1, -1, 0, 0, 0, -1], np.float32)
OBS_ALTO = np.ones(len(NOMBRES_OBS), np.float32)


class FollowCityP3Env(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, perfil, modo='entrenamiento', configuracion='C4',
                 carpeta_registros=None, etiqueta='p3', semilla=None, verbose=False,
                 ip_puente=None, puente_externo=False):
        """
        perfil: Perfil cargado con perfil.cargar_perfil().
        modo: 'entrenamiento' (episodios cortos, aleatorización activa) o
              'evaluacion' (episodios largos, sin aleatorización).
        configuracion: C1-C4, solo se anota en los registros (la escena se
              configura en el simulador o la programa el profesorado en real).
        carpeta_registros: si se indica, se guardan pasos.csv y episodios.csv.
        ip_puente: IP del simulador para el puente que lanza el entorno (por
              defecto simulacion.puente_ip del perfil).
        puente_externo: no lanzar el puente (ya lanzado a mano); entonces un
              reinicio que lo rompa termina con error.
        """
        super().__init__()
        self.perfil, self.modo, self.configuracion = perfil, modo, configuracion
        self.verbose = verbose
        self.rng = random.Random(semilla)
        self.np_rng = np.random.default_rng(semilla)
        p, s, ep = perfil.politica, perfil.seguridad, perfil.episodio
        self.v_min, self.v_max = p.v_min_m_s, p.v_max_m_s
        self.w_max, self.dw_max = p.w_max_rad_s, p.dw_max_rad_s
        self.periodo = s.periodo_control_s
        self.periodo_reenvio = s.periodo_reenvio_cmd_vel_s
        duracion = ep.duracion_entrenamiento_s if modo == 'entrenamiento' else ep.duracion_evaluacion_s
        self.pasos_max = max(1, int(round(duracion / self.periodo)))
        al = perfil.aleatorizacion
        self.aleatorizar = bool(al.activa) and modo == 'entrenamiento'

        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p3_follow_city_env')
        self.pub_vel = self.nodo.create_publisher(Twist, perfil.topic('topic_cmd_vel'), 10)
        self._ir, self._n_ir = None, 0
        self.nodo.create_subscription(Int32MultiArray, perfil.topic('topic_ir'), self._cb_ir, 1)
        self.guia = SuscriptorGuia(self.nodo, perfil, log=self._log)
        self.carril = SuscriptorCarril(self.nodo, perfil)
        self.sim = None
        self.gestor = None
        self.relanzamientos_puente = 0
        if perfil.es_simulacion:
            from sim_privilegiado import SimPrivilegiado
            self.sim = SimPrivilegiado(self.nodo, perfil)
            if perfil.simulacion.gestionar_puente and not puente_externo:
                self._arrancar_puente(ip_puente, carpeta_registros)
        self.evaluador = Evaluador(perfil)
        self.registros = Registros(carpeta_registros, '{}_{}_{}'.format(etiqueta, modo, configuracion)) \
            if carpeta_registros else None

        self.observation_space = spaces.Box(OBS_BAJO, OBS_ALTO, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.episodio = 0
        self.pasos = 0
        self.v_ant, self.w_ant = 0.0, 0.0
        self.factor_v = self.factor_w = 1.0
        self._t_ep = time.monotonic()
        self._fin_registrado = True
        self.guardar_imagenes_cada = 0      # >0: guarda 1 imagen de diagnóstico cada N pasos
        self._mover_tilt()

    def configurar(self, modo=None, carpeta_registros=None, etiqueta='p3', configuracion=None):
        """Cambia de modo (entrenamiento/evaluacion) y, si se indica, de carpeta de
        registros, sin crear otro nodo ROS. Cierra el episodio en curso."""
        self._cerrar_episodio(completado=False, motivo='cambio_de_modo')
        self._detener()
        if configuracion is not None:
            self.configuracion = configuracion
        if modo is not None:
            self.modo = modo
            ep = self.perfil.episodio
            duracion = ep.duracion_entrenamiento_s if modo == 'entrenamiento' else ep.duracion_evaluacion_s
            self.pasos_max = max(1, int(round(duracion / self.periodo)))
            self.aleatorizar = bool(self.perfil.aleatorizacion.activa) and modo == 'entrenamiento'
        if carpeta_registros is not None:
            if self.registros is not None:
                self.registros.cerrar()
            self.registros = Registros(carpeta_registros, '{}_{}_{}'.format(
                etiqueta, self.modo, self.configuracion))
        self.episodio = 0

    # ------------------------------------------------------------------
    # ROS
    # ------------------------------------------------------------------
    def _log(self, texto):
        if self.verbose:
            print('[p3_env]', texto, flush=True)

    def _cb_ir(self, msg):
        self._ir = list(msg.data)
        self._n_ir += 1

    def _spin(self, segundos=0.0):
        fin = time.monotonic() + segundos
        rclpy.spin_once(self.nodo, timeout_sec=0.0)
        while time.monotonic() < fin:
            rclpy.spin_once(self.nodo, timeout_sec=min(0.02, max(0.0, fin - time.monotonic())))

    def _publicar(self, v, w):
        t = Twist()
        t.linear.x = float(v * self.perfil.politica.escala_v * self.factor_v)
        t.angular.z = float(w * self.perfil.politica.escala_w * self.factor_w)
        self.pub_vel.publish(t)

    def _detener(self):
        for _ in range(2):
            self._publicar(0.0, 0.0)
            self._spin(0.05)

    def _mover_tilt(self):
        """Fija el tilt del perfil. Devuelve True si el puente lo confirma: sirve
        también para saber si la conexión de control con el robot sigue viva."""
        try:
            from robobo_ros2_interfaces.srv import MoveTilt
        except ImportError:
            return True
        cam = self.perfil.camara
        if getattr(self, '_cli_tilt', None) is None:
            self._cli_tilt = self.nodo.create_client(MoveTilt, self.perfil.topic('accion_tilt'))
        cli = self._cli_tilt
        if not cli.wait_for_service(timeout_sec=5.0):
            return False
        pet = MoveTilt.Request()
        pet.angle, pet.speed = float(cam.tilt_deg), float(cam.tilt_velocidad_deg_s)
        fut = cli.call_async(pet)
        rclpy.spin_until_future_complete(self.nodo, fut, timeout_sec=cam.timeout_movimiento_s)
        return fut.done() and fut.result() is not None

    # ------------------------------------------------------------------
    # Puente (solo simulación)
    # ------------------------------------------------------------------
    def _arrancar_puente(self, ip, carpeta_registros):
        from puente import GestorPuente
        from robobo_ros2_interfaces.srv import ResetSimulation
        cfg = self.perfil.simulacion
        cli = self.nodo.create_client(ResetSimulation, cfg.srv_reset)
        # Un puente que se acaba de cerrar (p. ej. el de un entrenamiento anterior)
        # tarda unos segundos en desaparecer de ROS: se espera hasta 20 s.
        limite, avisado = time.monotonic() + 20.0, False
        while cli.wait_for_service(timeout_sec=1.0) and time.monotonic() < limite:
            if not avisado:
                print('Hay un puente en marcha; esperando a que se cierre (hasta 20 s)...', flush=True)
                avisado = True
            time.sleep(1.0)
        if cli.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(
                'Ya hay un puente robobo_ros2 en marcha. Este script lanza y relanza su propio '
                'puente (simulacion.gestionar_puente): cerrad el otro (Ctrl+C en su terminal, o '
                '"pkill -f robobo_container" si quedó colgado; compruébalo con '
                '"pgrep -af robobo_container") y volved a lanzar, o pasad --puente-externo.')
        self.nodo.destroy_client(cli)
        ruta_log = Path(carpeta_registros or 'registros') / 'puente.log'
        self.gestor = GestorPuente(cfg, ip=ip, ruta_log=ruta_log, log=print)
        self._relanzar_puente(primera_vez=True)

    def _robot_responde(self, timeout_imagen_s):
        """Control vivo (move_tilt confirmado) e imágenes nuevas llegando."""
        if not self._mover_tilt():
            return False, 'move_tilt sin respuesta'
        if not self.carril.esperar_imagen(timeout_imagen_s, lambda: self._spin(0.02)):
            return False, 'sin imágenes nuevas'
        return True, ''

    def _relanzar_puente(self, primera_vez=False):
        cfg = self.perfil.simulacion
        for intento in range(3):
            self.gestor.lanzar()
            limite = time.monotonic() + cfg.puente_timeout_s
            motivo = 'no arranca'
            while time.monotonic() < limite and self.gestor.vivo():
                ok, motivo = self._robot_responde(5.0)
                if ok:
                    if not primera_vez:
                        self.relanzamientos_puente += 1
                    return
            self._log('Puente sin respuesta ({}), reintento {}'.format(motivo, intento + 1))
        raise RuntimeError('El puente no responde tras 3 lanzamientos. Revisad que el simulador '
                           'está abierto en Follow City y {}'.format(
                               self.gestor.ruta_log or 'el registro del puente'))

    def _ir_frontal(self):
        s = self.perfil.seguridad
        if not self._ir:
            return 0.0
        vals = [self._ir[i] for i in s.ir_indices_frontales if i < len(self._ir)]
        return min(1.0, max([max(int(v), 0) for v in vals] or [0]) / s.ir_saturacion)

    # ------------------------------------------------------------------
    # Observación
    # ------------------------------------------------------------------
    def _observar(self):
        og = self.guia.observar()
        zona = self.guia.percepcion.zona_exclusion()
        oc = self.carril.observar(excluir=[zona] if zona else None)
        if self.aleatorizar and og.visible and self.rng.random() < self.perfil.aleatorizacion.prob_perder_guia:
            og.visible, og.x, og.tamano = 0.0, 0.0, 0.0      # detección perdida simulada
        ir = self._ir_frontal()
        v_n = 0.0 if self.v_max <= self.v_min else (self.v_ant - self.v_min) / (self.v_max - self.v_min)
        obs = np.array([
            og.visible, og.x, og.tamano, og.x_ultima, og.tiempo_perdido,
            oc.validez if oc.valido else 0.0,
            oc.desplazamiento if oc.valido else 0.0,
            oc.angulo if oc.valido else 0.0,
            oc.tiempo_sin, ir, v_n, self.w_ant / self.w_max], np.float32)
        if self.aleatorizar and self.perfil.aleatorizacion.ruido_obs_std > 0:
            ruido = self.np_rng.normal(0.0, self.perfil.aleatorizacion.ruido_obs_std, obs.shape)
            ruido[[0, 9, 10, 11]] = 0.0          # indicadores y acciones sin ruido
            obs = obs + ruido.astype(np.float32)
        lim = self.perfil.politica.obs_recorte
        obs = np.clip(obs, np.maximum(OBS_BAJO, -lim), np.minimum(OBS_ALTO, lim)).astype(np.float32)
        return obs, og, oc, ir

    def _paso_metricas(self, og, oc, ir, v, w, dw, priv):
        return Paso(
            t=time.monotonic() - self._t_ep, guia_visible=og.visible, guia_tamano=og.tamano,
            guia_tiempo_perdido_s=self.guia.percepcion.segundos_sin_guia(),
            carril_valido=oc.valido, carril_validez=oc.validez,
            carril_desplazamiento=oc.desplazamiento,
            carril_tiempo_sin_s=self.carril.percepcion.segundos_sin_carril(),
            ir_frontal=ir, v=v, w=w, dw=dw, priv=priv)

    # ------------------------------------------------------------------
    # Gym
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self._cerrar_episodio(completado=False, motivo='reinicio')
        self._detener()
        ep = self.perfil.episodio
        info = {}
        if self.sim is not None:
            if not self.sim.esperar(10.0, lambda: self._spin(0.05)):
                raise RuntimeError('No llegan las poses del simulador (¿módulo sim del puente?).')
            cfg = self.perfil.simulacion
            listo = False
            for intento in range(int(ep.intentos_reinicio)):
                if cfg.modo_reinicio == 'reset_simulation':
                    if not self.sim.reiniciar_simulacion():
                        self._log('Aviso: reset_simulation no confirmó el reinicio')
                    self._spin(cfg.espera_tras_reset_s)
                    # El reinicio devuelve el tilt a su valor por defecto y puede cortar
                    # las conexiones del puente: se fija el tilt y se comprueba todo.
                    ok, fallo = self._robot_responde(cfg.timeout_imagen_tras_reset_s)
                    if not ok:
                        if self.gestor is None:
                            raise RuntimeError(
                                'Tras reset_simulation el puente no responde ({}). Lanzad el '
                                'script sin --puente-externo para que lo relance solo.'.format(fallo))
                        self._log('Tras el reinicio: {}; se relanza el puente'.format(fallo))
                        self._relanzar_puente()
                    self.sim.olvidar()          # descarta poses de antes del reinicio
                    if not self.sim.esperar(10.0, lambda: self._spin(0.05)):
                        raise RuntimeError('No llegan las poses del simulador tras el reinicio.')
                else:
                    info.update(self.sim.colocar_detras_del_guia(self.rng))
                    self._spin(ep.espera_tras_colocar_s)
                listo = self._esperar_percepcion(ep.espera_percepcion_s)
                if listo:
                    break
                self._log('Reinicio {} sin guía y carril válidos'.format(intento + 1))
            if not listo:
                print('Aviso: el episodio empieza sin guía o carril válidos.', flush=True)
                self.guia.reiniciar()
                self.carril.reiniciar()
            info['intentos_reinicio'] = intento + 1
            info['relanzamientos_puente'] = self.relanzamientos_puente
            m = self.sim.medir()
            info['separacion_inicial_m'] = m.distancia_m
        else:
            print('Coloca el seguidor detrás del guía, en su carril...', flush=True)
            if not self._esperar_percepcion(ep.espera_real_s):
                print('Aviso: se empieza sin confirmar guía y carril.', flush=True)
                # Los temporizadores de pérdida empiezan ahora, no al empezar la espera.
                self.guia.reiniciar()
                self.carril.reiniciar()

        if self.aleatorizar:
            al = self.perfil.aleatorizacion
            self.factor_v = self.rng.uniform(*al.escala_v_rango)
            self.factor_w = self.rng.uniform(*al.escala_w_rango)
        else:
            self.factor_v = self.factor_w = 1.0

        # Las percepciones se reiniciaron al empezar la espera: se conserva lo
        # visto desde entonces (guía y carril ya válidos al empezar).
        self.evaluador.reiniciar()
        self.v_ant = self.w_ant = 0.0
        self.pasos = 0
        self.episodio += 1
        self._t_ep = time.monotonic()
        self._fin_registrado = False
        self._spin(self.periodo)
        obs, og, oc, ir = self._observar()
        info.update({'episodio': self.episodio, 'factor_v': self.factor_v, 'factor_w': self.factor_w})
        return obs, info

    def _esperar_percepcion(self, timeout_s):
        """Espera a que guía y carril sean válidos en ``observaciones_validas_reinicio``
        imágenes nuevas seguidas. Así el episodio empieza con imágenes tomadas
        en la posición nueva, sea cual sea la latencia de la cámara (sim o real)."""
        ep = self.perfil.episodio
        self.guia.reiniciar()
        self.carril.reiniciar()
        limite, seguidas = time.monotonic() + timeout_s, 0
        while time.monotonic() < limite:
            if not self.carril.esperar_imagen(min(ep.timeout_imagen_s, max(0.1, limite - time.monotonic())),
                                              lambda: self._spin(0.02)):
                if time.monotonic() >= limite:
                    break
                raise RuntimeError(
                    'No llegan imágenes de {} en {} s. ¿Puente lanzado con el módulo camera? '
                    '(revisad registros/.../puente.log)'.format(
                        self.perfil.topic('topic_imagen'), ep.timeout_imagen_s))
            ok = self.guia.observar().visible and self.carril.observar().valido
            seguidas = seguidas + 1 if ok else 0
            if seguidas >= ep.observaciones_validas_reinicio:
                return True
        return False

    def _accion_a_velocidades(self, accion):
        a = np.clip(np.asarray(accion, np.float64), -1.0, 1.0)
        v = self.v_min + (a[0] + 1.0) / 2.0 * (self.v_max - self.v_min)
        w_obj = a[1] * self.w_max
        w = self.w_ant + float(np.clip(w_obj - self.w_ant, -self.dw_max, self.dw_max))
        return float(v), float(np.clip(w, -self.w_max, self.w_max))

    def step(self, accion):
        v, w = self._accion_a_velocidades(accion)
        dw = w - self.w_ant
        # Mantener la orden durante el periodo de control, reenviándola para
        # que no salte el watchdog del puente (0,5 s).
        fin = time.monotonic() + self.periodo
        while True:
            self._publicar(v, w)
            restante = fin - time.monotonic()
            if restante <= 0:
                break
            self._spin(min(self.periodo_reenvio, restante))
        self.v_ant, self.w_ant = v, w
        self.pasos += 1

        obs, og, oc, ir = self._observar()
        priv = self.sim.medir() if self.sim is not None else Privilegiada()
        paso = self._paso_metricas(og, oc, ir, v, w, dw, priv)
        motivo = self.evaluador.parada_operativa(paso)
        if not motivo and priv.distancia_m is not None and \
                priv.distancia_m > self.perfil.metricas.distancia_perdido_m:
            motivo = 'guia_lejos'
        terminado = bool(motivo)
        truncado = not terminado and self.pasos >= self.pasos_max
        recompensa, desglose = self.evaluador.recompensa(paso, motivo)
        self.evaluador.actualizar_metricas(paso)
        if motivo:
            self.evaluador.motivo_fin = motivo
            self._detener()

        self._registrar_paso(obs, og, oc, ir, accion, v, w, recompensa, desglose, priv, paso, motivo)
        info = {'motivo_fin': motivo, 'desglose': desglose}
        if terminado or truncado:
            info['metricas'] = self._cerrar_episodio(completado=truncado, motivo=motivo or 'tiempo')
        return obs, float(recompensa), terminado, truncado, info

    # ------------------------------------------------------------------
    # Registros
    # ------------------------------------------------------------------
    def _registrar_paso(self, obs, og, oc, ir, accion, v, w, r, desglose, priv, paso, motivo):
        if self.registros is None:
            return
        verdad, estimada = self.evaluador.fuera_de_carril(paso)
        fila = {
            't_s': round(time.monotonic() - self.registros.t0, 3),
            't_unix': round(time.time(), 3), 'episodio': self.episodio, 'paso': self.pasos,
            'configuracion': self.configuracion, 'perfil': self.perfil.nombre, 'modo': self.modo,
        }
        fila.update({'obs_' + n: float(x) for n, x in zip(NOMBRES_OBS, obs)})
        fila.update({'accion_0': float(accion[0]), 'accion_1': float(accion[1]),
                     'cmd_v_m_s': v, 'cmd_w_rad_s': w,
                     'cmd_v_enviada_m_s': v * self.perfil.politica.escala_v * self.factor_v,
                     'cmd_w_enviada_rad_s': w * self.perfil.politica.escala_w * self.factor_w})
        b = og.bruta or {}
        fila.update({'guia_visible': og.visible, 'guia_confianza': og.confianza,
                     'guia_motivo_invalida': og.motivo_invalida, 'guia_edad_s': None
                     if math.isinf(og.edad_s) else og.edad_s,
                     'guia_x_px': b.get('x_px'), 'guia_y_px': b.get('y_px'),
                     'guia_ancho_px': b.get('ancho_px'), 'guia_alto_px': b.get('alto_px')})
        fila.update(oc.registro() if isinstance(oc, ObservacionCarril) else {})
        fila.update({'ir_frontal': ir, 'recompensa': r,
                     'fuera_carril_verdad': None if verdad is None else int(verdad),
                     'fuera_carril_estimada': int(estimada),
                     'contacto': int(self.evaluador.contacto(paso)),
                     'priv_distancia_m': priv.distancia_m, 'priv_lateral_m': priv.lateral_m,
                     'priv_guia_parado': None if priv.guia_parado is None else int(priv.guia_parado),
                     'priv_guia_velocidad_m_s': priv.guia_velocidad_m_s,
                     'motivo_fin': motivo})
        self.registros.pasos.escribir(fila)
        if self.guardar_imagenes_cada and self.pasos % self.guardar_imagenes_cada == 0:
            self._guardar_imagen(og, oc, v, w)

    def _guardar_imagen(self, og, oc, v, w):
        """Imagen con la detección del carril y del guía dibujadas (diagnóstico)."""
        import cv2
        from percepcion_carril import PercepcionCarril
        img, _ = self.carril.ultima_imagen()
        if img is None:
            return
        img = PercepcionCarril.dibujar(img, oc)
        b = og.bruta or {}
        if b:
            x, y, an, al = b['x_px'], b['y_px'], b['ancho_px'], b['alto_px']
            color = (0, 0, 255) if og.visible else (128, 128, 128)
            cv2.rectangle(img, (int(x - an / 2), int(y - al / 2)), (int(x + an / 2), int(y + al / 2)), color, 2)
        alto = img.shape[0]
        cv2.rectangle(img, (0, alto - 22), (img.shape[1], alto), (0, 0, 0), -1)
        cv2.putText(img, 'ep{} p{} guia vis={:.0f} tam={:.2f} | v={:.2f} w={:+.2f}'.format(
            self.episodio, self.pasos, og.visible, og.tamano, v, w), (4, alto - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        carpeta = self.registros.carpeta / 'imagenes'
        carpeta.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(carpeta / 'ep{:03d}_p{:04d}.jpg'.format(self.episodio, self.pasos)), img)

    def _cerrar_episodio(self, completado, motivo):
        if self._fin_registrado:
            return None
        self._fin_registrado = True
        if self.evaluador.pasos == 0:
            return None          # reinicio sin pasos (p. ej. comprobaciones): no es un recorrido
        if not self.evaluador.motivo_fin:
            self.evaluador.motivo_fin = 'completado' if completado else motivo
        resumen = self.evaluador.resumen(completado)
        fila = {'episodio': self.episodio, 'configuracion': self.configuracion,
                'perfil': self.perfil.nombre, 'modo': self.modo,
                'duracion_s': round(time.monotonic() - self._t_ep, 2),
                'factor_v': round(self.factor_v, 4), 'factor_w': round(self.factor_w, 4)}
        fila.update(resumen)
        if self.registros is not None:
            self.registros.episodios.escribir(fila)
        self._log('Episodio {}: {}'.format(self.episodio, fila))
        return fila

    def close(self):
        try:
            self._detener()
        except Exception:
            pass
        self._cerrar_episodio(completado=False, motivo='cierre')
        if self.registros is not None:
            self.registros.cerrar()
        try:
            self.nodo.destroy_node()
        except Exception:
            pass
        if self.gestor is not None:
            self.gestor.parar()
        super().close()
