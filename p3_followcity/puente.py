"""Lanza y relanza el puente robobo_ros2 desde nuestros scripts (solo simulación).

Por qué: el protocolo exige reiniciar la simulación (``reset_simulation``) al
empezar cada episodio, para que guía y seguidor partan de la posición
inicial. En RoboboSim 4.4.6 con el puente actual, ese reinicio cierra las
conexiones del puente con el simulador:

- la conexión de control (robobopy, puerto 40404): dejan de funcionar
  ``cmd_vel`` y ``move_tilt``, aunque los topics siguen republicando el último
  valor (IR y detected_object quedan congelados);
- la conexión de vídeo (puerto 40405): ``image_raw`` deja de publicar.

El puente no se reconecta solo. En lugar de pedir que alguien lo relance a
mano, el script de entrenamiento/evaluación lanza el puente como proceso hijo
y, tras cada reinicio, comprueba que el robot responde (``move_tilt``) y que
llegan imágenes nuevas. Si algo falla, relanza el puente (unos segundos) y
sigue. Si en vuestro PC el reinicio no rompe nada, no se relanza nunca.

Así un entrenamiento largo corre en una sola terminal, sin intervención.
"""

import os
import signal
import subprocess
import time
from pathlib import Path


class GestorPuente:
    def __init__(self, cfg, ip=None, ruta_log=None, log=print):
        """cfg: sección ``simulacion`` del perfil (puente_ip, puente_modulos,
        puente_timeout_s). ip: sustituye a puente_ip (p. ej. 127.0.0.1)."""
        self.ip = ip or cfg.puente_ip
        self.modulos = list(cfg.puente_modulos)
        self.timeout_s = float(cfg.puente_timeout_s)
        self.ruta_log = Path(ruta_log) if ruta_log else None
        self.log = log
        self.proc = None
        self.lanzamientos = 0

    def comando(self):
        modulos = '[' + ','.join("'{}'".format(m) for m in self.modulos) + ']'
        return ['ros2', 'run', 'robobo_ros2', 'robobo_container', '--ros-args',
                '-p', 'ip:={}'.format(self.ip), '-p', 'modules:={}'.format(modulos)]

    def vivo(self):
        return self.proc is not None and self.proc.poll() is None

    def lanzar(self):
        self.parar()
        if self.ruta_log:
            self.ruta_log.parent.mkdir(parents=True, exist_ok=True)
            salida = open(self.ruta_log, 'ab')
            salida.write('\n===== lanzamiento {} · {} =====\n'.format(
                self.lanzamientos + 1, time.strftime('%Y-%m-%d %H:%M:%S')).encode())
            salida.flush()
        else:
            salida = subprocess.DEVNULL
        # Sesión propia: se puede parar el puente entero (ros2 run + nodo) sin tocar este proceso.
        self.proc = subprocess.Popen(self.comando(), stdout=salida, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        self.lanzamientos += 1
        self.log('Puente lanzado (pid {}, lanzamiento {}): {}'.format(
            self.proc.pid, self.lanzamientos, ' '.join(self.comando())))

    def parar(self, timeout_s=8.0):
        if not self.vivo():
            self.proc = None
            return
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.proc.wait(timeout=timeout_s if sig != signal.SIGKILL else 3.0)
                break
            except subprocess.TimeoutExpired:
                continue
        self.proc = None
