#!/usr/bin/env python3
"""Evaluador del Anexo B con magnitudes de los sensores del seguidor.
 
La versión de RoboboSim utilizada no publica la posición del guía, así
que la separación y la desviación no pueden medirse con la verdad del
terreno. Como permite el Anexo B, justificándolo en la memoria, se
sustituyen por magnitudes del propio seguidor:
  - Guía perdido: pelota sin detectar más de 3 s (igual que el protocolo).
  - Colisión: proximidad frontal IR >= 0,85 o tamaño del blob >= 0,35.
  - Separación: tamaño normalizado del blob, banda objetivo [0,10, 0,30].
  - Desviación: x del blob convertida a grados con el semicampo de visión.
Se registra además la posición real del seguidor (sim/robot_location).
 
Uso:
  python3 evaluador_b.py --medir-campo                   (una vez)
  python3 evaluador_b.py --config C1 --politica sac \
      --modelo p1_follow_city_sac_seed2026 --etiqueta "SAC, semilla 2026"
  python3 evaluador_b.py --config C1 --politica aleatoria --etiqueta Aleatoria
  (repetir con C2 y C3, cambiando antes la configuración del guía)
  python3 evaluador_b.py --resumir
"""
 
import argparse
import csv
import math
import os
import time
 
import numpy as np
import rclpy
from robobo_ros2_interfaces.msg import RobotLocation
 
import follow_city_config as config
 
DURACION_EPISODIO_S = 120.0     # tiempo real, como el protocolo
SIN_DETECCION_S = 3.0
BANDA_TAMANO = (config.TARGET_BLOB_SIZE_NORM - config.SIZE_SIGMA,
                config.TARGET_BLOB_SIZE_NORM + config.SIZE_SIGMA)
DESVIACION_UMBRAL_DEG = 15.0
 
# Semicampo de visión horizontal de la cámara, en grados. Rellenar con el
# valor que imprime --medir-campo. Mientras valga None, la desviación se
# da como |x| normalizada.
SEMICAMPO_VISION_DEG = None
 
DIR_SALIDA = 'evaluacion_anexoB'
RUTA_EPISODIOS = os.path.join(DIR_SALIDA, 'episodios_planB.csv')
CONFIGURACIONES = ['C1', 'C2', 'C3']
 
CAMPOS_PASO = ['t_s', 'paso', 'seg_x_m', 'seg_z_m', 'seg_rumbo_deg',
               'blob_visible', 'blob_error_x', 'blob_tamano', 'proximidad_ir',
               'accion_avance', 'accion_giro', 'v_m_s', 'w_rad_s']
CAMPOS_EPISODIO = ['politica', 'config', 'episodio', 'desenlace', 'detalle',
                   'duracion_s', 'pasos', 'archivo_pasos']
 
 
def error_x_de(blob):
    return (float(blob.x) - config.IMAGE_WIDTH_PX / 2) / (config.IMAGE_WIDTH_PX / 2)
 
 
def desviacion_deg(error_x):
    """Ángulo a la pelota con un modelo de cámara estenopeica."""
    return math.degrees(math.atan(
        error_x * math.tan(math.radians(SEMICAMPO_VISION_DEG))))
 
 
class PoseSeguidor:
    """Posición real del seguidor, solo para registro (nunca para decidir)."""
 
    def __init__(self, nodo, ns):
        self.nodo = nodo
        self.x = self.z = self.rumbo = None
        nodo.create_subscription(RobotLocation, ns + '/robot_location',
                                 self._cb, 10)
 
    def _cb(self, msg):
        self.x = msg.position.x / 1000.0
        self.z = msg.position.z / 1000.0
        self.rumbo = float(msg.rotation.y)
 
    def esperar(self, timeout=5.0):
        limite = time.monotonic() + timeout
        while self.x is None:
            if time.monotonic() > limite:
                raise RuntimeError('No llega sim/robot_location del seguidor.')
            rclpy.spin_once(self.nodo, timeout_sec=0.05)
 
 
def crear_politica(args):
    if args.politica == 'aleatoria':
        rng = np.random.default_rng(args.seed)
        return lambda obs: rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    from stable_baselines3 import PPO, SAC
    modelo = (SAC if args.politica == 'sac' else PPO).load(args.modelo)
    return lambda obs: modelo.predict(obs, deterministic=True)[0]
 
 
def ejecutar_episodio(env, pose, politica, ruta_pasos):
    obs, _ = env.reset()
    pose.esperar()
    t0 = time.monotonic()
    ultima_deteccion = t0
    desenlace, detalle = 'completado', ''
    filas = []
 
    while time.monotonic() - t0 < DURACION_EPISODIO_S:
        accion = np.asarray(politica(obs), dtype=np.float32)
        try:
            obs, _, terminado, _, info = env.step(accion)
        except RuntimeError as error:
            desenlace, detalle = 'otra causa', 'error ROS: {}'.format(error)
            break
        ahora = time.monotonic()
        visible = obs[0] > 0.5
        if visible:
            ultima_deteccion = ahora
        filas.append({
            't_s': round(ahora - t0, 3), 'paso': len(filas) + 1,
            'seg_x_m': pose.x, 'seg_z_m': pose.z, 'seg_rumbo_deg': pose.rumbo,
            'blob_visible': int(visible), 'blob_error_x': float(obs[1]),
            'blob_tamano': float(obs[3]),
            'proximidad_ir': info.get('proximidad_frontal'),
            'accion_avance': float(accion[0]), 'accion_giro': float(accion[1]),
            'v_m_s': info.get('velocidad_lineal'),
            'w_rad_s': info.get('velocidad_angular')})
 
        if ahora - ultima_deteccion > SIN_DETECCION_S:
            desenlace, detalle = 'guía perdido', 'sin detección > 3 s'
            break
        if terminado:
            motivo = info.get('motivo_fin')
            if motivo == 'blob perdido':
                desenlace, detalle = 'guía perdido', 'sin detección 3 s'
            elif info.get('contacto_aproximado') or info.get('demasiado_cerca'):
                desenlace, detalle = 'colisión', str(motivo)
            else:
                desenlace, detalle = 'otra causa', str(motivo)
            break
 
    env._detener()
    with open(ruta_pasos, 'w', newline='', encoding='utf-8') as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_PASO)
        escritor.writeheader()
        escritor.writerows(filas)
    return {'desenlace': desenlace, 'detalle': detalle,
            'duracion_s': round(time.monotonic() - t0, 2), 'pasos': len(filas)}
 
 
def evaluar(args):
    from follow_city_env import FollowCityEnv
    os.makedirs(DIR_SALIDA, exist_ok=True)
    etiqueta = args.etiqueta or args.politica
    politica = crear_politica(args)
    env = FollowCityEnv(usar_demostracion_inicial=False)
    env.pasos_max = 10 ** 9   # el final lo marcan los 120 s del protocolo
    pose = PoseSeguidor(env.nodo, args.ns_seguidor)
    nueva = not os.path.isfile(RUTA_EPISODIOS)
    try:
        with open(RUTA_EPISODIOS, 'a', newline='', encoding='utf-8') as f:
            escritor = csv.DictWriter(f, fieldnames=CAMPOS_EPISODIO)
            if nueva:
                escritor.writeheader()
            for ep in range(1, args.episodios + 1):
                archivo = 'pasos_{}_{}_ep{}_{}.csv'.format(
                    ''.join(c for c in etiqueta if c.isalnum()), args.config,
                    ep, time.strftime('%H%M%S'))
                r = ejecutar_episodio(env, pose, politica,
                                      os.path.join(DIR_SALIDA, archivo))
                escritor.writerow({'politica': etiqueta, 'config': args.config,
                                   'episodio': ep, 'archivo_pasos': archivo, **r})
                f.flush()
                print('{} {} ep {}: {} ({}), {:.1f} s, {} pasos'.format(
                    etiqueta, args.config, ep, r['desenlace'], r['detalle'],
                    r['duracion_s'], r['pasos']))
    finally:
        env.close()
 
 
def num(x, d):
    return '{:.{}f}'.format(x, d).replace('.', ',')
 
 
def resumir():
    with open(RUTA_EPISODIOS, encoding='utf-8') as f:
        episodios = list(csv.DictReader(f))
    for e in episodios:
        with open(os.path.join(DIR_SALIDA, e['archivo_pasos']), encoding='utf-8') as f:
            e['_pasos'] = list(csv.DictReader(f))
    politicas = list(dict.fromkeys(e['politica'] for e in episodios))
    con_grados = SEMICAMPO_VISION_DEG is not None
 
    t3 = ['| Política | Configuración | Completados | Colisión | Guía perdido | Otra causa |',
          '| --- | --- | --- | --- | --- | --- |']
    cab = ['Política', 'Configuración', 'Pasos con pelota visible (%)',
           'Pasos en banda (%)', 'Tamaño medio del blob']
    cab += (['Desviación media (°)', 'Pasos con desviación < 15° (%)']
            if con_grados else ['Desviación media (|x| normalizada)'])
    t4 = ['| ' + ' | '.join(cab) + ' |', '|' + ' --- |' * len(cab)]
 
    for pol in politicas:
        for cfg in CONFIGURACIONES + ['Total']:
            sel = [e for e in episodios if e['politica'] == pol
                   and (cfg == 'Total' or e['config'] == cfg)]
            if not sel:
                continue
            n = len(sel)
            cuenta = lambda d: sum(e['desenlace'] == d for e in sel)
            comp = cuenta('completado')
            t3.append('| {} | {} | {}/{} ({} %) | {} | {} | {} |'.format(
                pol, cfg, comp, n, num(100 * comp / n, 0), cuenta('colisión'),
                cuenta('guía perdido'), cuenta('otra causa')))
 
            pasos = [p for e in sel for p in e['_pasos']]
            vis = [p for p in pasos if p['blob_visible'] == '1']
            if not pasos or not vis:
                continue
            tam = [float(p['blob_tamano']) for p in vis]
            ex = [float(p['blob_error_x']) for p in vis]
            lo, hi = BANDA_TAMANO
            fila = [pol, cfg, num(100 * len(vis) / len(pasos), 1),
                    num(100 * sum(lo <= t <= hi for t in tam) / len(vis), 1),
                    num(float(np.mean(tam)), 3)]
            if con_grados:
                d = [abs(desviacion_deg(x)) for x in ex]
                fila += [num(float(np.mean(d)), 1),
                         num(100 * sum(x < DESVIACION_UMBRAL_DEG for x in d) / len(d), 1)]
            else:
                fila += [num(float(np.mean(np.abs(ex))), 3)]
            t4.append('| ' + ' | '.join(fila) + ' |')
 
    texto = '\n'.join(['Tabla 3', *t3, '', 'Tabla 4', *t4])
    print(texto)
    with open(os.path.join(DIR_SALIDA, 'resumen_planB.md'), 'w', encoding='utf-8') as f:
        f.write(texto + '\n')
 
 
def medir_campo(args):
    """Gira el seguidor en el sitio con la pelota a la vista y relaciona su
    rumbo real con la x del blob. Como el guía se aleja casi en línea recta,
    su dirección apenas cambia durante la medida."""
    from follow_city_env import FollowCityEnv
    env = FollowCityEnv(usar_demostracion_inicial=False)
    pose = PoseSeguidor(env.nodo, args.ns_seguidor)
    muestras = []
    try:
        env.preparar_calibracion()
        pose.esperar()
        for w, duracion in [(0.3, 1.0), (-0.3, 2.0), (0.3, 1.0)]:
            fin = time.monotonic() + duracion
            siguiente = 0.0
            while time.monotonic() < fin:
                if time.monotonic() >= siguiente:
                    env._publicar_velocidad(0.0, w)
                    siguiente = time.monotonic() + 0.1
                antes = env._n_blobs
                rclpy.spin_once(env.nodo, timeout_sec=0.02)
                if env._n_blobs != antes:
                    blob = env._blob_verde()
                    if blob is not None:
                        muestras.append((pose.rumbo, error_x_de(blob)))
        env._detener()
    finally:
        env.close()
    if len(muestras) < 5:
        print('Solo {} muestras con la pelota visible; repetid la medida.'
              .format(len(muestras)))
        return
    rumbos = np.degrees(np.unwrap(np.radians([m[0] for m in muestras])))
    pendiente = np.polyfit(rumbos, [m[1] for m in muestras], 1)[0]
    print('{} muestras. Pendiente: {:.4f} unidades de x por grado.'
          .format(len(muestras), pendiente))
    print('Semicampo de visión aproximado: {:.1f}°. Ponedlo en '
          'SEMICAMPO_VISION_DEG.'.format(1.0 / abs(pendiente)))
 
 
def main():
    p = argparse.ArgumentParser(description='Evaluador del Anexo B (plan B).')
    p.add_argument('--medir-campo', action='store_true')
    p.add_argument('--resumir', action='store_true')
    p.add_argument('--config', choices=CONFIGURACIONES)
    p.add_argument('--politica', choices=['sac', 'ppo', 'aleatoria'])
    p.add_argument('--modelo', default='p1_follow_city_sac.zip')
    p.add_argument('--etiqueta', default=None)
    p.add_argument('--episodios', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--ns-seguidor', default='/robobo/robot_0/sim')
    args = p.parse_args()
    if args.resumir:
        resumir()
    elif args.medir_campo:
        medir_campo(args)
    else:
        if not args.config or not args.politica:
            p.error('para evaluar hacen falta --config y --politica')
        evaluar(args)
 
 
if __name__ == '__main__':
    main()
 