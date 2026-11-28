#!/usr/bin/env python3
"""Calibra, entrena y evalúa el seguimiento del guía en la RoboboCity (P3).

Misma estructura que p1_entrenar.py, con estos cambios:
  - Todos los parámetros salen del perfil (--perfil sim|real), no de un
    módulo de configuración.
  - Solo SAC (el algoritmo de la P3). La "demostración inicial" de la P1 se
    sustituye por pasos de un controlador heurístico que se guardan en el
    buffer de repetición antes de aprender (--pasos-demostracion).
  - Puntos de control cada N episodios con el modelo, el buffer de
    repetición y el estado, y --continuar para reanudar un entrenamiento
    interrumpido (los entrenamientos en tiempo real duran horas).
  - Si se corta el entrenamiento (Ctrl+C o pérdida de datos de ROS), se
    guarda el punto de control en lugar de perder lo aprendido.
  - La evaluación calcula las métricas del enunciado (recorridos sin
    intervención, salidas de carril, contactos, paradas superadas) y deja
    pasos.csv y episodios.csv en registros/.

Modos de uso (desde p3_followcity, con el simulador y el puente en marcha):

  python3 p3_entrenar.py --calibrar --perfil sim
      Muestra en continuo la observación (guía, carril, IR) sin mover el robot.

  python3 p3_entrenar.py --entrenar --perfil sim --seed 2026 \\
      --episodios-entrenamiento 150
      Entrena, guarda modelos/p3_sac_seed2026.zip y lo evalúa.

  python3 p3_entrenar.py --continuar modelos/p3_sac_seed2026 \\
      --episodios-entrenamiento 150
      Reanuda desde el último punto de control hasta llegar a 150 episodios.

  python3 p3_entrenar.py --evaluar --perfil sim --episodios 5
      Evalúa el modelo del perfil (politica.ruta_modelo) de forma determinista.
  python3 p3_entrenar.py --evaluar --politica heuristica --episodios 5
      Evalúa el controlador heurístico de referencia (línea base).

TensorBoard: p3_logs/sac_seed<semilla>/  (tensorboard --logdir p3_logs)
Registros:   registros/entrenamiento/, registros/evaluacion/
Modelos:     modelos/p3_sac_seed<semilla>.zip y modelos/p3_sac_seed<semilla>/ (puntos de control)
"""

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

from perfil import anadir_argumento_perfil, cargar_perfil

CARPETA = Path(__file__).resolve().parent
RUTA_LOGS = CARPETA / 'p3_logs'
RUTA_MODELOS = CARPETA / 'modelos'
RUTA_REGISTROS = CARPETA / 'registros'

# Hiperparámetros de SAC. Como en la P1, respecto a Stable-Baselines3 se
# reduce el buffer (cabe de sobra todo el entrenamiento) y el lote, por el
# coste en tiempo real de cada paso. learning_starts se cuenta DESPUÉS de los
# pasos de demostración.
#
# Actualizaciones al final de cada episodio (train_freq=(1, 'episode'),
# gradient_steps=-1: tantas actualizaciones como pasos tuvo el episodio, la
# misma proporción 1:1 que train_freq=1). Con train_freq=1, SB3 hace la
# actualización entre un paso y el siguiente: en el contenedor de Windows eso
# alargó cada paso de 0,26 s a ~0,9 s (medido el 28/11), así que el robot
# recibía órdenes a ~1 Hz y los temporizadores de carril y guía (en tiempo
# real) vencían en 3-4 pasos. Al final del episodio el robot está parado y el
# control en marcha mantiene siempre su periodo, sea cual sea la CPU.
HIPER_SAC = dict(
    learning_rate=3e-4,
    buffer_size=100_000,
    learning_starts=200,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, 'episode'),
    gradient_steps=-1,
    ent_coef='auto',
    policy_kwargs=dict(net_arch=[256, 256]),
)


def ruta_experimento(semilla):
    return RUTA_MODELOS / 'p3_sac_seed{}'.format(semilla)


def fijar_semillas(semilla):
    """Fija la semilla en todas las librerías para la reproducibilidad."""
    import torch
    from stable_baselines3.common.utils import set_random_seed
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)
    set_random_seed(semilla)


IMAGENES_CADA = 0   # lo fija --imagenes
OPCIONES_PUENTE = {}  # lo fijan --ip-puente y --puente-externo


def crear_entorno(perfil, modo, configuracion, carpeta_registros, etiqueta, semilla):
    from follow_city_p3_env import FollowCityP3Env
    env = FollowCityP3Env(perfil, modo=modo, configuracion=configuracion,
                          carpeta_registros=carpeta_registros, etiqueta=etiqueta,
                          semilla=semilla, verbose=False, **OPCIONES_PUENTE)
    env.guardar_imagenes_cada = IMAGENES_CADA
    return env


def ruta_tensorboard(ruta):
    """La ruta de TensorBoard, o None (con aviso) si tensorboard no está instalado."""
    try:
        import tensorboard  # noqa: F401
        return str(ruta)
    except ImportError:
        print('Aviso: tensorboard no está instalado; no se guardarán curvas en {}.'.format(ruta))
        return None


def construir_sac(env, semilla, tensorboard_log):
    from stable_baselines3 import SAC
    return SAC('MlpPolicy', env, verbose=1, tensorboard_log=ruta_tensorboard(tensorboard_log),
               seed=semilla, **HIPER_SAC)


# ----------------------------------------------------------------------
# Calibrar
# ----------------------------------------------------------------------
def modo_calibrar(args):
    """Imprime en continuo la observación que recibiría la política, sin mover
    el robot: sirve para comprobar guía, carril e IR antes de entrenar."""
    from follow_city_p3_env import NOMBRES_OBS
    perfil = cargar_perfil(args.perfil)
    env = crear_entorno(perfil, 'evaluacion', args.configuracion, None, 'calibrar', args.seed)
    try:
        print('Calibración P3 (perfil {}); Ctrl-C para terminar.'.format(perfil.nombre))
        while True:
            env._spin(perfil.seguridad.periodo_control_s)
            obs, og, oc, ir = env._observar()
            print('guía vis={:.0f} x={:+.2f} tam={:.3f} perd={:.2f} | carril val={:.2f} '
                  'desp={:+.2f} ang={:+.2f} sin={:.2f} {:<9} | IR={:.2f}'.format(
                      og.visible, og.x, og.tamano, og.tiempo_perdido, obs[5], obs[6], obs[7],
                      obs[8], oc.lados or oc.motivo_invalida, ir), flush=True)
    except KeyboardInterrupt:
        print('\nCalibración terminada. Componentes:', ', '.join(NOMBRES_OBS))
    finally:
        env.close()


# ----------------------------------------------------------------------
# Entrenar / continuar
# ----------------------------------------------------------------------
def rellenar_demostracion(modelo, env, perfil, pasos, ruido, rng):
    """Ejecuta el controlador heurístico (con ruido) y guarda sus transiciones
    en el buffer de repetición de SAC, con la acción realmente ejecutada."""
    from controlador_heuristico import ControladorHeuristico
    ctrl = ControladorHeuristico(perfil)
    obs, _ = env.reset()
    hechos, episodios = 0, 0
    print('Demostración heurística: {} pasos al buffer (ruido {:.2f}).'.format(pasos, ruido))
    while hechos < pasos:
        accion = np.clip(ctrl(obs) + rng.normal(0, ruido, 2), -1, 1).astype(np.float32)
        obs2, r, term, trunc, info = env.step(accion)
        # Igual que SB3: done = terminado o truncado, y TimeLimit.truncated para que
        # el buffer no trate el fin por tiempo como estado terminal.
        modelo.replay_buffer.add(obs[None], obs2[None], accion[None], np.array([r]),
                                 np.array([term or trunc]), [{'TimeLimit.truncated': trunc and not term}])
        hechos += 1
        obs = obs2
        if term or trunc:
            episodios += 1
            m = info.get('metricas') or {}
            print('  demo episodio {}: {} pasos, fin={}'.format(
                episodios, m.get('pasos'), m.get('motivo_fin')), flush=True)
            obs, _ = env.reset()
    return hechos


def modo_entrenar(args, continuar=None):
    """Entrena hasta completar el número de episodios pedido, con puntos de
    control; al terminar guarda el modelo y lo evalúa de forma determinista."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    # --- experimento nuevo o reanudado ---
    if continuar:
        dir_exp = Path(continuar)
        dir_exp = dir_exp if dir_exp.is_absolute() else (CARPETA / dir_exp).resolve()
        estado = json.load(open(dir_exp / 'estado.json', encoding='utf-8'))
        args.seed = estado['semilla']
        perfil = cargar_perfil(args.perfil or estado['perfil_ruta'])
        episodios_previos = estado['episodios']
        print('Reanudando {} desde el episodio {} (perfil {}).'.format(
            dir_exp.name, episodios_previos, perfil.nombre))
    else:
        perfil = cargar_perfil(args.perfil)
        dir_exp = ruta_experimento(args.seed)
        if (dir_exp / 'estado.json').exists() and not args.sobrescribir:
            raise SystemExit('Ya existe {}. Usa --continuar {} para reanudarlo o '
                             '--sobrescribir para empezar de cero.'.format(dir_exp, dir_exp))
        episodios_previos = 0
    dir_exp.mkdir(parents=True, exist_ok=True)
    fijar_semillas(args.seed)
    nombre_exp = 'sac_seed{}'.format(args.seed)

    class EpisodiosCallback(BaseCallback):
        """Cuenta episodios, imprime su resumen, guarda puntos de control y
        detiene el entrenamiento al alcanzar el objetivo."""

        def __init__(self, objetivo, previos):
            super().__init__()
            self.objetivo, self.completados = objetivo, previos
            self.recompensa, self.pasos = 0.0, 0
            self.t0 = time.time()
            self.guardado = previos

        def _on_step(self):
            for r, terminado, info in zip(self.locals['rewards'], self.locals['dones'],
                                          self.locals['infos']):
                self.recompensa += float(r)
                self.pasos += 1
                if terminado:
                    self.completados += 1
                    m = info.get('metricas') or {}
                    print('episodio {:4d}: pasos={:4d} recompensa={:8.2f} fin={:<15} '
                          'fuera_carril={:5.1f}% contactos={} paradas={}/{} sep_media={} '
                          '[{:.1f} min]'.format(
                              self.completados, self.pasos, self.recompensa,
                              m.get('motivo_fin', info.get('motivo_fin') or 'tiempo'),
                              m.get('salidas_carril_pct_pasos', 0.0), m.get('contactos', 0),
                              m.get('paradas_superadas', 0), m.get('paradas_guia', 0),
                              m.get('separacion_media_m'), (time.time() - self.t0) / 60),
                          flush=True)
                    for clave in ('salidas_carril_pct_pasos', 'contactos', 'paradas_superadas'):
                        if m.get(clave) is not None:
                            self.logger.record('p3/' + clave, float(m[clave]))
                    self.logger.record('p3/completado', float(m.get('completado_sin_intervencion', 0)))
                    self.recompensa, self.pasos = 0.0, 0
                    if self.completados % args.guardar_cada == 0:
                        guardar_punto_control(self.model, dir_exp, self.completados, perfil, args.seed)
                        self.guardado = self.completados
            return self.completados < self.objetivo

    env_base = crear_entorno(perfil, 'entrenamiento', args.configuracion,
                             RUTA_REGISTROS / 'entrenamiento', nombre_exp, args.seed)
    env = Monitor(env_base)
    try:
        _entrenar(args, continuar, perfil, dir_exp, nombre_exp, episodios_previos,
                  env_base, env, EpisodiosCallback)
    finally:
        env.close()


def _entrenar(args, continuar, perfil, dir_exp, nombre_exp, episodios_previos,
              env_base, env, EpisodiosCallback):
    from stable_baselines3 import SAC
    modelo, callback, error_ros = None, None, None
    try:
        if continuar:
            modelo = SAC.load(dir_exp / 'ultimo.zip', env=env,
                              tensorboard_log=ruta_tensorboard(RUTA_LOGS / nombre_exp))
            # Modelos guardados con train_freq=1: se pasa a actualizar al final del episodio.
            from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
            modelo.train_freq = TrainFreq(1, TrainFrequencyUnit.EPISODE)
            modelo.gradient_steps = -1
            if (dir_exp / 'replay_buffer.pkl').exists():
                modelo.load_replay_buffer(dir_exp / 'replay_buffer.pkl')
                print('Buffer de repetición recuperado: {} transiciones.'.format(
                    modelo.replay_buffer.size()))
        else:
            modelo = construir_sac(env, args.seed, RUTA_LOGS / nombre_exp)
            if args.pasos_demostracion > 0:
                n = rellenar_demostracion(modelo, env_base, perfil, args.pasos_demostracion,
                                          args.ruido_demostracion, np.random.default_rng(args.seed))
                # learning_starts cuenta desde ahora: SAC aprende en cuanto empieza.
                modelo.learning_starts = 0 if n >= HIPER_SAC['batch_size'] else HIPER_SAC['learning_starts']
        callback = EpisodiosCallback(args.episodios_entrenamiento, episodios_previos)
        if callback.completados >= args.episodios_entrenamiento:
            print('Ya hay {} episodios; nada que entrenar.'.format(callback.completados))
        else:
            pasos_ep = env_base.pasos_max
            limite = (args.episodios_entrenamiento - callback.completados) * pasos_ep
            print('Entrenando SAC hasta {} episodios (llevamos {}; límite de seguridad {} pasos, '
                  '~{:.0f} min como máximo).'.format(
                      args.episodios_entrenamiento, callback.completados, limite,
                      limite * perfil.seguridad.periodo_control_s / 60))
            try:
                modelo.learn(total_timesteps=limite, callback=callback, progress_bar=False,
                             reset_num_timesteps=not continuar, tb_log_name='SAC')
            except KeyboardInterrupt:
                print('\nEntrenamiento interrumpido por el usuario.')
            except RuntimeError as error:
                # Pérdida de datos de ROS (cámara, poses...): se guarda lo aprendido.
                error_ros = error
                print('\nEntrenamiento interrumpido por pérdida de datos ROS: {}'.format(error))
    finally:
        if modelo is not None and callback is not None:
            if callback.guardado != callback.completados or not (dir_exp / 'ultimo.zip').exists():
                guardar_punto_control(modelo, dir_exp, callback.completados, perfil, args.seed)
            final = RUTA_MODELOS / 'p3_sac_seed{}.zip'.format(args.seed)
            shutil.copy(dir_exp / 'ultimo.zip', final)
            print('Modelo guardado en {} (episodios: {}).'.format(final, callback.completados))
            print('Para reanudar: python3 p3_entrenar.py --continuar {} --episodios-entrenamiento N'
                  .format(dir_exp.relative_to(CARPETA)))

    if error_ros is None and callback is not None and callback.completados > episodios_previos \
            and args.episodios_evaluacion > 0:
        evaluar(env_base, lambda obs: modelo.predict(obs, deterministic=True)[0],
                args.episodios_evaluacion, 'EVALUACIÓN TRAS ENTRENAMIENTO (SAC seed {})'.format(
                    args.seed), args.configuracion, 'sac_seed{}'.format(args.seed))


def guardar_punto_control(modelo, dir_exp, episodios, perfil, semilla):
    """Guarda modelo, buffer de repetición y estado (para --continuar)."""
    dir_exp.mkdir(parents=True, exist_ok=True)
    modelo.save(dir_exp / 'ultimo.zip')
    modelo.save(dir_exp / 'modelo_ep{:04d}.zip'.format(episodios))
    modelo.save_replay_buffer(dir_exp / 'replay_buffer.pkl')
    json.dump({'semilla': semilla, 'episodios': episodios, 'pasos': int(modelo.num_timesteps),
               'perfil': perfil.nombre, 'perfil_ruta': str(perfil.ruta),
               'fecha': time.strftime('%Y-%m-%d %H:%M:%S'),
               'hiperparametros': {k: str(v) for k, v in HIPER_SAC.items()}},
              open(dir_exp / 'estado.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('  [punto de control: episodio {}, {} pasos -> {}]'.format(
        episodios, modelo.num_timesteps, dir_exp.name), flush=True)


# ----------------------------------------------------------------------
# Evaluar
# ----------------------------------------------------------------------
def modo_evaluar(args):
    """Evalúa un modelo guardado (determinista) o el controlador heurístico."""
    perfil = cargar_perfil(args.perfil)
    if args.politica == 'heuristica':
        from controlador_heuristico import ControladorHeuristico
        politica, etiqueta = ControladorHeuristico(perfil), 'heuristica'
        titulo = 'EVALUACIÓN DEL CONTROLADOR HEURÍSTICO'
    else:
        from stable_baselines3 import SAC
        ruta = Path(args.modelo) if args.modelo else perfil.ruta_relativa(perfil.politica.ruta_modelo)
        if not ruta.is_file():
            raise FileNotFoundError('No existe el modelo {} (usa --modelo).'.format(ruta))
        modelo = SAC.load(ruta)
        politica = lambda obs: modelo.predict(obs, deterministic=True)[0]  # noqa: E731
        etiqueta, titulo = ruta.stem, 'EVALUACIÓN DE {}'.format(ruta.name)
    env = crear_entorno(perfil, 'evaluacion', args.configuracion, None, etiqueta, args.seed)
    try:
        evaluar(env, politica, args.episodios, titulo, args.configuracion, etiqueta)
    finally:
        env.close()


def evaluar(env, politica, episodios, titulo, configuracion, etiqueta):
    """Ejecuta episodios de evaluación e imprime las métricas del enunciado."""
    env.configurar(modo='evaluacion', carpeta_registros=RUTA_REGISTROS / 'evaluacion',
                   etiqueta=etiqueta, configuracion=configuracion)
    print('\n' + titulo + ' · {} · {}'.format(configuracion, env.perfil.nombre))
    print('-' * 70)
    filas = []
    for k in range(episodios):
        obs, _ = env.reset()
        terminado = truncado = False
        recompensa, info = 0.0, {}
        while not (terminado or truncado):
            obs, r, terminado, truncado, info = env.step(politica(obs))
            recompensa += r
        m = dict(info.get('metricas') or {})
        m['recompensa'] = recompensa
        filas.append(m)
        print('recorrido {:2d}: {:<15} pasos={:4d} recompensa={:8.2f} fuera_carril={:5.1f}% '
              '({} salidas) contactos={} paradas superadas={}/{} sep_media={}'.format(
                  k + 1, m.get('motivo_fin'), m.get('pasos', 0), recompensa,
                  m.get('salidas_carril_pct_pasos', 0), m.get('salidas_carril_eventos', 0),
                  m.get('contactos', 0), m.get('paradas_superadas', 0), m.get('paradas_guia', 0),
                  m.get('separacion_media_m')), flush=True)
    n = len(filas)
    suma = lambda c: sum((f.get(c) or 0) for f in filas)  # noqa: E731
    print('-' * 70)
    print('Recorridos completados sin intervención: {}/{}'.format(suma('completado_sin_intervencion'), n))
    print('Salidas de carril: {:.1f} % de los pasos (media), {} salidas en total'.format(
        suma('salidas_carril_pct_pasos') / max(n, 1), suma('salidas_carril_eventos')))
    print('Contactos: {}'.format(suma('contactos')))
    if suma('paradas_guia'):
        print('Paradas del guía superadas: {}/{}'.format(suma('paradas_superadas'), suma('paradas_guia')))
    print('Paradas superadas (estimadas por percepción): {}/{}'.format(
        suma('paradas_superadas_estimadas'), suma('paradas_estimadas')))
    print('Recompensa media: {:.2f} (desviación {:.2f})'.format(
        float(np.mean([f['recompensa'] for f in filas])), float(np.std([f['recompensa'] for f in filas]))))
    if env.registros is not None:
        print('Registros: {}'.format(env.registros.carpeta))
    return filas


# ----------------------------------------------------------------------
def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description='Seguimiento del guía en la RoboboCity con SAC (P3).'))
    modos = parser.add_mutually_exclusive_group(required=True)
    modos.add_argument('--calibrar', action='store_true',
                       help='muestra la observación (guía, carril, IR) sin mover el robot')
    modos.add_argument('--entrenar', action='store_true',
                       help='entrena por episodios y evalúa al acabar')
    modos.add_argument('--continuar', metavar='CARPETA',
                       help='reanuda el entrenamiento de modelos/p3_sac_seed<N>')
    modos.add_argument('--evaluar', action='store_true',
                       help='evalúa un modelo (o el controlador heurístico) de forma determinista')
    parser.add_argument('--episodios-entrenamiento', type=int, default=150,
                        help='episodios de entrenamiento en total (por defecto 150)')
    parser.add_argument('--episodios-evaluacion', type=int, default=5,
                        help='episodios de evaluación al terminar el entrenamiento (0 = ninguno)')
    parser.add_argument('--episodios', type=int, default=5, help='episodios del modo --evaluar')
    parser.add_argument('--seed', type=int, default=2026, help='semilla (por defecto 2026)')
    parser.add_argument('--configuracion', default='C4', choices=['C1', 'C2', 'C3', 'C4'],
                        help='configuración del guía en la escena, para los registros')
    parser.add_argument('--modelo', default=None,
                        help='modelo .zip para --evaluar (por defecto politica.ruta_modelo del perfil)')
    parser.add_argument('--politica', default='sac', choices=['sac', 'heuristica'],
                        help='qué evaluar con --evaluar')
    parser.add_argument('--pasos-demostracion', type=int, default=600,
                        help='pasos del controlador heurístico al buffer antes de aprender (0 = ninguno)')
    parser.add_argument('--ruido-demostracion', type=float, default=0.25,
                        help='ruido gaussiano sobre las acciones de demostración')
    parser.add_argument('--guardar-cada', type=int, default=10,
                        help='episodios entre puntos de control')
    parser.add_argument('--imagenes', type=int, default=0, metavar='N',
                        help='guardar 1 imagen de diagnóstico (carril y guía dibujados) cada N pasos')
    parser.add_argument('--sobrescribir', action='store_true',
                        help='con --entrenar, empezar de cero aunque exista el experimento')
    parser.add_argument('--hilos-torch', type=int, default=1,
                        help='hilos de PyTorch (1 por defecto: el simulador ya usa la CPU y más '
                             'hilos en el contenedor ralentizan las actualizaciones)')
    parser.add_argument('--ip-puente', default=None,
                        help='IP del simulador para el puente que lanza el script '
                             '(por defecto simulacion.puente_ip del perfil: host.docker.internal)')
    parser.add_argument('--puente-externo', action='store_true',
                        help='no lanzar el puente: usar uno ya lanzado a mano (si un reinicio lo '
                             'rompe, el script se detiene)')
    args = parser.parse_args()
    for nombre in ('episodios_entrenamiento', 'episodios', 'guardar_cada'):
        if getattr(args, nombre) < 1:
            parser.error('--{} debe ser al menos 1'.format(nombre.replace('_', '-')))

    os.chdir(CARPETA)
    if args.hilos_torch > 0:
        try:
            import torch
            torch.set_num_threads(args.hilos_torch)
        except ImportError:
            pass
    global IMAGENES_CADA
    IMAGENES_CADA = max(0, args.imagenes)
    OPCIONES_PUENTE.update(ip_puente=args.ip_puente, puente_externo=args.puente_externo)
    if args.calibrar:
        modo_calibrar(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.continuar:
        modo_entrenar(args, continuar=args.continuar)
    elif args.evaluar:
        modo_evaluar(args)


if __name__ == '__main__':
    main()
