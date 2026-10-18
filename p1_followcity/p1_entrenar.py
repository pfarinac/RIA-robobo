#!/usr/bin/env python3

"""Calibra, entrena y evalúa el seguimiento de Follow City.
 
Modos de uso:

  python3 p1_entrenar.py --calibrar

      Muestra en continuo las lecturas de los infrarrojos y del blob.

  python3 p1_entrenar.py --entrenar --algo sac --seed 2026 \

      --episodios-entrenamiento 200

      Entrena durante el número de episodios indicado, guarda el modelo

      en p1_follow_city_<algo>.zip y lo evalúa de forma determinista.

  python3 p1_entrenar.py --evaluar --algo sac --modelo <ruta.zip>

      Evalúa un modelo guardado con la política determinista.
 
Los registros de TensorBoard se guardan en

p1_follow_city_logs/<algo>_seed<semilla>/.

"""
 
import argparse

import os

import time

import random

import numpy as np

import torch

from follow_city_env import FollowCityEnv

import follow_city_config as config

from stable_baselines3.common.utils import set_random_seed
 
 
# Carpeta raíz de los registros de TensorBoard.

RUTA_LOGS = 'p1_follow_city_logs'
 
# Hiperparámetros de SAC. Respecto a los valores por defecto de

# Stable-Baselines3 se reducen el buffer y el lote y se aumenta

# learning_starts, por el coste en tiempo real de cada paso.

HIPER_SAC = dict(

    learning_rate=3e-4,

    buffer_size=50_000,

    learning_starts=250,

    batch_size=128,

    tau=0.005,

    gamma=0.99,

    train_freq=1,

    gradient_steps=1,

)
 
# Hiperparámetros de PPO. n_steps se reduce de 2048 a 512 para que la

# política se actualice con más frecuencia en tiempo real.

HIPER_PPO = dict(

    learning_rate=3e-4,

    n_steps=512,

    batch_size=64,

    n_epochs=10,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.0,

)
 
 
def ruta_modelo(algoritmo):

    """Devuelve una ruta distinta para cada tipo de política."""

    return 'p1_follow_city_{}'.format(algoritmo.lower())
 
 
def fijar_semillas(semilla: int):

    """Fija la semilla en todas las librerías para garantizar reproducibilidad."""

    random.seed(semilla)

    np.random.seed(semilla)

    torch.manual_seed(semilla)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(semilla)

    set_random_seed(semilla)
 
def construir_algoritmo(env, algoritmo="sac", seed=42, tensorboard_log=None):

    """Crea el modelo de SAC o PPO con sus hiperparámetros y semilla."""

    if algoritmo.lower() == "ppo":

        from stable_baselines3 import PPO

        return PPO('MlpPolicy', env, verbose=1,

                   tensorboard_log=tensorboard_log, **HIPER_PPO, seed=seed)

    else:

        from stable_baselines3 import SAC

        return SAC('MlpPolicy', env, verbose=1,

                   tensorboard_log=tensorboard_log, **HIPER_SAC, seed=seed)
 
 
def modo_calibrar():

    """Herramienta de medida del sensor: imprime en continuo los

    infrarrojos frontales, la proximidad y los datos del blob."""

    env = FollowCityEnv()

    try:

        env.preparar_calibracion()

        print('Calibración de Follow City; Ctrl-C para terminar.')

        print('IR FrontL/C/R (crudos), proximidad, visibilidad y blob.')

        while True:

            lectura = env.calibracion()

            ir = lectura['ir_frontal']

            if ir is None:

                print('Esperando lecturas de infrarrojos...')

                continue

            print(

                'IR={:>5} {:>5} {:>5}  prox={:.3f}  visible={}  '

                'ex={:+.2f} ey={:+.2f} tamano={:.3f}'

                .format(

                    ir[0], ir[1], ir[2],

                    lectura['proximidad_frontal'],

                    lectura['blob_visible'],

                    lectura['error_x'], lectura['error_y'],

                    lectura['tamano_norm']))

    except KeyboardInterrupt:

        print('\nCalibración terminada.')

    finally:

        env.close()
 
 
def modo_entrenar(args):

    """Entrena hasta completar el número de episodios pedido, guarda el

    modelo y lo evalúa de forma determinista."""

    from stable_baselines3.common.callbacks import BaseCallback

    from stable_baselines3.common.monitor import Monitor
 
    class EpisodiosCallback(BaseCallback):

        """Cuenta episodios, imprime su resumen y detiene el

        entrenamiento al alcanzar el número de episodios objetivo."""
 
        def __init__(self, episodios_objetivo):

            super().__init__()

            self.episodios_objetivo = episodios_objetivo

            self.episodios_completados = 0

            self.recompensa_actual = 0.0

            self.pasos_actual = 0
 
        def _on_step(self):

            for indice, (recompensa, terminado) in enumerate(zip(

                    self.locals['rewards'], self.locals['dones'])):

                info = self.locals['infos'][indice]

                self.recompensa_actual += float(recompensa)

                self.pasos_actual += 1

                if terminado:

                    self.episodios_completados += 1

                    motivo_fin = info.get('motivo_fin')

                    if motivo_fin is None and info.get(

                            'TimeLimit.truncated', False):

                        motivo_fin = 'duración máxima (truncado)'

                    if motivo_fin is None:

                        motivo_fin = 'sin motivo informado'

                    print(

                        'episodio {:3d}: pasos={:3d} recompensa={:8.2f} '

                        'fin={}'

                        .format(

                            self.episodios_completados,

                            self.pasos_actual,

                            self.recompensa_actual,

                            motivo_fin),

                        flush=True)

                    self.recompensa_actual = 0.0

                    self.pasos_actual = 0

            # Devolver False detiene model.learn().

            return self.episodios_completados < self.episodios_objetivo

    fijar_semillas(args.seed)
 
    # Carpeta de registros propia de cada algoritmo y semilla.

    nombre_exp = f"{args.algo}_seed{args.seed}"

    dir_logs = os.path.join(RUTA_LOGS, nombre_exp)

    os.makedirs(dir_logs, exist_ok=True)

    # La demostración inicial solo se usa con SAC (off-policy).

    entorno_base = FollowCityEnv(

        usar_demostracion_inicial=args.algo.lower() == 'sac')

    # Monitor registra la recompensa y la duración de cada episodio.

    env = Monitor(entorno_base)

    try:

        modelo = construir_algoritmo(

            env,

            algoritmo=args.algo,

            seed=args.seed,

            tensorboard_log=dir_logs)

        callback = EpisodiosCallback(args.episodios_entrenamiento)

        # Límite de pasos de seguridad; el fin real lo marca el callback.

        pasos_limite = (

            args.episodios_entrenamiento

            * config.MAX_STEPS_PER_EPISODE)

        print(

            'Entrenando {} hasta completar {} episodios '

            '(límite de seguridad: {} pasos).'

            .format(args.algo.upper(), args.episodios_entrenamiento,

                    pasos_limite))

        inicio = time.time()

        error_sensor = None

        try:

            modelo.learn(

                total_timesteps=pasos_limite,

                callback=callback,

                progress_bar=False)

        except KeyboardInterrupt:

            print('\nEntrenamiento interrumpido por el usuario.')

        except RuntimeError as error:

            # Si se pierde la comunicación con ROS, se guarda el modelo

            # parcial en lugar de perder lo aprendido.

            mensaje = str(error)

            errores_sensor = (

                'No llegan mensajes de',

                'No llegan lecturas nuevas de',

                'No se ve la pelota verde',

            )

            if not any(fragmento in mensaje for fragmento in errores_sensor):

                raise

            error_sensor = error

            print('\nEntrenamiento interrumpido por pérdida de datos ROS: {}'

                  .format(mensaje))
 
        nombre_modelo = ruta_modelo(args.algo)

        modelo.save(nombre_modelo)

        print('Modelo guardado en {}.zip'.format(nombre_modelo))

        print('Episodios completados: {}. Duración: {:.1f} min.'

              .format(

                  callback.episodios_completados,

                  (time.time() - inicio) / 60.0))

        if error_sensor is not None:

            print('Se guardó el modelo parcial para conservar lo aprendido; '

                  'revisa el puente ROS antes de continuar.')

            return

        if callback.episodios_completados:

            # Evaluación sin demostración y con episodios de la duración

            # de evaluación.

            entorno_base.usar_demostracion_inicial = False

            entorno_base.pasos_max = max(1, int(

                config.EVAL_EPISODE_SECONDS / config.BLOB_PERIOD_S))

            resumen(

                env,

                politica=lambda obs: modelo.predict(

                    obs, deterministic=True)[0],

                episodios=args.episodios_evaluacion,

                titulo='EVALUACIÓN TRAS ENTRENAMIENTO')

        else:

            print('Se omite la evaluación: no terminó ningún episodio.')

    finally:

        env.close()
 
 
def modo_evaluar(args):

    """Carga un modelo guardado y lo evalúa de forma determinista."""

    from stable_baselines3 import SAC, PPO

    fijar_semillas(args.seed)

    ruta_zip = args.modelo or (ruta_modelo(args.algo) + '.zip')

    if not os.path.isfile(ruta_zip):

        raise FileNotFoundError(

            'No existe el archivo de modelo: {}'.format(ruta_zip))
 
    env = FollowCityEnv(

        usar_demostracion_inicial=False,

        episode_seconds=config.EVAL_EPISODE_SECONDS)

    try:

        # Cargamos el modelo adecuado dependiendo del argumento

        if args.algo.lower() == 'ppo':

            modelo = PPO.load(ruta_zip, env=env)

            titulo_eval = 'EVALUACIÓN DE LA POLÍTICA PPO'

        else:

            modelo = SAC.load(ruta_zip, env=env)

            titulo_eval = 'EVALUACIÓN DE LA POLÍTICA SAC'

        resumen(

            env,

            politica=lambda obs: modelo.predict(

                obs, deterministic=True)[0],

            episodios=args.episodios,

            titulo=titulo_eval)

    finally:

        env.close()
 
 
def resumen(env, politica, episodios, titulo):

    """Ejecuta varios episodios con la política dada e imprime la

    recompensa, la duración y el motivo de fin de cada uno, más las

    medias y el recuento de desenlaces."""

    recompensas = []

    longitudes = []

    desenlaces = {}
 
    print('\n' + titulo)

    print('-' * len(titulo))

    for episodio in range(episodios):

        obs, _ = env.reset()

        recompensa_total = 0.0

        pasos = 0

        terminado = False

        truncado = False

        info = {}

        while not (terminado or truncado):

            accion = politica(obs)

            obs, recompensa, terminado, truncado, info = env.step(accion)

            recompensa_total += float(recompensa)

            pasos += 1
 
        motivo = info.get('motivo_fin') or 'fin sin motivo terminal'

        recompensas.append(recompensa_total)

        longitudes.append(pasos)

        desenlaces[motivo] = desenlaces.get(motivo, 0) + 1

        print(

            'episodio {:2d}: pasos={:3d} recompensa={:8.2f} '

            'blob={:.3f} prox={:.3f} fin={}'

            .format(

                episodio + 1, pasos, recompensa_total,

                info.get('tamano_norm', 0.0),

                info.get('proximidad_frontal', 0.0),

                motivo))
 
    print('Recompensa media: {:.2f} (desviación {:.2f})'

          .format(float(np.mean(recompensas)), float(np.std(recompensas))))

    print('Longitud media: {:.1f} pasos'.format(float(np.mean(longitudes))))

    print('Desenlaces: {}'.format(desenlaces))
 
 
def main():

    """Lee los argumentos de la línea de órdenes y lanza el modo elegido."""

    parser = argparse.ArgumentParser(

        description='Seguimiento del guía en Follow City con SAC o PPO.')

    modos = parser.add_mutually_exclusive_group(required=True)

    modos.add_argument(

        '--calibrar', action='store_true',

        help='muestra lecturas del blob y de los IR FrontL/C/R')

    modos.add_argument(

        '--entrenar', action='store_true',

        help='entrena por episodios y evalúa al acabar')

    modos.add_argument(

        '--evaluar', action='store_true',

        help='evalúa el modelo guardado de forma determinista')

    parser.add_argument(

        '--episodios-entrenamiento', type=int,

        default=config.TRAIN_EPISODES,

        help='episodios completos de entrenamiento (por defecto: {})'

        .format(config.TRAIN_EPISODES))

    parser.add_argument(

        '--episodios-evaluacion', type=int,

        default=config.EVAL_EPISODES,

        help='episodios de evaluación al terminar el entrenamiento')

    parser.add_argument(

        '--episodios', type=int,

        default=config.EVAL_EPISODES,

        help='episodios del modo --evaluar')

    parser.add_argument(

        '--algo', type=str, default='sac', choices=['sac', 'ppo'],

        help='algoritmo de aprendizaje por refuerzo a utilizar (sac o ppo)')

    parser.add_argument(

        '--modelo', type=str, default=None,

        help='ruta del archivo .zip que se evaluará; por defecto se usa '

        'p1_follow_city_<algo>.zip')

    parser.add_argument(

        '--seed', type=int, default = 42,

        help='Configuración de la semilla aleatoria para la reproductibilidad (por defecto 42)'

    )

    args = parser.parse_args()
 
 
    # Todos los recuentos de episodios deben ser positivos.

    for nombre in (

            'episodios_entrenamiento',

            'episodios_evaluacion',

            'episodios'):

        if getattr(args, nombre) < 1:

            parser.error('--{} debe ser al menos 1'.format(

                nombre.replace('_', '-')))
 
    if args.calibrar:

        modo_calibrar()

    elif args.entrenar:

        modo_entrenar(args)

    elif args.evaluar:

        modo_evaluar(args)
 
 
if __name__ == '__main__':

    main()
 