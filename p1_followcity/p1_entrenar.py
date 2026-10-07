#!/usr/bin/env python3
"""Calibra, entrena y evalúa el seguimiento de Follow City."""

import argparse
import os
import time

import numpy as np

from follow_city_env import FollowCityEnv
import follow_city_config as config


RUTA_MODELO = 'p1_follow_city_sac'
RUTA_LOGS = 'p1_follow_city_logs'

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


def construir_algoritmo(env):
    from stable_baselines3 import SAC

    return SAC(
        'MlpPolicy', env, verbose=1,
        tensorboard_log=RUTA_LOGS, **HIPER_SAC)


def modo_calibrar():
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
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    class EpisodiosCallback(BaseCallback):
        def __init__(self, episodios_objetivo):
            super().__init__()
            self.episodios_objetivo = episodios_objetivo
            self.episodios_completados = 0
            self.recompensa_actual = 0.0
            self.pasos_actual = 0

        def _on_step(self):
            for indice, (recompensa, terminado) in enumerate(zip(
                    self.locals['rewards'], self.locals['dones'])):
                accion_ejecutada = self.locals['infos'][indice].get(
                    '_accion_ejecutada')
                if accion_ejecutada is not None:
                    self.locals['buffer_actions'][indice] = np.asarray(
                        accion_ejecutada, dtype=np.float32)
                self.recompensa_actual += float(recompensa)
                self.pasos_actual += 1
                if terminado:
                    self.episodios_completados += 1
                    print(
                        'episodio {:3d}: pasos={:3d} recompensa={:8.2f}'
                        .format(
                            self.episodios_completados,
                            self.pasos_actual,
                            self.recompensa_actual),
                        flush=True)
                    self.recompensa_actual = 0.0
                    self.pasos_actual = 0
            return self.episodios_completados < self.episodios_objetivo

    entorno_base = FollowCityEnv(usar_demostracion_inicial=True)
    env = Monitor(entorno_base)
    try:
        modelo = construir_algoritmo(env)
        callback = EpisodiosCallback(args.episodios_entrenamiento)
        pasos_limite = (
            args.episodios_entrenamiento
            * config.MAX_STEPS_PER_EPISODE)
        print(
            'Entrenando SAC hasta completar {} episodios '
            '(límite de seguridad: {} pasos).'
            .format(args.episodios_entrenamiento, pasos_limite))
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

        modelo.save(RUTA_MODELO)
        print('Modelo guardado en {}.zip'.format(RUTA_MODELO))
        print('Episodios completados: {}. Duración: {:.1f} min.'
              .format(
                  callback.episodios_completados,
                  (time.time() - inicio) / 60.0))
        if error_sensor is not None:
            print('Se guardó el modelo parcial para conservar lo aprendido; '
                  'revisa el puente ROS antes de continuar.')
            return
        if callback.episodios_completados:
            entorno_base.usar_demostracion_inicial = False
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
    from stable_baselines3 import SAC

    ruta_zip = RUTA_MODELO + '.zip'
    if not os.path.isfile(ruta_zip):
        raise FileNotFoundError(
            'No existe {}.zip; ejecuta primero --entrenar.'.format(
                RUTA_MODELO))

    env = FollowCityEnv(usar_demostracion_inicial=False)
    try:
        modelo = SAC.load(ruta_zip, env=env)
        resumen(
            env,
            politica=lambda obs: modelo.predict(
                obs, deterministic=True)[0],
            episodios=args.episodios,
            titulo='EVALUACIÓN DE LA POLÍTICA SAC')
    finally:
        env.close()


def resumen(env, politica, episodios, titulo):
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
    parser = argparse.ArgumentParser(
        description='Seguimiento del guía en Follow City con SAC.')
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
    args = parser.parse_args()

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
