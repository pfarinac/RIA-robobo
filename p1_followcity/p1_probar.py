#!/usr/bin/env python3
"""Un episodio con salida detallada. Uso: python3 p1_probar.py [aleatorio|p]"""
import sys
import numpy as np
from follow_city_env import FollowCityEnv

modo = sys.argv[1] if len(sys.argv) > 1 else 'p'

env = FollowCityEnv(verbose=True, usar_demostracion_inicial=True)
try:
    obs, _ = env.reset()
    terminado = truncado = False
    total, pasos = 0.0, 0
    while not (terminado or truncado):
        if modo == 'aleatorio':
            accion = env.action_space.sample()
        else:
            # Controlador proporcional: gira hacia donde está el blob.
            # error_x < 0 (blob a la izquierda) -> giro positivo (izquierda).
            ex = obs[1] if obs[0] else obs[4]     # si no se ve, último error_x
            accion = np.array([0.0, np.clip(1.0 * ex, -1, 1)], dtype=np.float32)
        obs, r, terminado, truncado, info = env.step(accion)
        total += r
        pasos += 1
    print('\nFin: {} | pasos={} | recompensa total={:.2f} | media/paso={:.3f}'
          .format(info['motivo_fin'], pasos, total, total / pasos))
finally:
    env.close()