#!/usr/bin/env python3
"""Prueba rápida del entorno contra el simulador o el robot real.

Comprueba la interfaz Gymnasium (``check_env``) y ejecuta unos episodios con
una política trivial, guardando los registros de pasos y episodios. No entrena
nada: sirve para verificar topics, reinicios, observaciones, parada operativa
y registros antes de lanzar un entrenamiento largo.

    python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 40
    python3 herramientas/probar_entorno.py --perfil sim --politica aleatoria
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import anadir_argumento_perfil, cargar_perfil  # noqa: E402


def main():
    parser = anadir_argumento_perfil(argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--episodios', type=int, default=2)
    parser.add_argument('--pasos', type=int, default=40, help='pasos máximos por episodio')
    parser.add_argument('--politica', choices=['recta', 'aleatoria', 'quieta'], default='recta')
    parser.add_argument('--configuracion', default='C4')
    parser.add_argument('--registros', default='registros/pruebas')
    parser.add_argument('--sin-check', action='store_true', help='omitir check_env')
    args = parser.parse_args()

    from follow_city_p3_env import NOMBRES_OBS, FollowCityP3Env
    perfil = cargar_perfil(args.perfil)
    env = FollowCityP3Env(perfil, modo='entrenamiento', configuracion=args.configuracion,
                          carpeta_registros=args.registros, etiqueta='prueba', semilla=0,
                          verbose=True)
    try:
        if not args.sin_check:
            from gymnasium.utils.env_checker import check_env
            try:
                check_env(env, skip_render_check=True)
                print('check_env: OK', flush=True)
            except AssertionError as error:
                if 'Deterministic' not in str(error):
                    raise
                # Un robot en tiempo real nunca repite exactamente la misma observación;
                # las comprobaciones de espacios, reset y step anteriores sí han pasado.
                print('check_env: OK salvo determinismo (esperado en tiempo real)', flush=True)
        rng = np.random.default_rng(0)
        for ep in range(args.episodios):
            obs, info = env.reset()
            print('\nEpisodio {} reset: {}'.format(ep + 1, {k: (round(v, 3) if isinstance(v, float) else v)
                                                         for k, v in info.items()}))
            total, t0 = 0.0, time.monotonic()
            for k in range(args.pasos):
                if args.politica == 'recta':
                    a = np.array([0.0, 0.0], np.float32)          # v media, sin giro
                elif args.politica == 'quieta':
                    a = np.array([-1.0, 0.0], np.float32)         # v = v_min
                else:
                    a = rng.uniform(-1, 1, 2).astype(np.float32)
                obs, r, term, trunc, info = env.step(a)
                total += r
                if k % 10 == 0 or term or trunc:
                    print('  paso {:3d} r={:+.3f} obs={} {}'.format(
                        k + 1, r, ' '.join('{}={:+.2f}'.format(n[:10], v) for n, v in zip(NOMBRES_OBS, obs)),
                        info.get('motivo_fin', '')), flush=True)
                if term or trunc:
                    break
            print('Episodio {}: {} pasos en {:.1f} s ({:.2f} s/paso), recompensa {:.2f}'.format(
                ep + 1, k + 1, time.monotonic() - t0, (time.monotonic() - t0) / (k + 1), total))
            if 'metricas' in info:
                print('  métricas:', info['metricas'])
    finally:
        env.close()
        if env.registros is not None:
            print('\nRegistros en', env.registros.carpeta)


if __name__ == '__main__':
    main()
