"""Prueba del ciclo de p3_entrenar.py con un entorno FALSO (sin ROS ni simulador).

Comprueba demostración heurística -> entrenamiento SAC -> puntos de control ->
reanudación con --continuar -> evaluación, en una carpeta temporal. Necesita
stable-baselines3 (está en la imagen Docker); si no está, se omite.

    python3 tests/test_entrenar_falso.py
"""

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import stable_baselines3  # noqa: F401
except ImportError:
    print('stable-baselines3 no instalado: prueba omitida')
    sys.exit(0)

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

import p3_entrenar  # noqa: E402
from perfil import cargar_perfil  # noqa: E402


class EntornoFalso(gym.Env):
    """Mismos espacios e interfaz que FollowCityP3Env, dinámica trivial: premia
    acercar guia_x a 0 con el giro. Episodios de 12 pasos."""

    def __init__(self, perfil, modo, configuracion, *a, **k):
        super().__init__()
        from follow_city_p3_env import OBS_ALTO, OBS_BAJO
        self.perfil, self.modo, self.configuracion = perfil, modo, configuracion
        self.observation_space = spaces.Box(OBS_BAJO, OBS_ALTO, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.pasos_max, self.registros, self.rng = 12, None, np.random.default_rng(0)
        self.reseteos = 0

    def configurar(self, modo=None, carpeta_registros=None, etiqueta='', configuracion=None):
        self.modo = modo or self.modo

    def _obs(self):
        o = np.zeros(12, np.float32)
        o[0], o[1], o[2], o[5] = 1.0, self.gx, 0.2, 0.9
        return o

    def reset(self, seed=None, options=None):
        self.k, self.gx = 0, float(self.rng.uniform(-0.8, 0.8))
        self.reseteos += 1
        return self._obs(), {}

    def step(self, a):
        self.k += 1
        self.gx = float(np.clip(self.gx + 0.3 * a[1], -1, 1))
        r = 1.0 - abs(self.gx)
        trunc = self.k >= self.pasos_max
        info = {'motivo_fin': ''}
        if trunc:
            info['metricas'] = {'pasos': self.k, 'motivo_fin': 'completado', 'completado_sin_intervencion': 1,
                                'salidas_carril_pct_pasos': 0.0, 'salidas_carril_eventos': 0, 'contactos': 0,
                                'paradas_guia': 0, 'paradas_superadas': 0, 'paradas_estimadas': 0,
                                'paradas_superadas_estimadas': 0, 'separacion_media_m': 0.4}
        return self._obs(), r, False, trunc, info

    def close(self):
        pass


def args_base(**cambios):
    a = dict(perfil='sim', seed=7, configuracion='C4', episodios_entrenamiento=3,
             episodios_evaluacion=1, episodios=1, modelo=None, politica='sac',
             pasos_demostracion=40, ruido_demostracion=0.2, guardar_cada=2, sobrescribir=False)
    a.update(cambios)
    return SimpleNamespace(**a)


def test_ciclo_completo():
    tmp = Path(tempfile.mkdtemp())
    originales = (p3_entrenar.RUTA_MODELOS, p3_entrenar.RUTA_LOGS, p3_entrenar.RUTA_REGISTROS,
                  p3_entrenar.crear_entorno, p3_entrenar.CARPETA, dict(p3_entrenar.HIPER_SAC))
    try:
        p3_entrenar.CARPETA = tmp
        p3_entrenar.RUTA_MODELOS, p3_entrenar.RUTA_LOGS = tmp / 'modelos', tmp / 'logs'
        p3_entrenar.RUTA_REGISTROS = tmp / 'registros'
        p3_entrenar.HIPER_SAC.update(batch_size=16, buffer_size=2000, learning_starts=10,
                                     policy_kwargs=dict(net_arch=[32, 32]))
        p3_entrenar.crear_entorno = lambda perfil, modo, conf, *a: EntornoFalso(perfil, modo, conf)

        # 1) Entrenamiento nuevo: 3 episodios, puntos de control cada 2.
        p3_entrenar.modo_entrenar(args_base())
        exp = tmp / 'modelos' / 'p3_sac_seed7'
        import json
        estado = json.load(open(exp / 'estado.json'))
        assert estado['episodios'] == 3, estado
        assert (exp / 'ultimo.zip').exists() and (exp / 'replay_buffer.pkl').exists()
        assert (exp / 'modelo_ep0002.zip').exists()
        assert (tmp / 'modelos' / 'p3_sac_seed7.zip').exists()
        pasos_1 = estado['pasos']

        # 2) Sin --sobrescribir no se pisa un experimento existente.
        try:
            p3_entrenar.modo_entrenar(args_base())
            raise AssertionError('debería negarse a sobrescribir')
        except SystemExit:
            pass

        # 3) Reanudar hasta 5 episodios: conserva contadores y buffer.
        p3_entrenar.modo_entrenar(args_base(perfil=None, episodios_entrenamiento=5,
                                            episodios_evaluacion=0), continuar=str(exp))
        estado = json.load(open(exp / 'estado.json'))
        assert estado['episodios'] == 5 and estado['pasos'] > pasos_1, estado

        # 4) Evaluar el modelo guardado y el heurístico.
        p3_entrenar.modo_evaluar(args_base(modelo=str(tmp / 'modelos' / 'p3_sac_seed7.zip'), episodios=2))
        p3_entrenar.modo_evaluar(args_base(politica='heuristica', episodios=1))
    finally:
        (p3_entrenar.RUTA_MODELOS, p3_entrenar.RUTA_LOGS, p3_entrenar.RUTA_REGISTROS,
         p3_entrenar.crear_entorno, p3_entrenar.CARPETA, hiper) = originales
        p3_entrenar.HIPER_SAC.clear()
        p3_entrenar.HIPER_SAC.update(hiper)
        shutil.rmtree(tmp, ignore_errors=True)


def test_controlador_heuristico():
    from controlador_heuristico import ControladorHeuristico
    perfil = cargar_perfil('sim')
    c = ControladorHeuristico(perfil)
    o = np.zeros(12, np.float32)
    o[0], o[2], o[5] = 1.0, 0.05, 0.9                 # guía lejos, carril válido
    assert c(o)[0] > 0.0                               # acelera (por encima de la v media)
    o[2] = perfil.guia.tamano_demasiado_cerca_norm     # demasiado cerca
    assert c(o)[0] == -1.0                             # v = v_min = 0
    o[2], o[7] = 0.2, 0.5                              # punto adelantado a la derecha
    assert c(o)[1] < 0                                 # gira a la derecha (w < 0)
    o[7], o[1] = 0.0, -0.6                             # guía a la izquierda
    assert c(o)[1] > 0
    o[9] = 1.0                                         # IR: obstáculo delante
    assert c(o)[0] == -1.0


if __name__ == '__main__':
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for prueba in pruebas:
        prueba()
        print('ok', prueba.__name__)
    print('{} pruebas superadas'.format(len(pruebas)))
