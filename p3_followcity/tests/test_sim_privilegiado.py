"""Pruebas sin ROS de las medidas privilegiadas del simulador.

Se sustituyen los módulos de ROS por módulos vacíos: solo se prueba la
geometría (lateral respecto al rastro del guía) y la detección de paradas.

    python3 tests/test_sim_privilegiado.py
"""

import sys
import types
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _m in ('rclpy', 'robobo_ros2_interfaces', 'robobo_ros2_interfaces.msg',
           'robobo_ros2_interfaces.srv'):
    sys.modules.setdefault(_m, types.ModuleType(_m))
sys.modules['robobo_ros2_interfaces.msg'].RobotLocation = object
sys.modules['robobo_ros2_interfaces.msg'].SimObjectArray = object
sys.modules['robobo_ros2_interfaces.srv'].SetRobotLocation = object

import sim_privilegiado  # noqa: E402
from sim_privilegiado import SimPrivilegiado  # noqa: E402


class Cfg:
    distancia_max_rastro_mm = 400.0
    ventana_velocidad_s = 2.0
    v_guia_parado_m_s = 0.005
    parada_min_s = 1.5


def sim_con_rastro(puntos):
    s = SimPrivilegiado.__new__(SimPrivilegiado)
    s.cfg = Cfg()
    s.rastro = deque(puntos)
    s._hist_guia = deque()
    s._desde_lento = None
    return s


# Rastro: recta hacia +z y curva hacia -x (en RoboboSim, +x es la izquierda del avance).
RASTRO = [(0, z) for z in range(0, 1000, 20)] + [(-x, 980 + x) for x in range(20, 400, 20)]


def test_lateral_en_recta_con_signo():
    s = sim_con_rastro(RASTRO)
    s.seg = (50, 500)          # +x = izquierda -> lateral negativo
    assert abs(s.lateral_m() + 0.05) < 1e-9
    s.seg = (-50, 500)
    assert abs(s.lateral_m() - 0.05) < 1e-9


def test_lateral_en_curva():
    s = sim_con_rastro(RASTRO)
    s.seg = (30, 1010)
    assert abs(s.lateral_m() + 0.0424) < 1e-3


def test_lateral_desconocido_fuera_del_rastro():
    s = sim_con_rastro(RASTRO)
    s.seg = (0, -300)          # detrás del inicio del rastro
    assert s.lateral_m() is None
    s.seg = (-500, 1400)       # más allá del final
    assert s.lateral_m() is None
    s.seg = (600, 500)         # demasiado lejos
    assert s.lateral_m() is None


def test_parada_con_histeresis(monkeypatch=None):
    s = sim_con_rastro(RASTRO)
    reloj = {'t': 0.0}
    sim_privilegiado.time.monotonic = lambda: reloj['t']
    try:
        def pose(t, z):
            reloj['t'] = t
            s._hist_guia.append((t, 0.0, z))
            while s._hist_guia and t - s._hist_guia[0][0] > s.cfg.ventana_velocidad_s:
                s._hist_guia.popleft()
        for k in range(30):                     # en marcha a 4 cm/s
            pose(0.1 * k, 40.0 * 0.1 * k)
        assert s.guia_parado() == (False, s.velocidad_guia_m_s())
        z = 40.0 * 2.9
        for k in range(30, 80):                 # parado 5 s
            pose(0.1 * k, z)
            parado, _ = s.guia_parado()
            if 0.1 * k < 2.9 + s.cfg.parada_min_s:
                assert not parado, 0.1 * k      # nunca antes de la histéresis
            if 0.1 * k > 2.9 + s.cfg.ventana_velocidad_s + s.cfg.parada_min_s + 0.15:
                assert parado, 0.1 * k          # como mucho ventana + histéresis
        assert s.guia_parado()[0] is True
    finally:
        import time as _time
        sim_privilegiado.time.monotonic = _time.monotonic


if __name__ == '__main__':
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for prueba in pruebas:
        prueba()
        print('ok', prueba.__name__)
    print('{} pruebas superadas'.format(len(pruebas)))
