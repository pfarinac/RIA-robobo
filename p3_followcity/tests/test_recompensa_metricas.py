"""Pruebas sin ROS de la recompensa, la parada operativa y las métricas.

    python3 tests/test_recompensa_metricas.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import cargar_perfil  # noqa: E402
from recompensa_metricas import Evaluador, Paso, Privilegiada  # noqa: E402

P = cargar_perfil('sim')
PERIODO = P.seguridad.periodo_control_s


def paso(t=0.0, visible=1.0, tam=0.2, perdido=0.0, valido=1.0, validez=0.9, desp=0.0,
         sin_carril=0.0, ir=0.0, v=0.05, w=0.0, dw=0.0, **priv):
    return Paso(t, visible, tam, perdido, valido, validez, desp, sin_carril, ir, v, w, dw,
                Privilegiada(**priv))


def test_recompensa_mayor_en_situacion_ideal():
    e = Evaluador(P)
    ideal, _ = e.recompensa(paso(distancia_m=0.40, lateral_m=0.0), '')
    lejos, _ = e.recompensa(paso(distancia_m=0.70, lateral_m=0.0), '')
    fuera, d = e.recompensa(paso(distancia_m=0.40, lateral_m=0.20), '')
    assert ideal > lejos and ideal > fuera
    assert d['fuera_carril'] < 0


def test_sin_privilegiada_usa_percepcion():
    e = Evaluador(P)
    r_bien, d = e.recompensa(paso(tam=P.guia.tamano_objetivo_norm, desp=0.0), '')
    r_mal, _ = e.recompensa(paso(visible=0.0, tam=0.0, desp=0.6), '')
    assert r_bien > r_mal and 'separacion' in d


def test_fin_penaliza():
    e = Evaluador(P)
    r, d = e.recompensa(paso(distancia_m=0.4, lateral_m=0.0), 'contacto')
    assert d['fin'] == -P.recompensa.penalizacion_fin and r < 0


def test_parar_con_guia_parado_se_premia():
    e = Evaluador(P)
    quieto, _ = e.recompensa(paso(v=0.0, distancia_m=0.38, lateral_m=0.0, guia_parado=True), '')
    moviendo, _ = e.recompensa(paso(v=P.politica.v_max_m_s, distancia_m=0.38, lateral_m=0.0,
                                    guia_parado=True), '')
    assert quieto > moviendo


def test_parada_operativa():
    e = Evaluador(P)
    s = P.seguridad
    assert e.parada_operativa(paso()) == ''
    assert e.parada_operativa(paso(ir=1.0)) == 'contacto'
    assert e.parada_operativa(paso(distancia_m=0.1)) == 'contacto'
    assert e.parada_operativa(paso(perdido=s.tiempo_max_sin_guia_s)) == 'guia_perdido'
    assert e.parada_operativa(paso(sin_carril=s.tiempo_max_sin_carril_s)) == 'carril_perdido'
    # guía cerca tapando las líneas: se tolera hasta el otro límite
    cerca = P.metricas.tamano_cerca_parada + 0.05
    assert e.parada_operativa(paso(tam=cerca, sin_carril=s.tiempo_max_sin_carril_s)) == ''
    assert e.parada_operativa(paso(tam=cerca, sin_carril=s.tiempo_max_sin_carril_guia_cerca_s)) \
        == 'carril_perdido'


def test_acercamiento_da_gradiente_lejos():
    e = Evaluador(P)
    r6, _ = e.recompensa(paso(distancia_m=0.60, lateral_m=0.0), '')
    r8, d = e.recompensa(paso(distancia_m=0.80, lateral_m=0.0), '')
    assert r6 > r8 + 0.1 and d['acercamiento'] > 0


def test_salidas_de_carril_y_contactos():
    e = Evaluador(P)
    laterales = [0.0, 0.2, 0.2, 0.0, 0.2, 0.0]
    for i, lat in enumerate(laterales):
        e.actualizar_metricas(paso(t=i * PERIODO, lateral_m=lat, distancia_m=0.4))
    e.actualizar_metricas(paso(t=7 * PERIODO, lateral_m=0.0, distancia_m=0.1))
    e.actualizar_metricas(paso(t=8 * PERIODO, lateral_m=0.0, distancia_m=0.1))
    r = e.resumen(completado=False)
    assert r['salidas_carril_eventos'] == 2
    assert abs(r['salidas_carril_pct_pasos'] - 100 * 3 / 8) < 0.01
    assert r['contactos'] == 1           # un contacto continuo cuenta una vez


def test_parada_superada_con_verdad():
    e = Evaluador(P)
    t = 0.0
    for _ in range(4):                     # guía en marcha
        e.actualizar_metricas(paso(t=t, distancia_m=0.4, lateral_m=0.0, guia_parado=False)); t += PERIODO
    for _ in range(10):                    # guía parado, seguidor quieto sin contacto
        e.actualizar_metricas(paso(t=t, v=0.0, distancia_m=0.35, lateral_m=0.0, guia_parado=True)); t += PERIODO
    for _ in range(10):                    # guía reanuda: aún no han pasado 10 s
        e.actualizar_metricas(paso(t=t, v=0.08, distancia_m=0.3, lateral_m=0.0, guia_parado=False)); t += PERIODO
    assert e.resumen(completado=False)['paradas_superadas'] == 0
    for _ in range(int(P.metricas.tiempo_sin_fallo_tras_reanudar_s / PERIODO)):
        e.actualizar_metricas(paso(t=t, v=0.08, distancia_m=0.3, lateral_m=0.0, guia_parado=False)); t += PERIODO
    r = e.resumen(completado=True)
    assert r['paradas_guia'] == 1 and r['paradas_superadas'] == 1
    assert r['separacion_en_banda_pct_pasos'] == 100.0


def test_parada_no_superada_si_hay_contacto():
    e = Evaluador(P)
    t = 0.0
    for d in (0.4, 0.3, 0.10, 0.3):        # se para el guía y el seguidor le toca
        e.actualizar_metricas(paso(t=t, v=0.05, distancia_m=d, lateral_m=0.0, guia_parado=True)); t += PERIODO
    for _ in range(int(12 / PERIODO)):
        e.actualizar_metricas(paso(t=t, v=0.08, distancia_m=0.3, lateral_m=0.0, guia_parado=False)); t += PERIODO
    r = e.resumen(completado=False)
    assert r['paradas_guia'] == 1 and r['paradas_superadas'] == 0 and r['contactos'] == 1


def test_parada_estimada_sin_verdad():
    """Robot real: parada estimada por percepción (seguidor quieto con el guía cerca)."""
    e = Evaluador(P)
    t = 0.0
    for _ in range(int(2.0 / PERIODO)):
        e.actualizar_metricas(paso(t=t, v=0.0, tam=P.metricas.tamano_cerca_parada + 0.05)); t += PERIODO
    for _ in range(3):
        e.actualizar_metricas(paso(t=t, v=P.politica.v_max_m_s, tam=0.2)); t += PERIODO
    r = e.resumen(completado=True)
    assert r['paradas_estimadas'] == 1 and r['paradas_superadas_estimadas'] == 1
    assert r['paradas_guia'] == 0 and r['separacion_media_m'] is None


if __name__ == '__main__':
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for prueba in pruebas:
        prueba()
        print('ok', prueba.__name__)
    print('{} pruebas superadas'.format(len(pruebas)))
