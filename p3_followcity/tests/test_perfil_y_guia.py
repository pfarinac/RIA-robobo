"""Pruebas sin ROS de los perfiles y de la percepción del guía.

    cd p3_followcity && python3 -m pytest tests -q
    (o, sin pytest: python3 tests/test_perfil_y_guia.py)
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import ErrorPerfil, cargar_perfil  # noqa: E402
from percepcion_guia import PercepcionGuia, COMPONENTES_OBS_GUIA  # noqa: E402


def det(x=240, y=320, w=64, h=48, conf=0.9, label='robobo', ts=1):
    return {'x': x, 'y': y, 'width': w, 'height': h, 'confidence': conf,
            'label': label, 'timestamp': ts}


# ---------------------------------------------------------------- perfiles
def test_perfiles_validos_y_con_unidades():
    for nombre in ('sim', 'real'):
        p = cargar_perfil(nombre)
        assert p.nombre == nombre
        assert p.politica.v_min_m_s == 0.0
        assert 'm/s' in p.politica.unidad('v_max_m_s')
        assert p.topic('topic_objeto') == '/robobo/robot_0/smartphone/detected_object'
    assert cargar_perfil('sim').es_simulacion
    real = cargar_perfil('real')
    assert not real.es_simulacion and real.simulacion is None


def test_mismas_claves_en_sim_y_real():
    sim, real = cargar_perfil('sim').datos, cargar_perfil('real').datos
    for seccion in ('ros', 'camara', 'carril', 'guia', 'politica', 'seguridad'):
        assert set(sim[seccion]) == set(real[seccion]), seccion


def test_perfil_sin_unidad_se_rechaza():
    datos = cargar_perfil('sim').datos
    datos = json.loads(json.dumps(datos))
    datos['guia']['nuevo_parametro'] = 3
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(datos, f)
    try:
        cargar_perfil(f.name)
    except ErrorPerfil as error:
        assert 'nuevo_parametro' in str(error)
    else:
        raise AssertionError('debería rechazar un parámetro sin unidad')


def test_perfil_incoherente_se_rechaza():
    datos = json.loads(json.dumps(cargar_perfil('sim').datos))
    datos['politica']['v_min_m_s'] = 0.5
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(datos, f)
    try:
        cargar_perfil(f.name)
    except ErrorPerfil as error:
        assert 'v_min' in str(error)
    else:
        raise AssertionError('debería rechazar v_min > v_max')


# ---------------------------------------------------------------- guía
def percepcion(**cambios):
    g = cargar_perfil('sim').guia.como_dict()
    g.update(cambios)
    return PercepcionGuia(g)


def test_normalizacion_basica():
    p = percepcion()
    p.reiniciar(0.0)
    assert p.actualizar(det(x=240), t_llegada=0.0)
    o = p.observar(0.1)
    assert o.visible == 1.0 and abs(o.x) < 1e-9 and o.tiempo_perdido == 0.0
    # raiz_area: sqrt(64·48)/sqrt(480·640) = 0.1
    assert abs(o.tamano - 0.1) < 1e-9
    assert len(o.vector()) == len(COMPONENTES_OBS_GUIA)
    p.actualizar(det(x=0, ts=2), 0.2)
    assert p.observar(0.2).x == -1.0
    p.actualizar(det(x=480, ts=3), 0.3)
    assert p.observar(0.3).x == 1.0


def test_metricas_y_escala():
    assert abs(percepcion(metrica_tamano='altura').tamano_aparente(64, 64) - 0.1) < 1e-9
    assert abs(percepcion(metrica_tamano='ancho').tamano_aparente(48, 64) - 0.1) < 1e-9
    assert abs(percepcion(escala_tamano=2.0).tamano_aparente(64, 48) - 0.2) < 1e-9
    assert percepcion(escala_tamano=50.0).tamano_aparente(480, 640) == 1.0


def test_filtros_confianza_y_etiqueta():
    p = percepcion(etiquetas_validas=['Robobo'])
    p.reiniciar(0.0)
    assert not p.actualizar(det(conf=0.4), 0.0)
    assert not p.actualizar(det(label='person', ts=2), 0.0)
    assert p.observar(0.0).visible == 0.0
    assert p.actualizar(det(label='ROBOBO', ts=3), 0.0)
    assert p.contadores['baja_confianza'] == 1 and p.contadores['etiqueta'] == 1


def test_detecciones_repetidas_caducan():
    """El puente reenvía la última detección: con el mismo timestamp no es nueva."""
    p = percepcion(antiguedad_max_deteccion_s=0.6, tiempo_perdido_max_s=3.0)
    p.reiniciar(0.0)
    assert p.actualizar(det(x=360, ts=10), 0.0)
    for k in range(1, 20):           # 5 Hz, mismo timestamp durante ~4 s
        assert not p.actualizar(det(x=360, ts=10), 0.2 * k)
    o = p.observar(0.5)
    assert o.visible == 1.0
    o = p.observar(2.1)               # 2,1 s sin detección nueva
    assert o.visible == 0.0 and o.motivo_invalida == 'caducada'
    assert o.x == 0.0 and o.tamano == 0.0
    assert abs(o.x_ultima - 0.5) < 1e-9
    assert abs(o.tiempo_perdido - 0.5) < 1e-9     # (2,1 - 0,6) / 3
    assert p.observar(10.0).tiempo_perdido == 1.0
    assert abs(p.segundos_sin_guia(2.1) - 1.5) < 1e-9
    # Una detección nueva lo recupera.
    assert p.actualizar(det(x=240, ts=11), 10.0)
    assert p.observar(10.0).visible == 1.0


def test_timestamp_cero_cuenta_como_nueva():
    p = percepcion()
    p.reiniciar(0.0)
    assert p.actualizar(det(ts=0), 0.0)
    assert p.actualizar(det(ts=0), 0.2)


def test_sin_detecciones_desde_el_inicio():
    p = percepcion(tiempo_perdido_max_s=3.0)
    p.reiniciar(100.0)
    o = p.observar(101.5)
    assert o.visible == 0.0 and abs(o.tiempo_perdido - 0.5) < 1e-9
    assert o.motivo_invalida == 'sin_detecciones'


def test_fuera_de_imagen():
    p = percepcion()
    p.reiniciar(0.0)
    assert not p.actualizar(det(x=900), 0.0)
    assert p.contadores['fuera_de_imagen'] == 1


def test_suavizado():
    p = percepcion(suavizado_alfa=0.5)
    p.reiniciar(0.0)
    p.actualizar(det(x=240, ts=1), 0.0)
    p.actualizar(det(x=480, ts=2), 0.2)
    assert abs(p.observar(0.2).x - 0.5) < 1e-9


def test_registro_plano():
    p = percepcion()
    p.reiniciar(0.0)
    p.actualizar(det(), 0.0)
    r = p.observar(0.1).registro()
    assert r['guia_visible'] == 1.0 and r['guia_bruto_x_px'] == 240
    r = percepcion().observar(0.0).registro()
    assert r['guia_edad_s'] is None and not any(
        isinstance(v, float) and math.isinf(v) for v in r.values())


if __name__ == '__main__':
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for prueba in pruebas:
        prueba()
        print('ok', prueba.__name__)
    print('{} pruebas superadas'.format(len(pruebas)))
