"""Pruebas sin ROS del detector de carril con imágenes sintéticas.

Se dibujan las líneas del carril con el mismo modelo geométrico (A, B) que
usa el detector y se comprueba que recupera desplazamiento y orientación.

    python3 tests/test_carril.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import cargar_perfil  # noqa: E402
from percepcion_carril import PercepcionCarril  # noqa: E402

PERFIL = cargar_perfil('sim')
ANCHO, ALTO = 480, 640


def sintetica(desplazamiento, orientacion_deg, discontinua=True, solo=None, ruido=0, d_horizonte=0.0):
    """Imagen BGR con el carril para un desplazamiento (anchos, + derecha) y orientación."""
    c = PERFIL.carril
    yh, k = (c.horizonte_y_frac + d_horizonte) * ALTO, c.ancho_carril_por_px
    f, cx = c.focal_frac * ANCHO, c.centro_optico_x_frac * ANCHO
    A = cx + f * math.tan(math.radians(orientacion_deg))
    B = -desplazamiento * k
    img = np.full((ALTO, ANCHO, 3), (40, 40, 40), np.uint8)
    img[:int(yh)] = (60, 110, 160)        # "tierra" marrón por encima del horizonte
    for s in (-1, 1):
        if solo is not None and s != solo:
            continue
        for y in range(int(yh) + 2, ALTO):
            if s == 1 and discontinua and (y // 25) % 2:
                continue
            dy = y - yh
            x = A + (B + s * k / 2.0) * dy
            grosor = max(2, int(0.06 * dy))
            x0, x1 = int(x - grosor / 2), int(x + grosor / 2)
            if x1 >= 0 and x0 < ANCHO:
                img[y, max(0, x0):min(ANCHO, x1 + 1)] = (235, 235, 235)
    if ruido:
        img = np.clip(img.astype(int) + np.random.default_rng(0).integers(-ruido, ruido, img.shape), 0, 255).astype(np.uint8)
    return img


def detectar(img):
    det = PercepcionCarril(PERFIL.carril, PERFIL.seguridad)
    det.reiniciar(0.0)
    return det.procesar(img, t=0.0)


def test_centrado():
    o = detectar(sintetica(0.0, 0.0))
    assert o.valido and o.lados == 'ambos'
    assert abs(o.desplazamiento) < 0.05 and abs(math.degrees(o.orientacion_rad)) < 2


def test_desplazamientos():
    for d in (-0.3, -0.15, 0.15, 0.3):
        o = detectar(sintetica(d, 0.0))
        assert o.valido, d
        assert abs(o.desplazamiento - d) < 0.06, (d, o.desplazamiento)


def test_orientaciones_desacopladas():
    for th in (-12, -6, 6, 12):
        o = detectar(sintetica(0.0, th))
        assert o.valido, th
        assert abs(math.degrees(o.orientacion_rad) - th) < 2.0, (th, o.orientacion_rad)
        assert abs(o.desplazamiento) < 0.08, (th, o.desplazamiento)


def test_una_sola_linea():
    for solo in (-1, 1):
        o = detectar(sintetica(0.1, 0.0, solo=solo))
        assert o.valido and o.lados in ('izquierda', 'derecha'), solo
        assert abs(o.desplazamiento - 0.1) < 0.08
        assert o.validez <= 0.7


def test_sin_lineas_e_imagen_negra():
    det = PercepcionCarril(PERFIL.carril, PERFIL.seguridad)
    det.reiniciar(0.0)
    o = det.procesar(np.zeros((ALTO, ANCHO, 3), np.uint8), t=1.0)
    assert not o.valido and o.motivo_invalida == 'sin_lineas'
    assert abs(o.tiempo_sin - 1.0 / PERFIL.seguridad.tiempo_max_sin_carril_s) < 1e-9


def test_angulo_pure_pursuit():
    """Con el robot a la izquierda (desp < 0), el centro adelantado queda a la derecha."""
    assert detectar(sintetica(-0.3, 0.0)).angulo_rad > 0
    assert detectar(sintetica(0.3, 0.0)).angulo_rad < 0
    assert detectar(sintetica(0.0, 10.0)).angulo_rad > 0


def test_seguimiento_entre_fotogramas_y_ruido():
    det = PercepcionCarril(PERFIL.carril, PERFIL.seguridad)
    det.reiniciar(0.0)
    for i, d in enumerate(np.linspace(-0.3, 0.3, 13)):
        o = det.procesar(sintetica(d, 0.0, ruido=15), t=0.1 * i)
        assert o.valido and abs(o.desplazamiento - d) < 0.07, (d, o.desplazamiento)


def test_cabeceo_se_compensa():
    """Horizonte desplazado (el robot cabecea): con las dos líneas visibles el
    desplazamiento sigue siendo correcto y el horizonte efectivo converge."""
    det = PercepcionCarril(PERFIL.carril, PERFIL.seguridad)
    det.reiniciar(0.0)
    for i in range(10):
        o = det.procesar(sintetica(0.2, 0.0, d_horizonte=-0.04), t=0.1 * i)
        assert o.valido and o.lados == 'ambos'
        assert abs(o.desplazamiento - 0.2) < 0.06, (i, o.desplazamiento)
    assert abs(det.horizonte_ef - (PERFIL.carril.horizonte_y_frac - 0.04)) < 0.01


def test_guia_blanco_se_excluye():
    """Un robot blanco en el centro del carril se toma por línea si no se excluye."""
    img = sintetica(0.15, 0.0)
    cv2.ellipse(img, (240, 430), (60, 70), 0, 0, 360, (230, 230, 230), -1)   # el "guía"
    det = PercepcionCarril(PERFIL.carril, PERFIL.seguridad)
    det.reiniciar(0.0)
    con = det.procesar(img, t=0.0, excluir=[(240 - 85, 0, 240 + 85, 520)])
    assert con.valido and abs(con.desplazamiento - 0.15) < 0.08, con.desplazamiento


def test_vector_y_registro():
    o = detectar(sintetica(0.1, 5.0))
    assert len(o.vector()) == 4 and all(-1 <= v <= 1 for v in o.vector())
    r = o.registro()
    assert r['carril_valido'] == 1.0 and 'carril_orientacion_rad' in r


if __name__ == '__main__':
    pruebas = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for prueba in pruebas:
        prueba()
        print('ok', prueba.__name__)
    print('{} pruebas superadas'.format(len(pruebas)))
