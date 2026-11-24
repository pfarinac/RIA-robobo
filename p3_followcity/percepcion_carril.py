"""Percepción del carril a partir de ``camera/image_raw`` (enunciado, 2.1).

Método clásico con OpenCV, elegido frente a una red porque cada paso se puede
inspeccionar, se ajusta desde el perfil y se recalibra en el robot real con
unas pocas imágenes, sin reentrenar nada.

Modelo geométrico
-----------------
La cámara va a altura fija y con inclinación (tilt) fija. Para un punto del
suelo a distancia Z y desplazamiento lateral X respecto al robot, con el
horizonte en la fila y_h, el eje óptico en la columna c_x y focal f:

    y - y_h = f·h / Z          x - c_x = f·X / Z = (X / h)·(y - y_h)

Las dos líneas del carril son rectas paralelas en el suelo, separadas un
ancho de carril W. Si el centro del carril está en X_c(Z) = c0 + Z·tan(θ),
con θ la orientación del carril respecto al robot, cada línea cumple en la
imagen:

    x = A + B·(y - y_h) + s·(k/2)·(y - y_h),   s = -1 izquierda, +1 derecha

    A = c_x + f·tan(θ)        (columna del punto de fuga)
    B = k·c0 / W              (inclinación del centro: desplazamiento lateral)
    k = W / h                 (ancho del carril en píxeles por píxel bajo el horizonte)

Con y_h, k, c_x y f calibrados, cada fotograma solo tiene dos incógnitas
(A, B), que se obtienen por mínimos cuadrados con los puntos de UNA o de LAS
DOS líneas. El desplazamiento y la orientación quedan desacoplados: girar el
robot mueve A, desplazarlo cambia B.

Procedimiento por fotograma
---------------------------
1. ROI entre ``roi_y_inicio_frac`` y ``roi_y_fin_frac`` y máscara HSV de las
   líneas (blanco: saturación baja, brillo alto).
2. Barrido de ``n_filas`` filas. En cada una se buscan tramos blancos de al
   menos ``ancho_min_linea_px`` y se asignan a la línea izquierda o derecha
   si caen a menos de ``tolerancia_ancho``·ancho de la posición que predice
   el modelo (A, B) del fotograma anterior. Sin fotograma anterior fiable, se
   buscan parejas de tramos separadas un ancho de carril.
3. Ajuste robusto de (A, B) con rechazo de atípicos (dos pasadas).
4. Salidas normalizadas (``ObservacionCarril``):
   - desplazamiento = -c0/W, en anchos de carril (> 0: robot a la derecha del centro),
   - ángulo al punto adelantado del centro del carril (fila ``fila_adelantada_frac``
     de la ROI), > 0 si está a la derecha: el que usaría un *pure pursuit*,
   - orientación θ del carril respecto al robot,
   - validez en [0, 1]: filas que sostienen el ajuste, lados vistos y residuo.

Este módulo no depende de ROS.
"""

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

COMPONENTES_OBS_CARRIL = ('carril_valido', 'carril_desplazamiento', 'carril_angulo',
                          'carril_tiempo_sin')
LIMITES_OBS_CARRIL = ((0.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0))


@dataclass
class ObservacionCarril:
    """Resultado de procesar un fotograma.

    desplazamiento: anchos de carril, > 0 si el robot está a la DERECHA del centro.
    angulo_rad: ángulo al punto adelantado del centro, > 0 si está a la DERECHA.
    angulo: angulo_rad / escala_angulo_rad, recortado a [-1, 1].
    orientacion_rad: θ, dirección del carril respecto al robot (> 0: hacia la derecha).
    validez: [0, 1]; por debajo de validez_min el carril se considera perdido.
    """
    valido: float = 0.0
    validez: float = 0.0
    desplazamiento: float = 0.0
    angulo_rad: float = 0.0
    angulo: float = 0.0
    orientacion_rad: float = 0.0
    tiempo_sin: float = 1.0
    lados: str = ''
    puntos: int = 0
    residuo_px: float = 0.0
    edad_imagen_s: float = math.inf
    motivo_invalida: str = 'sin_imagen'
    depuracion: dict = field(default_factory=dict, repr=False)

    def vector(self):
        return [self.valido, self.desplazamiento, self.angulo, self.tiempo_sin]

    def registro(self):
        return {
            'carril_valido': self.valido, 'carril_validez': round(self.validez, 4),
            'carril_desplazamiento': round(self.desplazamiento, 4),
            'carril_angulo_rad': round(self.angulo_rad, 4), 'carril_angulo': round(self.angulo, 4),
            'carril_orientacion_rad': round(self.orientacion_rad, 4),
            'carril_tiempo_sin': round(self.tiempo_sin, 4), 'carril_lados': self.lados,
            'carril_puntos': self.puntos, 'carril_residuo_px': round(self.residuo_px, 2),
            'carril_edad_imagen_s': None if math.isinf(self.edad_imagen_s) else round(self.edad_imagen_s, 4),
            'carril_motivo_invalida': self.motivo_invalida,
        }


def _recortar(v, a, b):
    return max(a, min(b, v))


def tramos(fila_binaria, ancho_min):
    """Centros (x) de los tramos de píxeles activos de al menos ancho_min px."""
    d = np.diff(np.concatenate(([0], fila_binaria.astype(np.int8), [0])))
    inicios, fines = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    sel = (fines - inicios) >= ancho_min
    return (inicios[sel] + fines[sel] - 1) / 2.0


def ajustar_modelo(puntos, k, umbral_px, pasadas=2):
    """Ajuste de (A, B) a puntos (dy, x, s) con dy = y - y_h y s = ±1.

    x - s·(k/2)·dy = A + B·dy. Devuelve (A, B, máscara de inliers, residuo medio).
    """
    if len(puntos) < 3:
        return None
    p = np.asarray(puntos, float)
    dy, x, s = p[:, 0], p[:, 1], p[:, 2]
    objetivo = x - s * (k / 2.0) * dy
    M = np.column_stack([np.ones_like(dy), dy])
    dentro = np.ones(len(p), bool)
    for _ in range(pasadas + 1):
        if dentro.sum() < 3:
            return None
        (A, B), *_ = np.linalg.lstsq(M[dentro], objetivo[dentro], rcond=None)
        residuo = np.abs(M @ (A, B) - objetivo)
        nuevo = residuo <= umbral_px
        if (nuevo == dentro).all():
            break
        dentro = nuevo
    if dentro.sum() < 3:
        return None
    return float(A), float(B), dentro, float(residuo[dentro].mean())


class PercepcionCarril:
    """Detector del carril y estado temporal (tiempo sin carril válido)."""

    def __init__(self, config_carril, config_seguridad=None):
        leer = config_carril.__getitem__
        self.roi_y0 = float(leer('roi_y_inicio_frac'))
        self.roi_y1 = float(leer('roi_y_fin_frac'))
        self.hsv_min = np.array(leer('hsv_linea_min'), np.uint8)
        self.hsv_max = np.array(leer('hsv_linea_max'), np.uint8)
        self.ancho_min_linea = int(leer('ancho_min_linea_px'))
        self.n_filas = int(leer('n_filas'))
        self.fila_lejos = float(leer('fila_adelantada_frac'))
        self.horizonte = float(leer('horizonte_y_frac'))
        self.k = float(leer('ancho_carril_por_px'))
        self.cx = float(leer('centro_optico_x_frac'))
        self.focal_frac = float(leer('focal_frac'))
        self.tol = float(leer('tolerancia_ancho'))
        self.escala_desp = float(leer('escala_desplazamiento'))
        self.escala_ang = float(leer('escala_angulo_rad'))
        self.validez_min = float(leer('validez_min'))
        self.tiempo_max_sin = (float(config_seguridad['tiempo_max_sin_carril_s'])
                               if config_seguridad else 2.0)
        self.reiniciar()

    # ------------------------------------------------------------------
    def reiniciar(self, t=None):
        self._t_inicio = time.monotonic() if t is None else t
        self._t_ultimo_valido = None
        self._modelo_previo = None       # (A, B) del último fotograma válido
        # Horizonte y ancho por píxel efectivos: parten de la calibración y se
        # corrigen con los fotogramas en que se ven las dos líneas (cabeceo del
        # robot al acelerar, holgura del tilt).
        self.horizonte_ef = self.horizonte
        self.k_ef = self.k
        self.ultima = ObservacionCarril()

    def mascara(self, imagen_bgr):
        alto = imagen_bgr.shape[0]
        y0, y1 = int(self.roi_y0 * alto), int(self.roi_y1 * alto)
        hsv = cv2.cvtColor(imagen_bgr[y0:y1], cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.hsv_min, self.hsv_max)
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), y0

    def _seleccionar(self, filas, ancho, alto, modelo):
        """Asigna tramos a las líneas. filas: lista de (y, xs)."""
        yh, k = self.horizonte_ef * alto, self.k_ef
        puntos = []
        for y, xs in filas:
            dy = y - yh
            if dy <= 0 or len(xs) == 0:
                continue
            w = k * dy
            if modelo is not None:
                A, B = modelo
                c = A + B * dy
                for s in (-1, 1):
                    esperado = c + s * w / 2.0
                    cand = xs[np.abs(xs - esperado) <= self.tol * w]
                    if len(cand):
                        puntos.append((dy, float(cand[np.argmin(np.abs(cand - esperado))]), s))
        return puntos

    def buscar_modelo(self, mascara, y0, ancho, alto, previo=None):
        """Búsqueda tipo Hough de (A, B) cuando no hay modelo previo fiable.

        Recorre una rejilla de hipótesis (punto de fuga A, inclinación del
        centro B) y puntúa cada una con los píxeles de línea que hay a lo
        largo de las dos líneas que predice. Funciona aunque solo se vea una
        línea o la discontinua tenga huecos.
        """
        yh, k = self.horizonte_ef * alto, self.k_ef
        ys = np.linspace(0, mascara.shape[0] - 1, 40).astype(int)
        dy = (ys + y0 - yh).astype(float)
        ys, dy = ys[dy > 0], dy[dy > 0]
        if len(ys) == 0:
            return None
        # Tolerancia: se ensancha la máscara horizontalmente.
        tol_px = max(3, int(0.012 * ancho))
        m = cv2.dilate((mascara > 0).astype(np.uint8),
                       np.ones((1, 2 * tol_px + 1), np.uint8)).astype(np.float32)
        cx = self.cx * ancho
        A_val = cx + np.linspace(-0.6, 0.6, 61) * ancho        # punto de fuga
        # El carril propio es el que contiene al robot: su centro está a menos de
        # medio ancho. Limitar B evita confundirlo con el carril (o el arcén) vecino.
        B_val = np.linspace(-0.55, 0.55, 45) * k
        Ag, Bg = np.meshgrid(A_val, B_val, indexing='ij')
        puntuacion = np.zeros(Ag.shape, np.float32)
        lados = []
        for s in (-1, 1):
            xs = Ag[..., None] + (Bg[..., None] + s * k / 2.0) * dy[None, None, :]
            xi = np.round(xs).astype(int)
            dentro = (xi >= 0) & (xi < ancho)
            valores = np.where(dentro, m[ys[None, None, :], np.clip(xi, 0, ancho - 1)], 0.0)
            # Se normaliza por filas visibles para no premiar líneas fuera de imagen,
            # pero exigiendo un mínimo de filas visibles.
            n_vis = dentro.sum(-1)
            lado = valores.sum(-1) / np.maximum(n_vis, 8)
            lados.append(lado)
            puntuacion += lado
        # Preferencia suave por el modelo previo o por el robot centrado.
        A0, B0 = previo if previo is not None else (cx, 0.0)
        puntuacion -= 0.05 * (np.abs(Ag - A0) / ancho + np.abs(Bg - B0) / k)
        i, j = np.unravel_index(np.argmax(puntuacion), puntuacion.shape)
        if lados[0][i, j] + lados[1][i, j] < 0.25:
            return None
        return float(Ag[i, j]), float(Bg[i, j])

    def procesar(self, imagen_bgr, t_imagen=None, t=None):
        """Procesa un fotograma BGR. Devuelve y guarda la ObservacionCarril."""
        ahora = time.monotonic() if t is None else t
        alto, ancho = imagen_bgr.shape[:2]
        mascara, y0 = self.mascara(imagen_bgr)
        indices = np.linspace(0, mascara.shape[0] - 1, self.n_filas).astype(int)
        filas = [(int(i) + y0, tramos(mascara[i] > 0, self.ancho_min_linea)) for i in indices]
        umbral = max(4.0, 0.025 * ancho)

        ajuste, puntos = None, []
        candidatos = []
        if self._modelo_previo is not None:
            candidatos.append(self._modelo_previo)
        hough = self.buscar_modelo(mascara, y0, ancho, alto, self._modelo_previo)
        if hough is not None:
            candidatos.append(hough)
        mejor = None
        for modelo in candidatos:
            pts = self._seleccionar(filas, ancho, alto, modelo)
            aj = ajustar_modelo(pts, self.k_ef, umbral)
            if aj is None:
                continue
            # Segunda selección con el modelo recién ajustado (recupera filas).
            pts2 = self._seleccionar(filas, ancho, alto, aj[:2])
            aj2 = ajustar_modelo(pts2, self.k_ef, umbral)
            if aj2 is not None and aj2[2].sum() >= aj[2].sum():
                aj, pts = aj2, pts2
            if abs(aj[1] / self.k_ef) > 0.65:
                continue  # el ajuste se ha ido al carril vecino
            if mejor is None or aj[2].sum() > mejor[0][2].sum():
                mejor = (aj, pts)
        if mejor is not None:
            ajuste, puntos = mejor

        # Con las dos líneas bien sostenidas, el fotograma mide su propio horizonte
        # y ancho por píxel: se ajusta cada línea por separado y se cortan.
        medido = self._medir_geometria(puntos, ajuste, alto, umbral) if ajuste else None
        if medido is not None:
            ajuste = medido
        obs = ObservacionCarril(depuracion={'y0': y0, 'puntos': puntos, 'ajuste': ajuste,
                                            'yh': self.horizonte_ef * alto, 'k': self.k_ef})
        obs.edad_imagen_s = (ahora - t_imagen) if t_imagen is not None else 0.0
        if ajuste is not None:
            A, B, dentro, residuo = ajuste
            lados = {int(p[2]) for p, d in zip(puntos, dentro) if d}
            n_izq = sum(1 for p, d in zip(puntos, dentro) if d and p[2] < 0)
            n_der = sum(1 for p, d in zip(puntos, dentro) if d and p[2] > 0)
            obs.lados = 'ambos' if lados == {-1, 1} else ('izquierda' if -1 in lados else 'derecha')
            obs.puntos, obs.residuo_px = int(dentro.sum()), residuo
            soporte = (min(n_izq, self.n_filas) + min(n_der, self.n_filas)) / (2.0 * self.n_filas)
            calidad = 1.0 - _recortar(residuo / umbral, 0.0, 1.0) * 0.5
            validez = _recortar(0.3 + 0.7 * soporte / 0.6, 0.0, 1.0) * calidad
            if obs.lados != 'ambos':
                validez = min(validez, 0.7)
            obs.validez = validez

        if ajuste is not None and obs.validez >= self.validez_min:
            A, B = ajuste[0], ajuste[1]
            f = self.focal_frac * ancho
            obs.desplazamiento = _recortar(-self.escala_desp * B / self.k_ef, -1.0, 1.0)
            tan_theta = (A - self.cx * ancho) / f
            obs.orientacion_rad = math.atan(tan_theta)
            # Punto adelantado del centro, en coordenadas del suelo (unidades de h).
            y_l = self.roi_y0 * alto + self.fila_lejos * (self.roi_y1 - self.roi_y0) * alto
            dy_l = y_l - self.horizonte_ef * alto
            x_l = A + B * dy_l
            obs.angulo_rad = math.atan2(x_l - self.cx * ancho, f)
            obs.angulo = _recortar(obs.angulo_rad / self.escala_ang, -1.0, 1.0)
            obs.valido, obs.tiempo_sin, obs.motivo_invalida = 1.0, 0.0, ''
            obs.depuracion['punto_adelantado'] = (x_l, y_l)
            self._modelo_previo = (A, B)
            self._t_ultimo_valido = ahora
        else:
            obs.motivo_invalida = 'sin_lineas' if ajuste is None else 'baja_validez'
            obs.tiempo_sin = _recortar(self.segundos_sin_carril(ahora) / self.tiempo_max_sin, 0.0, 1.0)
            if self.segundos_sin_carril(ahora) > 0.5 * self.tiempo_max_sin:
                self._modelo_previo = None
        self.ultima = obs
        return obs

    def _medir_geometria(self, puntos, ajuste, alto, umbral, n_min=4, alfa=0.3):
        """Si hay >= n_min inliers por lado, mide horizonte y k del fotograma.

        Actualiza horizonte_ef y k_ef (filtro exponencial, acotado a ±15 % de
        la calibración) y devuelve el ajuste (A, B) reexpresado con ellos.
        """
        # Se usan todos los puntos seleccionados (no solo los inliers del modelo
        # con el horizonte anterior, que es justo lo que puede estar mal).
        p = np.asarray(puntos, float)
        if len(p) == 0:
            return None
        yh_px = self.horizonte_ef * alto
        rectas = {}
        for s in (-1, 1):
            q = p[p[:, 2] == s]
            if len(q) < n_min or np.ptp(q[:, 0]) < 0.15 * alto:
                return None
            y, x = q[:, 0] + yh_px, q[:, 1]
            sel = np.ones(len(q), bool)
            for _ in range(3):
                if sel.sum() < n_min:
                    return None
                coef = np.polyfit(y[sel], x[sel], 1)
                nuevo = np.abs(np.polyval(coef, y) - x) <= umbral
                if (nuevo == sel).all():
                    break
                sel = nuevo
            rectas[s] = coef                                       # x = b·y + a (y absoluta)
        (bl, al), (br, ar) = rectas[-1], rectas[1]
        if br - bl <= 1e-6:
            return None
        k_med = br - bl
        y_fuga = (al - ar) / (br - bl)
        if not (0.85 * self.k <= k_med <= 1.15 * self.k and
                abs(y_fuga / alto - self.horizonte) <= 0.15):
            return None
        self.k_ef += alfa * (k_med - self.k_ef)
        self.horizonte_ef += alfa * (y_fuga / alto - self.horizonte_ef)
        # Reexpresar con el horizonte efectivo: A en la fila del horizonte.
        B = (bl + br) / 2.0 * (self.k_ef / k_med)
        A = (al + ar) / 2.0 + (bl + br) / 2.0 * self.horizonte_ef * alto
        return (float(A), float(B)) + tuple(ajuste[2:])

    def segundos_sin_carril(self, t=None):
        ahora = time.monotonic() if t is None else t
        ref = self._t_ultimo_valido if self._t_ultimo_valido is not None else self._t_inicio
        return max(0.0, ahora - ref)

    # ------------------------------------------------------------------
    @staticmethod
    def dibujar(imagen_bgr, obs):
        """Copia de la imagen con la ROI, los puntos, las líneas del modelo y las salidas."""
        img = imagen_bgr.copy()
        d = obs.depuracion
        alto, ancho = img.shape[:2]
        if 'y0' in d:
            cv2.line(img, (0, d['y0']), (ancho, d['y0']), (255, 255, 0), 1)
        yh = d.get('yh', 0)
        for dy, x, s in d.get('puntos', []):
            color = (255, 0, 0) if s < 0 else (0, 165, 255)
            cv2.circle(img, (int(x), int(dy + yh)), 4, color, -1)
        ajuste = d.get('ajuste')
        if ajuste is not None:
            A, B, k = ajuste[0], ajuste[1], d['k']
            for s, color in ((-1, (255, 0, 0)), (1, (0, 165, 255)), (0, (0, 255, 0))):
                ys = np.array([d.get('y0', 0), alto - 1], float)
                xs = A + (B + s * k / 2.0) * (ys - yh)
                cv2.line(img, (int(xs[0]), int(ys[0])), (int(xs[1]), int(ys[1])), color, 2)
        if 'punto_adelantado' in d:
            x, y = d['punto_adelantado']
            cv2.circle(img, (int(x), int(y)), 7, (0, 0, 255), 2)
        texto = 'val={:.2f} desp={:+.2f} ang={:+.1f} or={:+.1f}deg {}'.format(
            obs.validez, obs.desplazamiento, math.degrees(obs.angulo_rad),
            math.degrees(obs.orientacion_rad), obs.lados or obs.motivo_invalida)
        cv2.rectangle(img, (0, 0), (ancho, 22), (0, 0, 0), -1)
        cv2.putText(img, texto, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return img
