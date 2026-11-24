"""Percepción y redescripción del Robobo guía (enunciado, 2.2).

La detección se toma de ``smartphone/detected_object``
(``robobo_ros2_interfaces/DetectedObject``), que publica a 5 Hz el resultado
de ``readDetectedObject``:

    int32 x, y          centro de la caja, en píxeles desde la esquina superior izquierda
    int32 width, height tamaño de la caja, en píxeles
    float32 confidence  confianza de la clase, en [0.5, 1]
    string label        clase detectada
    int64 timestamp     instante de la detección en el teléfono

``readDetectedObject`` devuelve la **última** detección conocida, así que el
puente puede seguir publicando un objeto que ya no está en la imagen. Por
eso la frescura no se decide por la llegada del mensaje, sino por el cambio
de ``timestamp``: una detección solo es nueva si su ``timestamp`` es distinto
del anterior (``guia.usar_timestamp_para_frescura``).

Representación para la política (análoga al blob de P1 y P2), todo
normalizado y con los parámetros del perfil:

    visible            1 si hay una detección nueva, fiable y reciente; si no, 0
    x                  posición horizontal del centro en [-1, 1] (0 = centrada,
                       negativo = guía a la izquierda)
    tamano             tamaño aparente en [0, 1] (crece al acercarse)
    x_ultima           última x conocida (para saber hacia dónde se perdió)
    tiempo_perdido     tiempo sin verlo, normalizado con tiempo_perdido_max_s

Cuando no hay detección válida, ``x`` y ``tamano`` valen 0 (igual que en P1),
``visible`` vale 0 y la política dispone de ``x_ultima`` y
``tiempo_perdido`` para decidir. Una detección se descarta si su etiqueta no
está en ``etiquetas_validas`` (si la lista no está vacía) o si su confianza
es menor que ``confianza_min``.

Este módulo no depende de ROS: recibe diccionarios o mensajes con los
campos anteriores y el instante de llegada, para poder probarlo sin el
simulador y reutilizarlo igual en simulación y en el robot real.
"""

import math
import time
from dataclasses import dataclass, field, asdict

# Nombres de las componentes del vector de observación del guía, en orden.
COMPONENTES_OBS_GUIA = ('guia_visible', 'guia_x', 'guia_tamano',
                        'guia_x_ultima', 'guia_tiempo_perdido')
LIMITES_OBS_GUIA = ((0.0, 1.0), (-1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0))


@dataclass
class DeteccionBruta:
    """Campos de un DetectedObject más el instante local de llegada."""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    label: str
    timestamp: int
    t_llegada: float

    @classmethod
    def desde_mensaje(cls, msg, t_llegada=None):
        """Acepta un mensaje ROS o un diccionario con los mismos campos."""
        leer = (msg.get if isinstance(msg, dict)
                else lambda c, d=None: getattr(msg, c, d))
        return cls(
            x=float(leer('x', 0)), y=float(leer('y', 0)),
            width=float(leer('width', 0)), height=float(leer('height', 0)),
            confidence=float(leer('confidence', 0.0)),
            label=str(leer('label', '')),
            timestamp=int(leer('timestamp', 0) or 0),
            t_llegada=time.monotonic() if t_llegada is None else float(t_llegada))


@dataclass
class ObservacionGuia:
    """Resultado de la percepción del guía en un instante."""
    visible: float = 0.0
    x: float = 0.0
    y: float = 0.0
    tamano: float = 0.0
    x_ultima: float = 0.0
    tiempo_perdido: float = 1.0
    confianza: float = 0.0
    edad_s: float = math.inf
    etiqueta: str = ''
    motivo_invalida: str = 'sin_detecciones'
    bruta: dict = field(default_factory=dict)

    def vector(self):
        """Componentes que recibe la política (orden de COMPONENTES_OBS_GUIA)."""
        return [self.visible, self.x, self.tamano, self.x_ultima,
                self.tiempo_perdido]

    def registro(self):
        """Diccionario plano para los CSV de registro."""
        datos = asdict(self)
        bruta = datos.pop('bruta')
        datos['edad_s'] = None if math.isinf(self.edad_s) else self.edad_s
        for clave, valor in bruta.items():
            datos['bruto_' + clave] = valor
        return {'guia_' + k: v for k, v in datos.items()}


def _recortar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


class PercepcionGuia:
    """Filtra, normaliza y mantiene el estado temporal de la detección del guía.

    Uso:
        percepcion = PercepcionGuia(perfil.guia)
        percepcion.actualizar(msg)          # en cada mensaje detected_object
        obs = percepcion.observar()         # en cada paso de control
    """

    def __init__(self, config_guia):
        """config_guia: sección ``guia`` del perfil (Seccion o dict)."""
        leer = config_guia.__getitem__
        self.etiquetas_validas = {str(e).lower() for e in leer('etiquetas_validas')}
        self.confianza_min = float(leer('confianza_min'))
        self.ancho = float(leer('ancho_imagen_px'))
        self.alto = float(leer('alto_imagen_px'))
        self.metrica_tamano = str(leer('metrica_tamano'))
        self.escala_tamano = float(leer('escala_tamano'))
        self.usar_timestamp = bool(leer('usar_timestamp_para_frescura'))
        self.antiguedad_max = float(leer('antiguedad_max_deteccion_s'))
        self.tiempo_perdido_max = float(leer('tiempo_perdido_max_s'))
        self.alfa = float(leer('suavizado_alfa'))
        self.reiniciar()

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------
    def reiniciar(self, t=None):
        """Olvida las detecciones previas (al comienzo de cada episodio).

        El reloj de pérdida arranca en ``t``: si el guía no aparece, el
        tiempo perdido crece desde el inicio del episodio.
        """
        ahora = time.monotonic() if t is None else t
        self._ultima_valida = None       # DeteccionBruta aceptada más reciente
        self._t_ultima_valida = None     # instante local en que se aceptó
        self._timestamp_anterior = None
        self._x_filtrada = None
        self._tamano_filtrado = None
        self._x_ultima = 0.0
        self._t_inicio = ahora
        self._ultimo_motivo = 'sin_detecciones'
        self.contadores = {'mensajes': 0, 'aceptadas': 0, 'repetidas': 0,
                           'baja_confianza': 0, 'etiqueta': 0,
                           'fuera_de_imagen': 0}

    # ------------------------------------------------------------------
    # Normalización
    # ------------------------------------------------------------------
    def normalizar_x(self, x_px):
        """Píxeles -> [-1, 1], 0 en el centro de la imagen."""
        mitad = self.ancho / 2.0
        return _recortar((x_px - mitad) / mitad, -1.0, 1.0)

    def normalizar_y(self, y_px):
        mitad = self.alto / 2.0
        return _recortar((y_px - mitad) / mitad, -1.0, 1.0)

    def tamano_aparente(self, ancho_px, alto_px):
        """Tamaño aparente en [0, 1] según ``metrica_tamano`` y ``escala_tamano``.

        - raiz_area: sqrt(w·h) / sqrt(W·H). Proporcional a 1/distancia y
          menos sensible que el área a cajas alargadas o recortadas.
        - altura: h / H.
        - ancho:  w / W (se ve afectado cuando el guía está de lado).
        """
        if self.metrica_tamano == 'altura':
            medida = alto_px / self.alto
        elif self.metrica_tamano == 'ancho':
            medida = ancho_px / self.ancho
        else:
            medida = math.sqrt(max(ancho_px, 0.0) * max(alto_px, 0.0)) / math.sqrt(
                self.ancho * self.alto)
        return _recortar(self.escala_tamano * medida, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Entrada de detecciones
    # ------------------------------------------------------------------
    def actualizar(self, msg, t_llegada=None):
        """Procesa un mensaje DetectedObject. Devuelve True si se acepta."""
        det = (msg if isinstance(msg, DeteccionBruta)
               else DeteccionBruta.desde_mensaje(msg, t_llegada))
        self.contadores['mensajes'] += 1

        # 1) Frescura: el puente reenvía la última detección aunque sea vieja.
        if self.usar_timestamp and det.timestamp != 0:
            if det.timestamp == self._timestamp_anterior:
                self.contadores['repetidas'] += 1
                self._ultimo_motivo = 'repetida'
                return False
            self._timestamp_anterior = det.timestamp

        # 2) Clase y confianza.
        if self.etiquetas_validas and det.label.lower() not in self.etiquetas_validas:
            self.contadores['etiqueta'] += 1
            self._ultimo_motivo = 'etiqueta'
            return False
        if det.confidence < self.confianza_min:
            self.contadores['baja_confianza'] += 1
            self._ultimo_motivo = 'baja_confianza'
            return False

        # 3) Coherencia geométrica: avisa si el perfil tiene mal la resolución.
        if (det.x > self.ancho or det.y > self.alto or det.width <= 0
                or det.height <= 0):
            self.contadores['fuera_de_imagen'] += 1
            self._ultimo_motivo = 'fuera_de_imagen'
            return False

        # 4) Aceptada: normalizar y suavizar.
        x = self.normalizar_x(det.x)
        tamano = self.tamano_aparente(det.width, det.height)
        if self._x_filtrada is None or self.alfa >= 1.0:
            self._x_filtrada, self._tamano_filtrado = x, tamano
        else:
            a = self.alfa
            self._x_filtrada = a * x + (1 - a) * self._x_filtrada
            self._tamano_filtrado = a * tamano + (1 - a) * self._tamano_filtrado
        self._ultima_valida = det
        self._t_ultima_valida = det.t_llegada
        self._x_ultima = self._x_filtrada
        self.contadores['aceptadas'] += 1
        self._ultimo_motivo = ''
        return True

    # ------------------------------------------------------------------
    # Salida para la política
    # ------------------------------------------------------------------
    def observar(self, t=None):
        """Devuelve la ObservacionGuia en el instante ``t`` (monotónico)."""
        ahora = time.monotonic() if t is None else t
        det = self._ultima_valida
        if det is None:
            perdido = ahora - self._t_inicio
            return ObservacionGuia(
                visible=0.0, x_ultima=0.0,
                tiempo_perdido=_recortar(perdido / self.tiempo_perdido_max, 0.0, 1.0),
                motivo_invalida=self._ultimo_motivo or 'sin_detecciones')

        edad = ahora - self._t_ultima_valida
        bruta = {'x_px': det.x, 'y_px': det.y, 'ancho_px': det.width,
                 'alto_px': det.height, 'confianza': det.confidence,
                 'timestamp': det.timestamp}
        if edad <= self.antiguedad_max:
            return ObservacionGuia(
                visible=1.0, x=self._x_filtrada, y=self.normalizar_y(det.y),
                tamano=self._tamano_filtrado, x_ultima=self._x_ultima,
                tiempo_perdido=0.0, confianza=det.confidence, edad_s=edad,
                etiqueta=det.label, motivo_invalida='', bruta=bruta)

        # Detección caducada: no visible; el reloj de pérdida cuenta desde
        # que dejó de ser válida.
        perdido = edad - self.antiguedad_max
        return ObservacionGuia(
            visible=0.0, x_ultima=self._x_ultima,
            tiempo_perdido=_recortar(perdido / self.tiempo_perdido_max, 0.0, 1.0),
            confianza=det.confidence, edad_s=edad, etiqueta=det.label,
            motivo_invalida='caducada', bruta=bruta)

    def segundos_sin_guia(self, t=None):
        """Segundos sin una detección válida (para la parada operativa)."""
        ahora = time.monotonic() if t is None else t
        if self._t_ultima_valida is None:
            return ahora - self._t_inicio
        return max(0.0, ahora - self._t_ultima_valida - self.antiguedad_max)
