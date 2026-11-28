"""Recompensa, condiciones de fin y métricas de recorrido (sin ROS).

Separado del entorno para poder probarlo sin simulador y para que la misma
lógica de métricas sirva en simulación y en el robot real.

Fuentes de información
----------------------
- Percepción (siempre disponible, sim y real): observación del guía y del
  carril, infrarrojos y la acción enviada.
- Verdad del simulador (``privilegiada``, solo en simulación): distancia real
  al guía, desplazamiento lateral real respecto a la trayectoria del guía
  (que circula por el centro del carril), si el guía está parado.

La política nunca recibe la información privilegiada. Se usa (si
``recompensa.usar_privilegiada`` es true) para dar una recompensa más
fiel durante el entrenamiento en simulación y, siempre que exista, para las
métricas de simulación. En el robot real las métricas se estiman con la
percepción (columnas ``*_estimada``).
"""

import math
from dataclasses import dataclass, field


def _recortar(v, a, b):
    return max(a, min(b, v))


@dataclass
class Privilegiada:
    """Verdad del simulador en un paso (todo None si no hay simulador)."""
    distancia_m: float = None          # separación del anexo B: centro del seguidor al punto
                                       # de referencia (0,08 m detrás del centro del guía)
    desviacion_deg: float = None       # rumbo del seguidor vs dirección al punto de referencia
    lateral_m: float = None            # > 0 seguidor a la derecha de la trayectoria del guía
    guia_parado: bool = None
    guia_velocidad_m_s: float = None


@dataclass
class Paso:
    """Lo necesario de un paso para recompensa y métricas."""
    t: float
    guia_visible: float
    guia_tamano: float
    guia_tiempo_perdido_s: float
    carril_valido: float
    carril_validez: float
    carril_desplazamiento: float
    carril_tiempo_sin_s: float
    ir_frontal: float                  # [0, 1]
    v: float                           # m/s enviada
    w: float                           # rad/s enviada
    dw: float                          # cambio de w respecto al paso anterior (rad/s)
    priv: Privilegiada = field(default_factory=Privilegiada)


class Evaluador:
    """Calcula recompensa y fin de episodio, y acumula las métricas del recorrido."""

    def __init__(self, perfil):
        self.r = perfil.recompensa
        self.m = perfil.metricas
        self.s = perfil.seguridad
        self.p = perfil.politica
        self.g = perfil.guia
        self.reiniciar()

    def reiniciar(self):
        self.pasos = 0
        self.salidas_pasos = 0              # pasos fuera del carril (verdad si existe)
        self.salidas_pasos_estimadas = 0    # pasos fuera del carril según percepción
        self.salidas_eventos = 0            # número de salidas (transiciones dentro -> fuera)
        self._fuera_anterior = False
        self.contactos = 0
        self._en_contacto = False
        self.paradas_guia = 0
        self.paradas_superadas = 0
        self._parada = None                 # estado de la parada en curso
        self.paradas_estimadas = 0
        self.paradas_superadas_estimadas = 0
        self._parada_est = None
        self.motivo_fin = ''
        self.separaciones = []
        self.desviaciones = []

    # ------------------------------------------------------------------
    # Condiciones
    # ------------------------------------------------------------------
    def fuera_de_carril(self, paso):
        """(verdad o None, estimada por percepción)."""
        verdad = None
        if paso.priv.lateral_m is not None:
            verdad = abs(paso.priv.lateral_m) > self.m.semiancho_carril_m
        estimada = (paso.carril_valido < 0.5 or
                    abs(paso.carril_desplazamiento) > self.m.desplazamiento_salida)
        return verdad, estimada

    def contacto(self, paso):
        if paso.ir_frontal >= self.s.ir_umbral_contacto:
            return True
        d = paso.priv.distancia_m
        return d is not None and d < self.m.distancia_contacto_m

    def parada_operativa(self, paso):
        """Motivo de la parada operativa ('' si no procede). Igual en sim y en real."""
        if self.contacto(paso):
            return 'contacto'
        if paso.guia_tiempo_perdido_s >= self.s.tiempo_max_sin_guia_s:
            return 'guia_perdido'
        if paso.carril_tiempo_sin_s >= self.s.tiempo_max_sin_carril_s:
            return 'carril_perdido'
        return ''

    # ------------------------------------------------------------------
    # Recompensa
    # ------------------------------------------------------------------
    def recompensa(self, paso, motivo_fin):
        r, R = 0.0, self.r
        priv = paso.priv if R.usar_privilegiada else Privilegiada()
        desglose = {}

        # Separación: campana alrededor de la separación objetivo.
        if priv.distancia_m is not None:
            err = (priv.distancia_m - self.m.separacion_objetivo_m) / R.sigma_separacion_m
            desglose['separacion'] = R.peso_separacion * math.exp(-err * err)
        elif paso.guia_visible:
            g = paso.guia_tamano - self.g.tamano_objetivo_norm
            desglose['separacion'] = R.peso_separacion * math.exp(-(g / R.sigma_tamano) ** 2)
        else:
            desglose['separacion'] = 0.0
        if not paso.guia_visible:
            desglose['guia_no_visible'] = -R.penalizacion_guia_no_visible

        # Carril: centrado (con verdad si existe; si no, con la percepción).
        if priv.lateral_m is not None:
            frac = abs(priv.lateral_m) / self.m.semiancho_carril_m
            desglose['carril'] = R.peso_carril * (1.0 - _recortar(frac, 0.0, 1.0))
            fuera = abs(priv.lateral_m) > self.m.semiancho_carril_m
        else:
            frac = abs(paso.carril_desplazamiento) / self.m.desplazamiento_salida
            desglose['carril'] = R.peso_carril * paso.carril_validez * (1.0 - _recortar(frac, 0.0, 1.0))
            fuera = self.fuera_de_carril(paso)[1]
        if fuera:
            desglose['fuera_carril'] = -R.penalizacion_fuera_carril

        # Suavidad del giro.
        desglose['suavidad'] = -R.peso_suavidad * abs(paso.dw) / self.p.dw_max_rad_s

        # Con el guía parado y cerca, premiar estar quieto (aprender a detenerse).
        if priv.guia_parado and priv.distancia_m is not None and \
                priv.distancia_m < self.m.separacion_objetivo_m + R.sigma_separacion_m:
            desglose['parado_con_guia'] = R.peso_parar * (1.0 - paso.v / max(self.p.v_max_m_s, 1e-6))

        if motivo_fin:
            desglose['fin'] = -R.penalizacion_fin
        r = sum(desglose.values())
        return r, desglose

    # ------------------------------------------------------------------
    # Métricas
    # ------------------------------------------------------------------
    def actualizar_metricas(self, paso):
        self.pasos += 1
        verdad, estimada = self.fuera_de_carril(paso)
        fuera = verdad if verdad is not None else estimada
        self.salidas_pasos += int(bool(fuera))
        self.salidas_pasos_estimadas += int(bool(estimada))
        if fuera and not self._fuera_anterior:
            self.salidas_eventos += 1
        self._fuera_anterior = bool(fuera)

        c = self.contacto(paso)
        if c and not self._en_contacto:
            self.contactos += 1
        self._en_contacto = c
        if paso.priv.distancia_m is not None:
            self.separaciones.append(paso.priv.distancia_m)
        if paso.priv.desviacion_deg is not None:
            self.desviaciones.append(abs(paso.priv.desviacion_deg))

        self._paradas_verdad(paso)
        self._paradas_estimadas(paso)

    def _paradas_verdad(self, paso):
        """Paradas del guía con la verdad del simulador (anexo B de la P2).

        Parada: el guía va a menos de ``v_guia_parado_m_s`` durante más de
        ``parada_min_s`` (lo decide SimPrivilegiado). Superada: no hay contacto
        mientras dura y el seguidor sigue sin fallo ``tiempo_sin_fallo_tras_reanudar_s``
        después de que el guía reanude. Si el episodio termina antes (contacto,
        guía perdido...), la parada queda sin superar.
        """
        parado = paso.priv.guia_parado
        if parado is None:
            return
        st = self._parada
        if parado and st is None:
            self._parada = {'inicio': paso.t, 'contacto': False, 'reanudada': None}
            self.paradas_guia += 1
            return
        if st is None:
            return
        st['contacto'] |= self.contacto(paso)
        if parado and st['reanudada'] is not None:
            st['reanudada'] = None          # se vuelve a parar antes de cumplir el plazo: misma parada
        elif not parado and st['reanudada'] is None:
            st['reanudada'] = paso.t
        if st['contacto']:
            if st['reanudada'] is not None:
                self._parada = None         # no superada; la próxima parada se cuenta aparte
            return
        if st['reanudada'] is not None and \
                paso.t - st['reanudada'] >= self.m.tiempo_sin_fallo_tras_reanudar_s:
            self.paradas_superadas += 1
            self._parada = None

    def _paradas_estimadas(self, paso):
        """Estimación sin verdad (robot real): el seguidor se detiene con el guía
        visible y cerca durante >= tiempo_parada_min_s y después reanuda."""
        lento = paso.v < self.m.v_parado_m_s
        cerca = paso.guia_visible and paso.guia_tamano >= self.m.tamano_cerca_parada
        st = self._parada_est
        if st is None:
            if lento and cerca:
                self._parada_est = {'inicio': paso.t, 'contacto': False, 'contada': False}
        else:
            st['contacto'] |= self.contacto(paso)
            if lento and not st['contada'] and paso.t - st['inicio'] >= self.m.tiempo_parada_min_s:
                st['contada'] = True
                self.paradas_estimadas += 1
            if not lento:
                if st['contada'] and not st['contacto'] and paso.v >= 0.5 * self.p.v_max_m_s:
                    self.paradas_superadas_estimadas += 1
                if paso.v >= 0.5 * self.p.v_max_m_s or not st['contada']:
                    self._parada_est = None

    def resumen(self, completado):
        """Métricas del recorrido (una fila del CSV de episodios)."""
        n = max(self.pasos, 1)
        sep, dv = self.separaciones, self.desviaciones
        return {
            'pasos': self.pasos,
            'completado_sin_intervencion': int(bool(completado)),
            'motivo_fin': self.motivo_fin or ('completado' if completado else ''),
            'salidas_carril_pct_pasos': round(100.0 * self.salidas_pasos / n, 2),
            'salidas_carril_eventos': self.salidas_eventos,
            'salidas_carril_pct_pasos_estimada': round(100.0 * self.salidas_pasos_estimadas / n, 2),
            'contactos': self.contactos,
            'paradas_guia': self.paradas_guia,
            'paradas_superadas': self.paradas_superadas,
            'paradas_estimadas': self.paradas_estimadas,
            'paradas_superadas_estimadas': self.paradas_superadas_estimadas,
            'separacion_en_banda_pct_pasos': round(100.0 * sum(
                self.m.banda_objetivo_m[0] <= d <= self.m.banda_objetivo_m[1] for d in sep) / len(sep), 2)
            if sep else None,
            'separacion_media_m': round(sum(sep) / len(sep), 4) if sep else None,
            'desviacion_media_abs_deg': round(sum(dv) / len(dv), 2) if dv else None,
            'desviacion_menor_15_pct_pasos': round(100.0 * sum(d < 15.0 for d in dv) / len(dv), 2)
            if dv else None,
            'separacion_min_m': round(min(sep), 4) if sep else None,
        }
