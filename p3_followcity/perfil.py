"""Carga y validación de los perfiles de configuración (enunciado, 2.4).

Todos los valores ajustables de la P3 viven en ficheros JSON dentro de
``perfiles/``: uno para el simulador (``perfil_sim.json``) y otro para el
robot real (``perfil_real.json``). El código nunca contiene constantes de
calibración; las lee de aquí. Cambiar de entorno consiste solo en elegir el
perfil, sin modificar el código.

Selección del perfil, por orden de prioridad:

1. Argumento explícito: ``cargar_perfil('perfiles/perfil_real.json')`` o
   ``cargar_perfil('real')`` (desde la línea de comandos, ``--perfil``).
2. Variable de entorno ``RIA_PERFIL`` (nombre corto o ruta).
3. Por defecto, ``perfil_sim.json``.

Cada sección del JSON lleva un diccionario ``_doc`` que indica, para cada
parámetro, su unidad y su significado. La validación exige que todos los
parámetros estén documentados, para que las unidades nunca falten.
"""

import json
import os
from pathlib import Path

CARPETA_BASE = Path(__file__).resolve().parent
CARPETA_PERFILES = CARPETA_BASE / 'perfiles'
VARIABLE_ENTORNO = 'RIA_PERFIL'
PERFIL_POR_DEFECTO = 'sim'

# Claves mínimas que debe tener cada sección (enunciado, apartado 2.4).
CLAVES_OBLIGATORIAS = {
    'ros': ['ns_robot', 'topic_cmd_vel', 'topic_ir', 'accion_tilt',
            'topic_imagen', 'topic_objeto'],
    'camara': ['tilt_deg', 'resolucion_ancho_px', 'resolucion_alto_px',
               'frecuencia_esperada_hz', 'antiguedad_max_imagen_s'],
    'carril': ['roi_y_inicio_frac', 'roi_y_fin_frac', 'hsv_linea_min',
               'hsv_linea_max', 'ancho_min_linea_px', 'n_filas',
               'fila_adelantada_frac', 'horizonte_y_frac', 'ancho_carril_por_px',
               'centro_optico_x_frac', 'focal_frac', 'tolerancia_ancho',
               'escala_desplazamiento', 'escala_angulo_rad', 'validez_min'],
    'guia': ['etiquetas_validas', 'confianza_min', 'ancho_imagen_px',
             'alto_imagen_px', 'metrica_tamano', 'escala_tamano',
             'usar_timestamp_para_frescura', 'antiguedad_max_deteccion_s',
             'tiempo_perdido_max_s', 'suavizado_alfa'],
    'politica': ['v_min_m_s', 'v_max_m_s', 'w_max_rad_s', 'dw_max_rad_s',
                 'escala_v', 'escala_w', 'ruta_modelo', 'obs_recorte'],
    'seguridad': ['periodo_control_s', 'periodo_reenvio_cmd_vel_s',
                  'tiempo_max_sin_carril_s', 'tiempo_max_sin_guia_s',
                  'ir_saturacion', 'ir_indices_frontales', 'ir_umbral_contacto'],
    'episodio': ['duracion_entrenamiento_s', 'duracion_evaluacion_s',
                 'espera_percepcion_s', 'observaciones_validas_reinicio', 'intentos_reinicio',
                 'timeout_imagen_s', 'espera_real_s'],
    'recompensa': ['usar_privilegiada', 'peso_separacion', 'sigma_separacion_m',
                   'sigma_tamano', 'penalizacion_guia_no_visible', 'peso_carril',
                   'penalizacion_fuera_carril', 'peso_suavidad', 'peso_parar',
                   'penalizacion_fin'],
    'metricas': ['separacion_objetivo_m', 'semiancho_carril_m', 'desplazamiento_salida',
                 'distancia_contacto_m', 'distancia_perdido_m', 'banda_objetivo_m',
                 'tiempo_sin_fallo_tras_reanudar_s', 'v_parado_m_s', 'tamano_cerca_parada',
                 'tiempo_parada_min_s'],
    'aleatorizacion': ['activa', 'ruido_obs_std', 'escala_v_rango', 'escala_w_rango',
                       'prob_perder_guia'],
}

# Claves obligatorias de la sección simulacion (solo perfiles de simulación).
CLAVES_SIMULACION = ['modo_reinicio', 'srv_reset', 'espera_tras_reset_s', 'punto_referencia_detras_m',
                     'gestionar_puente', 'puente_ip', 'puente_modulos', 'puente_timeout_s',
                     'timeout_imagen_tras_reset_s',
                     'topic_objetos', 'topic_pose_seguidor', 'srv_colocar',
                     'id_objeto_guia', 'rastro_max_puntos', 'rastro_paso_mm',
                     'distancia_max_rastro_mm', 'ventana_velocidad_s',
                     'v_guia_parado_m_s', 'parada_min_s', 'altura_robot_mm']

METRICAS_TAMANO = ('raiz_area', 'altura', 'ancho')


class ErrorPerfil(ValueError):
    """El perfil no existe o no cumple la estructura esperada."""


class Seccion:
    """Vista de solo lectura de una sección del perfil.

    Permite ``perfil.guia.confianza_min`` además de
    ``perfil.guia['confianza_min']``, y ``unidad('clave')`` para consultar la
    documentación del parámetro.
    """

    def __init__(self, nombre, datos):
        self._nombre = nombre
        self._datos = {k: v for k, v in datos.items() if k != '_doc'}
        self._doc = dict(datos.get('_doc', {}))

    def __getattr__(self, clave):
        if clave.startswith('_'):
            raise AttributeError(clave)
        try:
            return self._datos[clave]
        except KeyError:
            raise AttributeError(
                "La sección '{}' del perfil no tiene '{}'".format(
                    self._nombre, clave)) from None

    def __getitem__(self, clave):
        return self._datos[clave]

    def __contains__(self, clave):
        return clave in self._datos

    def get(self, clave, defecto=None):
        return self._datos.get(clave, defecto)

    def unidad(self, clave):
        return self._doc.get(clave, '')

    def como_dict(self):
        return dict(self._datos)

    def items(self):
        return self._datos.items()


class Perfil:
    """Perfil cargado: secciones accesibles como atributos."""

    def __init__(self, datos, ruta):
        self.ruta = Path(ruta)
        self.datos = datos
        self.nombre = datos['nombre']
        self.es_simulacion = bool(datos['es_simulacion'])
        self._secciones = {}
        for nombre, contenido in datos.items():
            if isinstance(contenido, dict):
                self._secciones[nombre] = Seccion(nombre, contenido)

    def __getattr__(self, nombre):
        if nombre.startswith('_'):
            raise AttributeError(nombre)
        secciones = self.__dict__.get('_secciones', {})
        if nombre in secciones:
            return secciones[nombre]
        if nombre == 'simulacion':
            return None
        raise AttributeError("El perfil no tiene la sección '{}'".format(nombre))

    def topic(self, clave):
        """Nombre absoluto de un topic/servicio/acción de la sección ros."""
        relativo = self.ros[clave]
        if relativo.startswith('/'):
            return relativo
        return self.ros.ns_robot.rstrip('/') + '/' + relativo

    def ruta_relativa(self, ruta):
        """Resuelve una ruta del perfil respecto a la carpeta p3_followcity."""
        ruta = Path(ruta)
        return ruta if ruta.is_absolute() else CARPETA_BASE / ruta

    def resumen(self):
        """Texto con todos los parámetros, sus valores y sus unidades."""
        lineas = ['Perfil "{}" ({})'.format(self.nombre, self.ruta)]
        for nombre, seccion in self._secciones.items():
            lineas.append('[{}]'.format(nombre))
            for clave, valor in seccion.items():
                lineas.append('  {:<32} = {!r:<24} # {}'.format(
                    clave, valor, seccion.unidad(clave)))
        return '\n'.join(lineas)

    def como_dict_plano(self):
        """Diccionario plano 'seccion.clave' -> valor, para los registros."""
        plano = {'perfil': self.nombre}
        for nombre, seccion in self._secciones.items():
            for clave, valor in seccion.items():
                plano['{}.{}'.format(nombre, clave)] = valor
        return plano


def resolver_ruta(perfil=None):
    """Convierte un nombre corto ('sim', 'real') o una ruta en un Path."""
    if perfil is None:
        perfil = os.environ.get(VARIABLE_ENTORNO, PERFIL_POR_DEFECTO)
    candidato = Path(perfil)
    corto = CARPETA_PERFILES / 'perfil_{}.json'.format(perfil)
    if candidato.suffix == '' and corto.is_file():
        candidato = corto  # nombre corto: 'sim', 'real'
    elif not candidato.is_absolute() and not candidato.is_file():
        candidato = CARPETA_BASE / candidato
    if not candidato.is_file():
        raise ErrorPerfil('No existe el perfil: {}'.format(perfil))
    return candidato.resolve()


def _validar(datos, ruta):
    errores = []
    for clave in ('nombre', 'es_simulacion'):
        if clave not in datos:
            errores.append("falta la clave '{}'".format(clave))
    for seccion, claves in CLAVES_OBLIGATORIAS.items():
        contenido = datos.get(seccion)
        if not isinstance(contenido, dict):
            errores.append("falta la sección '{}'".format(seccion))
            continue
        for clave in claves:
            if clave not in contenido:
                errores.append("falta '{}.{}'".format(seccion, clave))
    # Unidades: todo parámetro de toda sección debe estar en _doc.
    for seccion, contenido in datos.items():
        if not isinstance(contenido, dict):
            continue
        doc = contenido.get('_doc', {})
        for clave in contenido:
            if clave != '_doc' and clave not in doc:
                errores.append("'{}.{}' no tiene unidad en _doc".format(
                    seccion, clave))
    if errores:
        raise ErrorPerfil('Perfil {} no válido:\n  - {}'.format(
            ruta, '\n  - '.join(errores)))

    # Coherencia de valores.
    g, p, c, k, s = (datos['guia'], datos['politica'], datos['camara'],
                     datos['carril'], datos['seguridad'])
    comprobaciones = [
        (g['metrica_tamano'] in METRICAS_TAMANO,
         'guia.metrica_tamano debe ser uno de {}'.format(METRICAS_TAMANO)),
        (0.0 <= g['confianza_min'] <= 1.0, 'guia.confianza_min fuera de [0, 1]'),
        (g['ancho_imagen_px'] > 0 and g['alto_imagen_px'] > 0,
         'guia: dimensiones de imagen no positivas'),
        (g['escala_tamano'] > 0, 'guia.escala_tamano debe ser positiva'),
        (0.0 < g['suavizado_alfa'] <= 1.0, 'guia.suavizado_alfa fuera de (0, 1]'),
        (g['antiguedad_max_deteccion_s'] > 0, 'guia.antiguedad_max_deteccion_s <= 0'),
        (g['tiempo_perdido_max_s'] > 0, 'guia.tiempo_perdido_max_s <= 0'),
        (0.0 <= p['v_min_m_s'] <= p['v_max_m_s'], 'politica: se requiere 0 <= v_min <= v_max'),
        (p['w_max_rad_s'] > 0 and p['dw_max_rad_s'] > 0, 'politica: w_max y dw_max deben ser positivos'),
        (p['escala_v'] > 0 and p['escala_w'] != 0, 'politica: escala_v debe ser > 0 y escala_w distinta de 0'),
        (c['antiguedad_max_imagen_s'] > 0, 'camara.antiguedad_max_imagen_s <= 0'),
        (0.0 <= k['roi_y_inicio_frac'] < k['roi_y_fin_frac'] <= 1.0,
         'carril: se requiere 0 <= roi_y_inicio < roi_y_fin <= 1'),
        (s['periodo_control_s'] > 0, 'seguridad.periodo_control_s <= 0'),
    ]
    fallos = [msg for ok, msg in comprobaciones if not ok]
    if datos['es_simulacion']:
        if not isinstance(datos.get('simulacion'), dict):
            fallos.append('un perfil de simulación necesita la sección simulacion')
        else:
            fallos += ["falta 'simulacion.{}'".format(c) for c in CLAVES_SIMULACION
                       if c not in datos['simulacion']]
    ep, me = datos['episodio'], datos['metricas']
    sim = datos.get('simulacion') or {}
    if datos['es_simulacion'] and sim.get('modo_reinicio') not in ('reset_simulation', 'colocar'):
        fallos.append("simulacion.modo_reinicio debe ser 'reset_simulation' o 'colocar'")
    if sim.get('modo_reinicio') == 'colocar':
        faltan = [k for k in ('distancia_inicial_m', 'lateral_inicial_m', 'giro_inicial_deg',
                              'espera_tras_colocar_s') if k not in ep]
        if faltan:
            fallos.append('modo_reinicio colocar necesita episodio.' + ', episodio.'.join(faltan))
        elif not (0 < ep['distancia_inicial_m'][0] <= ep['distancia_inicial_m'][1]):
            fallos.append('episodio.distancia_inicial_m debe ser [mín, máx] positivos')
    if not (me['banda_objetivo_m'][0] <= me['separacion_objetivo_m'] <= me['banda_objetivo_m'][1]):
        fallos.append('metricas: la separación objetivo debe estar dentro de la banda')
    if not (me['distancia_contacto_m'] < me['separacion_objetivo_m'] < me['distancia_perdido_m']):
        fallos.append('metricas: se requiere contacto < separación objetivo < perdido')
    if fallos:
        raise ErrorPerfil('Perfil {} incoherente:\n  - {}'.format(
            ruta, '\n  - '.join(fallos)))


def cargar_perfil(perfil=None):
    """Carga, valida y devuelve un Perfil (véase el docstring del módulo)."""
    ruta = resolver_ruta(perfil)
    with open(ruta, encoding='utf-8') as fichero:
        try:
            datos = json.load(fichero)
        except json.JSONDecodeError as error:
            raise ErrorPerfil('JSON no válido en {}: {}'.format(ruta, error))
    _validar(datos, ruta)
    return Perfil(datos, ruta)


def anadir_argumento_perfil(parser):
    """Añade --perfil a un argparse.ArgumentParser (mismo uso en todos los scripts)."""
    parser.add_argument(
        '--perfil', default=None,
        help="Perfil: 'sim', 'real' o ruta a un JSON. Por defecto, la variable "
             "de entorno {} o 'sim'.".format(VARIABLE_ENTORNO))
    return parser


if __name__ == '__main__':
    import argparse
    analizador = anadir_argumento_perfil(argparse.ArgumentParser(
        description='Valida un perfil y muestra sus parámetros con unidades.'))
    print(cargar_perfil(analizador.parse_args().perfil).resumen())
