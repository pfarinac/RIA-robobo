#!/usr/bin/env python3
"""Genera la configuración de escena de Follow City (RoboboSim 4.4.6).

Es el mismo JSON que guarda el botón Apply de "World Settings" (escena
interna ``CiudadFollow``). Sirve para lanzar el simulador ya configurado,
sin tocar la interfaz, mediante la opción ``-sceneConfig``:

    "Robobo Sim Linux.x86_64" -headless -world CiudadFollow -noLockstep \
        -speed 1 -cameraStream -sceneConfig "$(python3 escena_follow_city.py --config C4)"

Correspondencia con el enunciado (valores vacíos = valores por defecto del
simulador):
    C1 -> Mode1    C2 -> Mode2    C3 -> Mode3 ... C4 -> Mode3 (último modo)
La P3 exige el guía sin pelota verde, así que por defecto --bola no se activa.

AVISO: en modo -headless sin GPU (probado con renderizado por software) el
servidor de cámara (puerto 40405) no se abre; con GPU no se ha probado. Con
-headless el simulador arranca por defecto en lockstep (física parada) y a
velocidad sin límite: para usarlo con el puente ROS 2 hace falta -noLockstep
y -speed 1.
"""

import argparse
import json

MODOS = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 3}


def configuracion(modo, bola=False, v_min='', v_max='', t_min='', t_max='', prob=''):
    grupo = {'type': 'ToggleGroup', 'selected': 'Mode%d' % modo,
             'selectedIndex': modo - 1, 'boolValue': False, 'fields': {}, 'rawFields': {}}

    def campos(**valores):
        return {'type': 'Custom', 'fields': {},
                'rawFields': {k: str(v) for k, v in valores.items()}}

    velocidad = campos(minSpeed=v_min, maxSpeed=v_max)
    paradas = campos(minStopTime=t_min, maxStopTime=t_max)
    probabilidad = campos(stopChance=prob)
    pelota = {'type': 'Toggle', 'selected': str(bola), 'boolValue': bola,
              'value': float(bola), 'fields': {'value': float(bola)}, 'rawFields': {}}
    opciones = {'modeSelector': grupo, 'follow Mode': grupo,
                'speed': velocidad, 'speed Range': velocidad,
                'stopTimes': paradas, 'stop Times': paradas,
                'stopChance': probabilidad, 'stop Chance': probabilidad,
                'greenBall': pelota, 'green Ball': pelota}
    return {'sceneName': 'CiudadFollow', 'version': 1, 'options': opciones}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', choices=sorted(MODOS), default='C4')
    parser.add_argument('--bola', action='store_true', help='mostrar la pelota verde')
    parser.add_argument('--v-min', default='')
    parser.add_argument('--v-max', default='')
    parser.add_argument('--parada-min', default='', help='s')
    parser.add_argument('--parada-max', default='', help='s')
    parser.add_argument('--prob-parada', default='')
    a = parser.parse_args()
    print(json.dumps(configuracion(MODOS[a.config], a.bola, a.v_min, a.v_max,
                                   a.parada_min, a.parada_max, a.prob_parada),
                     separators=(',', ':')))


if __name__ == '__main__':
    main()
