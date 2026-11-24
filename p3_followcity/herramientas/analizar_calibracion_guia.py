#!/usr/bin/env python3
"""Compara la detección del guía en simulación y en real a distancias conocidas.

Lee el CSV generado por ``inspeccionar_guia.py --calibrar`` (con filas de los
perfiles 'sim' y 'real') y, para cada distancia:

  - media y desviación del tamaño aparente bruto (sin escala_tamano) y de x,
  - porcentaje de detecciones aceptadas por el filtro.

Después calcula la ``escala_tamano`` que hace que el tamaño real coincida
con el simulado por mínimos cuadrados:

    k = sum(s_d · r_d) / sum(r_d²)     (s_d, r_d: medias sim y real a la distancia d)

    escala_tamano_real = k · escala_tamano_sim

y ajusta el modelo tamaño ≈ a / distancia en cada entorno. Con --figura
guarda una gráfica (tamaño frente a distancia) para la memoria.

    python3 herramientas/analizar_calibracion_guia.py registros/calibracion_guia.csv \
        --figura figuras/calibracion_guia.png
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perfil import cargar_perfil  # noqa: E402
from percepcion_guia import PercepcionGuia  # noqa: E402


def medida_bruta(percepcion, fila):
    """Tamaño aparente con escala 1, según la métrica y resolución del perfil."""
    escala = percepcion.escala_tamano
    percepcion.escala_tamano = 1.0
    try:
        return percepcion.tamano_aparente(float(fila['width_px']), float(fila['height_px']))
    finally:
        percepcion.escala_tamano = escala


def resumir(filas, percepciones):
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila['perfil'], float(fila['distancia_m']))].append(fila)
    resumen = {}
    for (perfil, distancia), grupo in sorted(grupos.items()):
        p = percepciones[perfil]
        tam = [medida_bruta(p, f) for f in grupo]
        xs = [p.normalizar_x(float(f['x_px'])) for f in grupo]
        resumen[(perfil, distancia)] = {
            'n': len(grupo),
            'tam_media': statistics.fmean(tam),
            'tam_std': statistics.pstdev(tam) if len(tam) > 1 else 0.0,
            'x_media': statistics.fmean(xs),
            'x_std': statistics.pstdev(xs) if len(xs) > 1 else 0.0,
            'aceptadas_pct': 100.0 * sum(int(f['aceptada']) for f in grupo) / len(grupo),
        }
    return resumen


def ajuste_inverso(puntos):
    """Ajusta tamaño = a / d por mínimos cuadrados. Devuelve a."""
    num = sum(t / d for d, t in puntos)
    den = sum(1.0 / d ** 2 for d, _ in puntos)
    return num / den if den else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv')
    parser.add_argument('--perfil-sim', default='sim')
    parser.add_argument('--perfil-real', default='real')
    parser.add_argument('--figura', default=None, help='ruta PNG de la figura')
    args = parser.parse_args()

    perfil_sim = cargar_perfil(args.perfil_sim)
    perfil_real = cargar_perfil(args.perfil_real)
    percepciones = {perfil_sim.nombre: PercepcionGuia(perfil_sim.guia),
                    perfil_real.nombre: PercepcionGuia(perfil_real.guia)}

    with open(args.csv, encoding='utf-8') as fichero:
        filas = [f for f in csv.DictReader(fichero) if f['perfil'] in percepciones]
    if not filas:
        sys.exit('El CSV no tiene filas de los perfiles {}'.format(list(percepciones)))
    resumen = resumir(filas, percepciones)

    print('{:<6} {:>8} {:>4} {:>10} {:>9} {:>8} {:>8} {:>10}'.format(
        'perfil', 'dist(m)', 'n', 'tam_media', 'tam_std', 'x_media', 'x_std', 'acept(%)'))
    for (perfil, distancia), r in resumen.items():
        print('{:<6} {:>8.3f} {:>4d} {:>10.4f} {:>9.4f} {:>+8.3f} {:>8.3f} {:>10.1f}'.format(
            perfil, distancia, r['n'], r['tam_media'], r['tam_std'],
            r['x_media'], r['x_std'], r['aceptadas_pct']))

    ns, nr = perfil_sim.nombre, perfil_real.nombre
    comunes = sorted({d for p, d in resumen if p == ns} & {d for p, d in resumen if p == nr})
    for nombre in (ns, nr):
        puntos = [(d, r['tam_media']) for (p, d), r in resumen.items() if p == nombre]
        if puntos:
            print('Ajuste {}: tamaño ≈ {:.4f} / distancia(m)'.format(nombre, ajuste_inverso(puntos)))
    if not comunes:
        print('No hay distancias medidas en ambos entornos: no se puede calcular la escala.')
        k = None
    else:
        s = [resumen[(ns, d)]['tam_media'] for d in comunes]
        r = [resumen[(nr, d)]['tam_media'] for d in comunes]
        k = sum(a * b for a, b in zip(s, r)) / sum(b * b for b in r)
        recomendada = k * perfil_sim.guia.escala_tamano
        print('\nDistancias comunes: {}'.format(comunes))
        print('Factor real -> sim k = {:.4f}'.format(k))
        print('Recomendación: guia.escala_tamano en {} = {:.4f} (actual {:.4f})'.format(
            perfil_real.ruta.name, recomendada, perfil_real.guia.escala_tamano))

    if args.figura:
        dibujar(resumen, ns, nr, k, perfil_sim.guia.escala_tamano, args.figura)


def dibujar(resumen, ns, nr, k, escala_sim, ruta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, eje = plt.subplots(figsize=(6, 4))
    series = [(ns, 'Simulación', 'o', 1.0 * escala_sim), (nr, 'Real sin corregir', 's', 1.0)]
    if k is not None:
        series.append((nr, 'Real corregido (k = {:.2f})'.format(k), '^', k * escala_sim))
    for nombre, etiqueta, marca, factor in series:
        datos = sorted((d, r) for (p, d), r in resumen.items() if p == nombre)
        if not datos:
            continue
        eje.errorbar([d for d, _ in datos], [r['tam_media'] * factor for _, r in datos],
                     yerr=[r['tam_std'] * factor for _, r in datos],
                     marker=marca, capsize=3, label=etiqueta)
    eje.set_xlabel('Distancia frontal al guía (m)')
    eje.set_ylabel('Tamaño aparente normalizado (adimensional)')
    eje.set_title('Calibración del tamaño aparente del guía')
    eje.grid(alpha=0.3)
    eje.legend()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    print('Figura guardada en', ruta)


if __name__ == '__main__':
    main()
