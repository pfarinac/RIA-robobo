#!/usr/bin/env python3
"""Evolución y Evaluación del controlador neuronal (NEAT) para Follow City.

Dos modos (excluyentes):
  --entrenar : lanza la evolución con NEAT-Python y guarda CSV, genomas y puntos de control.
  --evaluar  : carga un genoma (.pkl), ejecuta un episodio largo y, con --dibujar, pinta la red.
"""

import argparse
import csv
import os
import pickle
import statistics
import time
import numpy as np
import neat
import sys
import shutil

from follow_city_env import FollowCityEnv
import follow_city_config as config

# Contadores globales de pasos y episodios evaluados (se vuelcan al CSV cada generación).
CONTADOR = {"pasos": 0, "evaluaciones": 0}
ENTORNO_GLOBAL = None  # Singleton para no crear múltiples nodos ROS 2
RUTA_EPISODIOS = None  # ruta del episodios.csv activo (se fija en modo_entrenar / modo_evaluar)

class RegistroGeneracion(neat.reporting.BaseReporter):
    """Guarda métricas, progreso y mejores genomas por generación en CSV."""
    CAMPOS = ["generacion", "cal_mejor", "cal_media", "cal_desv", "especies",
              "nodos_mejor", "conexiones_mejor", "evaluaciones", "pasos", "segundos"]

    def __init__(self, ruta_csv, carpeta_genomas):
        self.ruta = ruta_csv
        self.carpeta = carpeta_genomas
        os.makedirs(carpeta_genomas, exist_ok=True)
        # `base` guarda los acumulados previos para que, al reanudar desde un
        # checkpoint, los contadores sigan sumando sobre los del CSV existente.
        self.base = {"evaluaciones": 0, "pasos": 0, "segundos": 0.0}

        if os.path.exists(ruta_csv):
            with open(ruta_csv) as f:
                filas = list(csv.DictReader(f))
            if filas:
                self.base = {k: float(filas[-1][k]) for k in self.base}
        else:
            with open(ruta_csv, "w", newline="") as f:
                csv.writer(f).writerow(self.CAMPOS)

        # Para que no se pierdan pasos de una generación interrumpida al reanudar
        self.inicio = {"evaluaciones": 0, "pasos": 0}
        self.t0, self.gen = time.time(), 0

    def start_generation(self, generation):
        # NEAT avisa al empezar cada generación.
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        """Se ejecuta tras evaluar la población: escribe una fila del CSV y guarda el mejor genoma."""
        # Calidades de toda la población (para media y desviación).
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = len(best_genome.nodes), len([c for c in best_genome.connections.values() if c.enabled])
        # Distancias de compatibilidad entre todos los pares de genomas; se imprimen
        # por consola (p25, mediana, p75) pero NO se guardan en el CSV.
        distancias = [
            genoma_a.distance(genoma_b, config.genome_config)
            for indice, genoma_a in enumerate(population.values())
            for genoma_b in list(population.values())[indice + 1:]
        ]
        if distancias:
            p25, mediana, p75 = np.percentile(distancias, [25, 50, 75])
            print(
                f"Distancias de compatibilidad: "
                f"p25={p25:.3f}, mediana={mediana:.3f}, p75={p75:.3f}"
            )

        # Acumulados = lo previo (al reanudar) + lo ocurrido desde la última fila escrita.
        evals_totales = int(self.base["evaluaciones"] + CONTADOR["evaluaciones"] - self.inicio["evaluaciones"])
        pasos_totales = int(self.base["pasos"] + CONTADOR["pasos"] - self.inicio["pasos"])
        segundos_totales = round(self.base["segundos"] + time.time() - self.t0, 1)

        fila = [
            self.gen, best_genome.fitness,
            statistics.mean(cal), statistics.pstdev(cal),
            len(species.species), nodos, conexiones,
            evals_totales, pasos_totales, segundos_totales
        ]

        with open(self.ruta, "a", newline="") as f:
            csv.writer(f).writerow(fila)

        # Se guarda el mejor genoma de cada generación (mejor_gen_000.pkl, mejor_gen_001.pkl...).
        ruta = os.path.join(self.carpeta, f"mejor_gen_{self.gen:03d}.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(best_genome, f)

        # Se actualizan los acumulados base y se reinician los marcadores de la generación siguiente.
        self.base["evaluaciones"] = evals_totales
        self.base["pasos"] = pasos_totales
        self.base["segundos"] = segundos_totales
        self.inicio = dict(CONTADOR)
        self.t0 = time.time()


def calcular_calidad(registro, pasos_max):
    """Calcula una calidad en escala ~0-100 usando la separación real.

    `registro` es la lista de diccionarios `info` devueltos por env.step() en el episodio.
    Para cada paso se suma:
      + recompensa (solo si el guía ha arrancado y la pelota es visible):
        ALIVE_REWARD + (1 - ALIVE_REWARD) * g_separacion * (0,7 + 0,3 * g_centro)
      - penalización por velocidad angular y por cambios bruscos de giro
      - LOST_VISIBILITY_PENALTY si no se ve la pelota
      - PRE_START_PENALTY si el guía aún no ha arrancado
      - ZIGZAG_PENALTY si el giro cambia de sentido
    El total se normaliza: 100 * total / pasos_max, y si el episodio acaba en fallo se resta
    FAILURE_PENALTY_CAL + FAILURE_TIME_SCALE * (fracción de episodio que quedaba).
    """
    total, w_prev = 0.0, 0.0
    for info in registro:
        arr = float(info['guia_arrancado'])
        visible = float(info['visible'])
        g_centro = np.exp(-(float(info['error_x']) / config.CENTER_SIGMA) ** 2)
        g_separacion = float(info['g_separacion'])
        base = config.ALIVE_REWARD + (
            1.0 - config.ALIVE_REWARD
        ) * g_separacion * (0.7 + 0.3 * g_centro)
        recompensa = arr * visible * base
        penalizacion_giro = (
            config.ANGULAR_PENALTY_WEIGHT
            * abs(float(info['velocidad_angular']))
            / config.MAX_ANGULAR_SPEED_RAD_S
        )
        penalizacion_cambio_giro = (
            config.ANGULAR_CHANGE_PENALTY_WEIGHT
            * abs(float(info['delta_angular']))
            / config.MAX_ANGULAR_CHANGE_RAD_S
        )

        total += recompensa - penalizacion_giro - penalizacion_cambio_giro
        if not visible:
            total -= config.LOST_VISIBILITY_PENALTY
        if not arr:
            total -= config.PRE_START_PENALTY
        w = info['velocidad_angular']
        if w * w_prev < 0 and min(abs(w), abs(w_prev)) > 0.05:                      # el giro cambia de sentido
            total -= config.ZIGZAG_PENALTY
        w_prev = w
    # Se divide por pasos_max (no por los pasos realizados): un episodio corto puntúa menos.
    calidad = 100.0 * total / pasos_max
    # Fallo = terminó por un motivo distinto de agotar el tiempo.
    fallo = bool(registro) and registro[-1].get('motivo_fin') not in (None, 'duración máxima')
    if fallo:
        restante = 1.0 - len(registro) / pasos_max
        calidad -= config.FAILURE_PENALTY_CAL + config.FAILURE_TIME_SCALE * restante
    return calidad                               # sin recortar: los negativos valen


def registrar_episodio(registro, calidad, numero_evaluacion):
    """Añade una fila a episodios.csv: separación media, fracción visible, pasos, motivo de fin y calidad."""
    if RUTA_EPISODIOS is None or not registro:
        return
    with open(RUTA_EPISODIOS, "a", newline="", encoding="utf-8") as archivo:
        csv.writer(archivo).writerow([
            numero_evaluacion,
            float(np.mean([info['separacion_m'] for info in registro])),
            float(np.mean([info['visible'] for info in registro])),
            len(registro),
            registro[-1].get('motivo_fin'),
            calidad,
        ])


def evaluar_genoma(genoma, cfg, num_episodios=1):
    """Ejecuta `num_episodios` episodios con la red del genoma y devuelve su calidad.

    Con varios episodios se devuelve la peor (min), criterio prudente; con el valor
    por defecto (1) es la calidad de un único episodio.
    """
    red = neat.nn.FeedForwardNetwork.create(genoma, cfg)
    calidades = []
    for _ in range(num_episodios):
        obs, _ = ENTORNO_GLOBAL.reset()      # si falla, la excepción para la evolución
        registro = []
        while True:
            # La red recibe la observación y devuelve [v_lineal, w_angular] en [-1, 1].
            accion = np.array(red.activate(obs), dtype=np.float32)
            obs, _, terminado, truncado, info = ENTORNO_GLOBAL.step(accion)
            registro.append(info)
            if terminado or truncado:
                break
        CONTADOR["pasos"] += len(registro)
        CONTADOR["evaluaciones"] += 1
        calidad = calcular_calidad(registro, ENTORNO_GLOBAL.pasos_max)
        calidades.append(calidad)
        registrar_episodio(registro, calidad, CONTADOR["evaluaciones"])
    return float(min(calidades))


def eval_genomes(genomas, cfg):
    """Función de callback para Population.run

    1) Evalúa a todos los individuos con un episodio.
    2) Reevalúa a los 3 mejores y su calidad final es la media de las dos medidas
       (reduce el efecto del ruido sobre los candidatos más importantes).
    """
    for _, genoma in genomas:
        genoma.fitness = evaluar_genoma(genoma, cfg)
    for _, genoma in sorted(genomas, key=lambda par: par[1].fitness, reverse=True)[:3]:
        genoma.fitness = 0.5 * (
            genoma.fitness + evaluar_genoma(genoma, cfg)
        )


def dibujar_red(genoma, config_neat, nombres_entrada, nombres_salida, ruta):
    import matplotlib.pyplot as plt
    """Dibuja la topología de la red (según el Anexo)."""
    gc = config_neat.genome_config
    entradas, salidas = list(gc.input_keys), list(gc.output_keys)
    # Solo se dibujan las conexiones activas (enabled).
    activas = [c.key for c in genoma.connections.values() if c.enabled]

    # Se calcula la "profundidad" de cada nodo (nº de saltos desde las entradas)
    # para colocar las neuronas ocultas en columnas intermedias.
    prof = {n: 0 for n in entradas}
    for _ in range(len(genoma.nodes) + 1):
        for (a, b) in activas:
            nueva = prof[a] + 1 if a in prof and a != b else -1
            if prof.get(b, -1) < nueva <= len(genoma.nodes):
                prof[b] = nueva

    # Capas: entradas | una capa por cada profundidad de nodos ocultos | salidas.
    ocultas = [n for n in prof if n not in entradas and n not in salidas]
    capas = [entradas]
    for p in sorted({prof[n] for n in ocultas}):
        capas.append(sorted(n for n in ocultas if prof[n] == p))
    capas.append(salidas)

    # Posición de cada nodo: x = índice de capa, y = posición dentro de la capa (centrada).
    pos = {n: (i, j - (len(c)-1)/2) for i, c in enumerate(capas) for j, n in enumerate(c)}
    fig, ax = plt.subplots(figsize=(6,4))

    # Flechas: verde = peso positivo, rojo = negativo; el grosor crece con |peso| (tope en 3).
    for (a, b) in activas:
        if a in pos and b in pos:
            w = genoma.connections[(a, b)].weight
            ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(
                arrowstyle="->", lw=0.5 + 1.5 * min(abs(w), 3) / 3,
                color="tab:green" if w > 0 else "tab:red"))

    # Nombres legibles para entradas y salidas; las ocultas se rotulan con su id.
    nombres = dict(zip(entradas, nombres_entrada))
    nombres.update(zip(salidas, nombres_salida))

    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=300, color="lightgrey", edgecolor="black", zorder=3)
        ax.text(x, y+0.25, nombres.get(n, str(n)), ha="center", fontsize=8)

    ax.axis("off")
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    print(f"Topología guardada en {ruta}")


def modo_entrenar(args):
    """Lanza (o reanuda) la evolución y guarda mejor_genoma.pkl al terminar."""
    global ENTORNO_GLOBAL, RUTA_EPISODIOS

    ruta_csv = os.path.join(args.carpeta, "evolucion.csv")
    RUTA_EPISODIOS = os.path.join(args.carpeta, "episodios.csv")
    # Protección: no mezclar una evolución nueva con los datos de otra ya existente.
    if not args.checkpoint and os.path.exists(ruta_csv):
        print(f"\n[ERROR] La carpeta '{args.carpeta}' ya contiene un archivo evolucion.csv.")
        print("Lanzar una evolución nueva aquí mezclaría los datos o los sobreescribiría.")
        print("Por favor, añade --checkpoint para reanudar, o indica una nueva --carpeta.\n")
        sys.exit(1)

    carpeta_chk = os.path.join(args.carpeta, "chk")
    os.makedirs(carpeta_chk, exist_ok=True)

    # La configuración de NEAT (semilla, población, mutaciones...) se lee del .ini.
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    shutil.copy(args.config, args.carpeta)      # copia el .ini junto a los resultados

    # Reanudar desde un punto de control o crear una población nueva.
    if args.checkpoint:
        print(f"Reanudando desde punto de control: {args.checkpoint}")
        poblacion = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    else:
        poblacion = neat.Population(cfg)

    # Cabecera de episodios.csv (solo si el archivo no existe).
    if not os.path.exists(RUTA_EPISODIOS):
        with open(RUTA_EPISODIOS, "w", newline="", encoding="utf-8") as archivo:
            csv.writer(archivo).writerow([
                "evaluacion", "separacion_media_m", "fraccion_visible",
                "pasos", "motivo_fin", "calidad",
            ])

    # Reporteros: salida por consola, estadísticas, punto de control en cada generación
    # (chk/p2-N) y el registro propio en CSV.
    poblacion.add_reporter(neat.StdOutReporter(True))
    poblacion.add_reporter(neat.StatisticsReporter())
    poblacion.add_reporter(neat.Checkpointer(
        generation_interval=1, filename_prefix=os.path.join(carpeta_chk, "p2-")))
    poblacion.add_reporter(RegistroGeneracion(
        ruta_csv,
        os.path.join(args.carpeta, "genomas")))

    # Episodios de entrenamiento de 30 s (120 pasos nominales).
    ENTORNO_GLOBAL = FollowCityEnv(episode_seconds=config.EPISODE_SECONDS)

    try:
        # `mejor` es el mejor genoma de toda la ejecución según la calidad registrada.
        mejor = poblacion.run(eval_genomes, args.generaciones)
        ruta = os.path.join(args.carpeta, "mejor_genoma.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(mejor, f)
        print(f"Evolución completada. Mejor genoma guardado en {ruta}")
    except KeyboardInterrupt:
        print("\nEvolución interrumpida por el usuario.")
    except RuntimeError as e:
        # Fallo de comunicación con el simulador: no se asigna calidad 0 al individuo; se aborta para poder reanudar.
        print(f"\nFallo de comunicación: {e}")
        print("No se asigna calidad 0. Relanzad el puente y reanudad con "
              f"--checkpoint {carpeta_chk}/p2-N (el N más alto).")
        sys.exit(2)
    finally:
        ENTORNO_GLOBAL.close()


def modo_evaluar(args):
    """Evalúa un genoma guardado con un episodio largo (120 s nominales = 480 pasos)."""
    global ENTORNO_GLOBAL, RUTA_EPISODIOS
    if not os.path.exists(args.modelo):
        print(f"Error: no se encontró el genoma {args.modelo}")
        return

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)

    with open(args.modelo, "rb") as f:
        genoma = pickle.load(f)

    red = neat.nn.FeedForwardNetwork.create(genoma, cfg)

    # El resultado se añade a episodios.csv junto al genoma (las filas de evaluación
    # quedan a continuación de las de la evolución, con numeración que reinicia en 1).
    RUTA_EPISODIOS = os.path.join(os.path.dirname(args.modelo), "episodios.csv")
    if not os.path.exists(RUTA_EPISODIOS):
        with open(RUTA_EPISODIOS, "w", newline="", encoding="utf-8") as archivo:
            csv.writer(archivo).writerow([
                "evaluacion", "separacion_media_m", "fraccion_visible",
                "pasos", "motivo_fin", "calidad",
            ])
    ENTORNO_GLOBAL = FollowCityEnv(episode_seconds=config.EVAL_EPISODE_SECONDS)
    try:
        print(f"\nEvaluando el genoma {args.modelo}...")
        obs, _ = ENTORNO_GLOBAL.reset()
        pasos, recompensa_total = 0, 0.0
        registro = []
        terminado, truncado = False, False

        while not (terminado or truncado):
            accion = np.array(red.activate(obs), dtype=np.float32)
            obs, recompensa, terminado, truncado, info = ENTORNO_GLOBAL.step(accion)
            registro.append(info)
            pasos += 1

        motivo = info.get('motivo_fin', 'truncado por tiempo')
        print(f"Episodio finalizado en {pasos} pasos. Motivo: {motivo}")
        calidad = calcular_calidad(registro, ENTORNO_GLOBAL.pasos_max)
        registrar_episodio(registro, calidad, 1)

        # Con --dibujar se guarda topologia_red.png en el directorio actual.
        if args.dibujar:
            nombres_in = [
                "visible", "err_x", "err_y", "tam", "last_x", "t_perdido",
                "delta_tam", "delta_err_x"
            ]
            nombres_out = ["v_lineal", "w_angular"]
            dibujar_red(genoma, cfg, nombres_in, nombres_out, "topologia_red.png")

    finally:
        ENTORNO_GLOBAL.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neuroevolución con NEAT en Follow City.')
    modos = parser.add_mutually_exclusive_group(required=True)
    modos.add_argument('--entrenar', action='store_true', help='Lanza la evolución.')
    modos.add_argument('--evaluar', action='store_true', help='Evalúa un modelo (genoma) existente.')

    parser.add_argument('--generaciones', type=int, default=30, help='Número de generaciones a evolucionar.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Ruta al archivo de checkpoint para reanudar (ej. chk/p2-12).')
    parser.add_argument('--modelo', type=str, default='mejor_genoma.pkl', help='Ruta al genoma (.pkl) para evaluar.')
    parser.add_argument('--dibujar', action='store_true', help='Si se usa con --evaluar, genera un PNG con la red.')
    parser.add_argument('--config', default='config_p2.ini')
    parser.add_argument('--carpeta', default='evol_s42', help='carpeta de salida; una por semilla')

    args = parser.parse_args()

    if args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)
