#!/usr/bin/env python3
"""Evolución y Evaluación del controlador neuronal (NEAT) para Follow City."""

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

CONTADOR = {"pasos": 0, "evaluaciones": 0}
ENTORNO_GLOBAL = None  # Singleton para no crear múltiples nodos ROS 2

class RegistroGeneracion(neat.reporting.BaseReporter):
    """Guarda métricas, progreso y mejores genomas por generación en CSV."""
    CAMPOS = ["generacion", "cal_mejor", "cal_media", "cal_desv", "especies",
              "nodos_mejor", "conexiones_mejor", "evaluaciones", "pasos", "segundos"]

    def __init__(self, ruta_csv, carpeta_genomas):
        self.ruta = ruta_csv
        self.carpeta = carpeta_genomas
        os.makedirs(carpeta_genomas, exist_ok=True)
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
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = len(best_genome.nodes), len([c for c in best_genome.connections.values() if c.enabled])
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
            
        ruta = os.path.join(self.carpeta, f"mejor_gen_{self.gen:03d}.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(best_genome, f)
            
        self.base["evaluaciones"] = evals_totales
        self.base["pasos"] = pasos_totales
        self.base["segundos"] = segundos_totales
        self.inicio = dict(CONTADOR)
        self.t0 = time.time()


def calcular_calidad(registro, pasos_max):
    """Calcula una calidad en escala ~0-100 priorizando la distancia al objetivo."""
    total, w_prev = 0.0, 0.0
    for info in registro:
        visible = float(info['visible'])
        g_centro = np.exp(-(float(info['error_x']) / config.CENTER_SIGMA) ** 2)

        # El tamaño del blob es el proxy de separación disponible en este simulador.
        # Crece de forma continua desde lejos hasta el tamaño objetivo.
        tamano = float(info['tamano_norm'])
        g_dist = np.clip(tamano / config.TARGET_BLOB_SIZE_NORM, 0.0, 1.0)
        recompensa = visible * (0.1 + 0.3 * g_centro + 0.6 * g_dist)

        # Penalización creciente al acercarse al umbral de choque.
        exceso = max(
            0.0,
            (tamano - config.TARGET_BLOB_SIZE_NORM)
            / (config.TOO_CLOSE_BLOB_SIZE_NORM - config.TARGET_BLOB_SIZE_NORM),
        )
        penalizacion_cercania = config.CLOSE_BLOB_PENALTY * exceso ** 2
        penalizacion_giro = (
            config.ANGULAR_PENALTY_WEIGHT
            * abs(float(info['velocidad_angular']))
            / config.MAX_ANGULAR_SPEED_RAD_S
        )

        total += recompensa - penalizacion_cercania - penalizacion_giro
        w = info['velocidad_angular']
        if w * w_prev < 0:                       # el giro cambia de sentido
            total -= config.ZIGZAG_PENALTY * 0.1
        w_prev = w
    calidad = 100.0 * total / pasos_max
    fallo = bool(registro) and registro[-1].get('motivo_fin') not in (None, 'duración máxima')
    if fallo:
        calidad -= config.FAILURE_PENALTY_CAL
    return calidad                               # sin recortar: los negativos valen


def evaluar_genoma(genoma, cfg, num_episodios=1):
    red = neat.nn.FeedForwardNetwork.create(genoma, cfg)
    calidades = []
    for _ in range(num_episodios):
        obs, _ = ENTORNO_GLOBAL.reset()      # si falla, la excepción para la evolución
        registro = []
        while True:
            accion = np.array(red.activate(obs), dtype=np.float32)
            obs, _, terminado, truncado, info = ENTORNO_GLOBAL.step(accion)
            registro.append(info)
            if terminado or truncado:
                break
        CONTADOR["pasos"] += len(registro)
        CONTADOR["evaluaciones"] += 1
        calidades.append(calcular_calidad(registro, ENTORNO_GLOBAL.pasos_max))
    return float(min(calidades))


def eval_genomes(genomas, cfg):
    """Función de callback para Population.run"""
    for _, genoma in genomas:
        genoma.fitness = evaluar_genoma(genoma, cfg)


def dibujar_red(genoma, config_neat, nombres_entrada, nombres_salida, ruta):
    import matplotlib.pyplot as plt
    """Dibuja la topología de la red (según el Anexo)."""
    gc = config_neat.genome_config
    entradas, salidas = list(gc.input_keys), list(gc.output_keys)
    activas = [c.key for c in genoma.connections.values() if c.enabled]
    
    prof = {n: 0 for n in entradas}
    for _ in range(len(genoma.nodes) + 1):
        for (a, b) in activas:
            nueva = prof[a] + 1 if a in prof and a != b else -1
            if prof.get(b, -1) < nueva <= len(genoma.nodes):
                prof[b] = nueva
                
    ocultas = [n for n in prof if n not in entradas and n not in salidas]
    capas = [entradas]
    for p in sorted({prof[n] for n in ocultas}):
        capas.append(sorted(n for n in ocultas if prof[n] == p))
    capas.append(salidas)
    
    pos = {n: (i, j - (len(c)-1)/2) for i, c in enumerate(capas) for j, n in enumerate(c)}
    fig, ax = plt.subplots(figsize=(6,4))
    
    for (a, b) in activas:
        if a in pos and b in pos:
            w = genoma.connections[(a, b)].weight
            ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(
                arrowstyle="->", lw=0.5 + 1.5 * min(abs(w), 3) / 3,
                color="tab:green" if w > 0 else "tab:red"))
                
    nombres = dict(zip(entradas, nombres_entrada))
    nombres.update(zip(salidas, nombres_salida))
    
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=300, color="lightgrey", edgecolor="black", zorder=3)
        ax.text(x, y+0.25, nombres.get(n, str(n)), ha="center", fontsize=8)
        
    ax.axis("off")
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    print(f"Topología guardada en {ruta}")


def modo_entrenar(args):
    global ENTORNO_GLOBAL
    
    ruta_csv = os.path.join(args.carpeta, "evolucion.csv")
    if not args.checkpoint and os.path.exists(ruta_csv):
        print(f"\n[ERROR] La carpeta '{args.carpeta}' ya contiene un archivo evolucion.csv.")
        print("Lanzar una evolución nueva aquí mezclaría los datos o los sobreescribiría.")
        print("Por favor, añade --checkpoint para reanudar, o indica una nueva --carpeta.\n")
        sys.exit(1)
        
    carpeta_chk = os.path.join(args.carpeta, "chk")
    os.makedirs(carpeta_chk, exist_ok=True)

    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
    shutil.copy(args.config, args.carpeta)      # copia el .ini junto a los resultados

    if args.checkpoint:
        print(f"Reanudando desde punto de control: {args.checkpoint}")
        poblacion = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    else:
        poblacion = neat.Population(cfg)

    poblacion.add_reporter(neat.StdOutReporter(True))
    poblacion.add_reporter(neat.StatisticsReporter())
    poblacion.add_reporter(neat.Checkpointer(
        generation_interval=1, filename_prefix=os.path.join(carpeta_chk, "p2-")))
    poblacion.add_reporter(RegistroGeneracion(
        ruta_csv,
        os.path.join(args.carpeta, "genomas")))

    ENTORNO_GLOBAL = FollowCityEnv(episode_seconds=config.EPISODE_SECONDS)

    try:
        mejor = poblacion.run(eval_genomes, args.generaciones)
        ruta = os.path.join(args.carpeta, "mejor_genoma.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(mejor, f)
        print(f"Evolución completada. Mejor genoma guardado en {ruta}")
    except KeyboardInterrupt:
        print("\nEvolución interrumpida por el usuario.")
    except RuntimeError as e:
        print(f"\nFallo de comunicación: {e}")
        print("No se asigna calidad 0. Relanzad el puente y reanudad con "
              f"--checkpoint {carpeta_chk}/p2-N (el N más alto).")
        sys.exit(2)
    finally:
        ENTORNO_GLOBAL.close()


def modo_evaluar(args):
    global ENTORNO_GLOBAL
    if not os.path.exists(args.modelo):
        print(f"Error: no se encontró el genoma {args.modelo}")
        return
        
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, args.config)
                      
    with open(args.modelo, "rb") as f:
        genoma = pickle.load(f)
        
    red = neat.nn.FeedForwardNetwork.create(genoma, cfg)
    
    ENTORNO_GLOBAL = FollowCityEnv(episode_seconds=config.EVAL_EPISODE_SECONDS)
    try:
        print(f"\nEvaluando el genoma {args.modelo}...")
        obs, _ = ENTORNO_GLOBAL.reset()
        pasos, recompensa_total = 0, 0.0
        terminado, truncado = False, False
        
        while not (terminado or truncado):
            accion = np.array(red.activate(obs), dtype=np.float32)
            obs, recompensa, terminado, truncado, info = ENTORNO_GLOBAL.step(accion)
            pasos += 1
            
        motivo = info.get('motivo_fin', 'truncado por tiempo')
        print(f"Episodio finalizado en {pasos} pasos. Motivo: {motivo}")
        
        if args.dibujar:
            nombres_in = ["visible", "err_x", "err_y", "tam", "last_x", "t_perdido"]
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
