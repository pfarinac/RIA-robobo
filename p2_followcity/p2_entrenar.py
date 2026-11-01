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
import matplotlib.pyplot as plt
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
                
        self.inicio = dict(CONTADOR)
        self.t0, self.gen = time.time(), 0

    def start_generation(self, generation):
        self.gen = generation

    def post_evaluate(self, config, population, species, best_genome):
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = len(best_genome.nodes), len([c for c in best_genome.connections.values() if c.enabled])
        fila = [
            self.gen, best_genome.fitness,
            statistics.mean(cal), statistics.pstdev(cal),
            len(species.species), nodos, conexiones,
            int(self.base["evaluaciones"] + CONTADOR["evaluaciones"] - self.inicio["evaluaciones"]),
            int(self.base["pasos"] + CONTADOR["pasos"] - self.inicio["pasos"]),
            round(self.base["segundos"] + time.time() - self.t0, 1)
        ]
        with open(self.ruta, "a", newline="") as f:
            csv.writer(f).writerow(fila)
            
        ruta = os.path.join(self.carpeta, f"mejor_gen_{self.gen:03d}.pkl")
        with open(ruta, "wb") as f:
            pickle.dump(best_genome, f)


def calcular_calidad(registro, pasos_max):
    """Calidad en escala ~0-100. Mixta: suma por paso + penalización si hay fallo."""
    total, w_prev = 0.0, 0.0
    for info in registro:
        if info['visible']:
            total += config.ALIVE_REWARD + info['r_centro'] + info['r_tamano']
        else:
            total -= config.LOST_PENALTY
        total += info['r_suavidad']
        w = info['velocidad_angular']
        if w * w_prev < 0:                       # el giro cambia de sentido
            total -= config.ZIGZAG_PENALTY
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
        os.path.join(args.carpeta, "evolucion.csv"),
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
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, "args.config")
                      
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
            nombres_in = ["visible", "err_x", "err_y", "tam", "last_x", "t_perdido", "v_ant", "w_ant"]
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