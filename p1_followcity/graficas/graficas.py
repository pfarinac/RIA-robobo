"""Genera las figuras 1-3 del apartado 2 a partir de los CSV de TensorBoard.
 
Nombres esperados de los CSV (exportados desde TensorBoard):
  sac_seed2026_ep_rew_mean.csv   sac_seed2027_ep_rew_mean.csv   ppo_seed2026_ep_rew_mean.csv
  sac_seed2026_ep_len_mean.csv   sac_seed2027_ep_len_mean.csv   ppo_seed2026_ep_len_mean.csv
  sac_seed2026_ent_coef.csv      sac_seed2027_ent_coef.csv
  sac_seed2026_critic_loss.csv   sac_seed2027_critic_loss.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
REF_RECOMPENSA = -27.1   # política aleatoria: 8 primeros episodios de SAC 2027
REF_DURACION = 21        # duración media de esos episodios, en pasos
SEMILLAS = [2026, 2027]
COLORES = {2026: 'C2', 2027: 'C3'}
 
 
def cargar(ruta):
    if not os.path.isfile(ruta):
        print('Aviso: falta {}, se omite.'.format(ruta))
        return None
    df = pd.read_csv(ruta)
    return df['Step'].to_numpy(), df['Value'].to_numpy()
 
 
def dibujar_sac(ax, metrica):
    curvas = {s: cargar('sac_seed{}_{}.csv'.format(s, metrica)) for s in SEMILLAS}
    curvas = {s: c for s, c in curvas.items() if c is not None}
    for s, (x, y) in curvas.items():
        ax.plot(x, y, lw=1, alpha=0.35, color=COLORES[s], label='SAC, semilla {}'.format(s))
    if len(curvas) >= 2:
        ini = max(x.min() for x, _ in curvas.values())
        fin = min(x.max() for x, _ in curvas.values())
        rejilla = np.linspace(ini, fin, 400)
        ys = np.array([np.interp(rejilla, x, y) for x, y in curvas.values()])
        ax.fill_between(rejilla, ys.min(0), ys.max(0), color='C0', alpha=0.2,
                        label='SAC, mínimo–máximo')
        ax.plot(rejilla, ys.mean(0), lw=2, color='C0', label='SAC, media')
 
 
def figura_rollout(metrica, etiqueta_y, salida, referencia, texto_ref):
    fig, ax = plt.subplots(figsize=(7, 4))
    dibujar_sac(ax, metrica)
    ppo = cargar('ppo_seed2026_{}.csv'.format(metrica))
    if ppo is not None:
        ax.plot(*ppo, lw=2, color='C1', label='PPO, semilla 2026')
    ax.axhline(referencia, ls='--', color='grey', label=texto_ref)
    ax.set_xlabel('Pasos del entorno (llamadas a step)')
    ax.set_ylabel(etiqueta_y)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    print('Guardada {}'.format(salida))
 
 
figura_rollout('ep_rew_mean',
               'Recompensa por episodio (adim.)\nmedia móvil de 100 episodios',
               'figura1_recompensa.png', REF_RECOMPENSA, 'Política aleatoria (−27,1)')
figura_rollout('ep_len_mean',
               'Duración del episodio (pasos de 0,39 s)\nmedia móvil de 100 episodios',
               'figura2_duracion.png', REF_DURACION, 'Política aleatoria (21 pasos)')
 
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, metrica, etiqueta in [(a1, 'ent_coef', 'Coeficiente de entropía α'),
                              (a2, 'critic_loss', 'Pérdida del crítico')]:
    for s in SEMILLAS:
        datos = cargar('sac_seed{}_{}.csv'.format(s, metrica))
        if datos is not None:
            ax.plot(*datos, color=COLORES[s], label='SAC, semilla {}'.format(s))
    ax.set_xlabel('Pasos del entorno (llamadas a step)')
    ax.set_ylabel(etiqueta)
    ax.grid(alpha=0.3)
 