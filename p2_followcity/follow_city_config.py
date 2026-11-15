"""Parámetros de calibración y de diseño del seguimiento en Follow City (Práctica 2).

"""

# ------------------------------------------------------------------
# Cámara y detector de blobs
# ------------------------------------------------------------------
IMAGE_WIDTH_PX = 320            # resolución de la imagen (informativo; no se usa en el entorno)
IMAGE_HEIGHT_PX = 240
BLOB_SIZE_MAX = 1000.0          # tamaño máximo del blob; se usa para normalizar 'tam' a [0, 1]
# Tamaños normalizados del blob de referencia. Solo los usa el fallback visual
# de la separación (_separacion) cuando no hay datos del simulador.
TARGET_BLOB_SIZE_NORM = 0.2     # tamaño al que corresponde la separación objetivo
TOO_CLOSE_BLOB_SIZE_NORM = 0.35 # tamaño "demasiado cerca"
TOO_FAR_BLOB_SIZE_NORM = 0.05   # tamaño "demasiado lejos"

# Separación privilegiada del simulador. El fallback visual requiere calibrar
# estos tres tamaños con el escenario concreto antes de usarlo.
USE_SIM_SEPARATION = True       # True: la separación se mide con las poses reales del simulador
FOLLOWER_SIM_NAMESPACE = '/robobo/robot_0/sim'   # namespace de donde se lee la pose del seguidor
GUIDE_SIM_NAMESPACE = '/robobo/robot_0/sim'      # namespace de donde se lee la lista de objetos (guía)
GUIDE_SIM_OBJECT_ID = 'PATHFOLLOWROBOBO'         # id del objeto del guía dentro de esa lista
TARGET_SEPARATION_M = 0.325     # separación ideal (centro de la banda 0,20-0,45 m) en metros
SEPARATION_SIGMA_M = 0.1        # anchura de la campana gaussiana de la recompensa de separación
# No terminar por el tamaño aparente del guía cuando la escena fija la
# posición inicial de ambos robots. El contacto físico sigue controlado por IR.
MIN_SEPARATION_M = 0.0          # con 0,0 la condición "demasiado_cerca" nunca se cumple
MAX_SEPARATION_M = 0.45         # límite superior de la banda aceptable (también punto del fallback)
LOST_GUIDE_SEPARATION_M = 0.90  # separación a partir de la cual se daría por perdido al guía

# Nuevas constantes P2
BLOB_COORD_MAX = 100.0        # el detector entrega x e y en 0-100, no en píxeles
ARRANQUE_UMBRAL_GUIA_M = 0.05 # el guía "ha arrancado" cuando se desplaza más de 5 cm desde su pose inicial

# ------------------------------------------------------------------
# Infrarrojos
# ------------------------------------------------------------------
IR_SATURATION = 1000.0          # lectura IR a la que se considera saturado (normaliza a [0, 1])
# Se mantienen (1, 2, 3, 4, 5) porque 0 provoca terminaciones por choque trasero/esquina.
FRONT_IR_INDICES = (1, 2, 3, 4, 5)    # sensores frontales usados para detectar choque
FRONT_IR_CONTACT_THRESHOLD = 0.85     # proximidad normalizada a partir de la cual hay "choque"

# ------------------------------------------------------------------
# Inclinación de la cámara (TILT)
# ------------------------------------------------------------------
TILT_ANGLE_DEG = 90.0           # ángulo de inclinación de la cámara al empezar cada episodio
TILT_SPEED_DEG_S = 20.0         # velocidad del movimiento de inclinación
TILT_TIMEOUT_S = 10.0           # (no se usa en el entorno actual)

# ------------------------------------------------------------------
# Límites del espacio de acción
# ------------------------------------------------------------------
MIN_LINEAR_SPEED_M_S = 0.0      # velocidad lineal mínima: el robot nunca marcha atrás
MAX_LINEAR_SPEED_M_S = 0.1      # velocidad lineal máxima (acción +1)
INITIAL_LINEAR_SPEED_M_S = 0.075  # tope de velocidad durante los primeros pasos
INITIAL_SLOW_STEPS = 5          # nº de pasos con el tope reducido
# Pasos iniciales con avance recto para arrancar cada episodio.
AUTO_FORWARD_STEPS = 5          # la red no decide en estos pasos: acción forzada (1, 0)
GUIDE_START_DELAY_S = 1.5       # espera tras el reset para que el guía se ponga en marcha
MAX_ANGULAR_SPEED_RAD_S = 0.35  # velocidad angular máxima (acción ±1)
MAX_ANGULAR_CHANGE_RAD_S = 0.15 # máximo cambio de velocidad angular entre dos pasos (suaviza el giro)

# ------------------------------------------------------------------
# Temporización
# ------------------------------------------------------------------
CONTROL_PERIOD_S = 0.1          # cada cuánto se republica la orden de velocidad mientras se espera
BLOB_PERIOD_S = 0.2             # periodo de los mensajes de blobs (informativo)
REAL_STEP_S = 0.25            # horizonte nominal de control por paso (el real medido es mayor)

# Duración nominal de los episodios (más cortos para evolución, largos para evaluación)
# pasos_max = duración / REAL_STEP_S -> 30 s = 120 pasos; 120 s = 480 pasos.
EPISODE_SECONDS = 30.0
EVAL_EPISODE_SECONDS = 120.0

LOST_BLOB_SECONDS = 3.0         # tiempo sin ver la pelota que termina el episodio (y normaliza t_perdido)
SENSOR_TIMEOUT_S = 7.0          # espera máxima de un mensaje de sensor
RESET_SENSOR_TIMEOUT_S = 10.0   # espera máxima de sensores tras un reset
RESET_SIM_TIMEOUT_S = 10.0      # (no se usa en el entorno actual)
RESET_ATTEMPTS = 2              # intentos de reset antes de dar error
ACTION_SETTLE_S = 0.2           # tiempo que se deja actuar a la orden antes de leer sensores

# ------------------------------------------------------------------
# Coeficientes para el cálculo de la Calidad (Fitness)
# ------------------------------------------------------------------
ALIVE_REWARD = 0.1              # recompensa base por paso vivo (parte fija de 'base')
PRE_START_PENALTY = 0.5         # penalización por paso mientras el guía aún no ha arrancado
FAILURE_TIME_SCALE = 30.0       # penalización extra por fallo, proporcional al tiempo que quedaba
CENTER_SIGMA = 0.2              # anchura de la campana que premia tener la pelota centrada
CLOSE_BLOB_PENALTY = 2.0        # (no se usa en la calidad actual)
PROXIMITY_PENALTY_START_NORM = 0.25  # (no se usa en la calidad actual)
LOST_VISIBILITY_PENALTY = 0.3   # penalización por paso sin ver la pelota
ZIGZAG_PENALTY = 0.3            # penalización cada vez que el giro cambia de sentido
ANGULAR_PENALTY_WEIGHT = 0.05   # peso de la penalización por girar rápido
ANGULAR_CHANGE_PENALTY_WEIGHT = 0.1  # peso de la penalización por cambios bruscos de giro
FAILURE_PENALTY_CAL = 20.0    # penalización por fallo, en la escala 0-100 de la calidad