"""Parámetros de calibración y de diseño del seguimiento en Follow City.
 
Todo lo que depende de la calibración del sensor o del escenario está

aquí y no en el código, para poder ajustarlo al pasar del simulador al

robot real sin tocar el entorno ni la política (apartado 5.4).

"""
 
# ------------------------------------------------------------------

# Cámara y detector de blobs

# ------------------------------------------------------------------

# Escala con la que se normaliza la posición del blob. El detector

# entrega x e y en una escala de 0 a 100, no en píxeles; la política

# se entrenó con esta normalización y por eso no se ha modificado

# (véase el apartado 4.2 de la memoria).

IMAGE_WIDTH_PX = 320

IMAGE_HEIGHT_PX = 240

# Área máxima del blob según el detector (rango 1-1000, enunciado 5.1).

BLOB_SIZE_MAX = 1000.0

# Tamaño normalizado del blob que se considera la separación deseada.

TARGET_BLOB_SIZE_NORM = 0.2

# Tamaño a partir del cual el guía está demasiado cerca: fin de episodio.

TOO_CLOSE_BLOB_SIZE_NORM = 0.35
 
# ------------------------------------------------------------------

# Infrarrojos

# ------------------------------------------------------------------

# Valor bruto que se toma como saturación para normalizar a [0, 1].

IR_SATURATION = 1000.0

# Índices del array base/ir que se usan como sensores frontales.

FRONT_IR_INDICES = (1, 2, 3, 4, 5)

# Proximidad normalizada que se considera choque inminente.

FRONT_IR_CONTACT_THRESHOLD = 0.85
 
# ------------------------------------------------------------------

# Inclinación de la cámara (TILT)

# ------------------------------------------------------------------

TILT_ANGLE_DEG = 90.0

TILT_SPEED_DEG_S = 20.0

TILT_TIMEOUT_S = 10.0
 
# ------------------------------------------------------------------

# Límites del espacio de acción

# ------------------------------------------------------------------

# Velocidad lineal: la acción [-1, 1] se reescala a [MIN, MAX]. El mínimo

# es positivo para que el seguidor nunca se detenga ni retroceda.

MIN_LINEAR_SPEED_M_S = 0.03

MAX_LINEAR_SPEED_M_S = 0.1

# Velocidad máxima durante los primeros pasos del episodio (arranque suave).

INITIAL_LINEAR_SPEED_M_S = 0.075

INITIAL_SLOW_STEPS = 5

# Pasos iniciales con acción fija de avance recto (demostración, solo SAC).

AUTO_FORWARD_STEPS = 5

# Espera tras el reinicio para que el guía arranque antes que el seguidor.

GUIDE_START_DELAY_S = 7

# Velocidad angular máxima y cambio máximo de giro por paso.

MAX_ANGULAR_SPEED_RAD_S = 0.35

MAX_ANGULAR_CHANGE_RAD_S = 0.15
 
# ------------------------------------------------------------------

# Temporización

# ------------------------------------------------------------------

# Periodo con el que se reenvía cmd_vel durante un paso, por debajo del

# temporizador de seguridad de 0,5 s del puente.

CONTROL_PERIOD_S = 0.1

# Periodo nominal del detector de blobs (5 Hz). Se usa para convertir la

# duración de los episodios en pasos; la duración real medida de un paso

# es de unos 0,4 s.

BLOB_PERIOD_S = 0.2

# Duración nominal de los episodios de entrenamiento y de evaluación.

EPISODE_SECONDS = 45.0

EVAL_EPISODE_SECONDS = 120.0

# Tiempo sin ver la pelota tras el que se da al guía por perdido.

LOST_BLOB_SECONDS = 3.0

# Esperas máximas por mensajes de los sensores.

SENSOR_TIMEOUT_S = 7.0

RESET_SENSOR_TIMEOUT_S = 10.0

# Intentos de reinicio si se pierde el topic de blobs.

RESET_ATTEMPTS = 2
 
# ------------------------------------------------------------------

# Entrenamiento y evaluación

# ------------------------------------------------------------------

TOTAL_TIMESTEPS = 10000

TRAIN_EPISODES = 100

EVAL_EPISODES = 5

MAX_STEPS_PER_EPISODE = max(

    1, int(EPISODE_SECONDS / BLOB_PERIOD_S))
 
# ------------------------------------------------------------------

# Recompensa por paso (todas las magnitudes se infieren de la observación)

# ------------------------------------------------------------------

ALIVE_REWARD = 0.2         # el blob está visible

CENTER_WEIGHT = 0.5        # máximo del premio por centrado (error_x = 0)

CENTER_SIGMA = 0.6         # anchura de la campana en error_x

SIZE_WEIGHT = 0.3          # máximo del premio por tamaño

SIZE_SIGMA = 0.1           # anchura de la campana en tamaño normalizado

SMOOTHNESS_WEIGHT = 0.05   # penaliza |cambio de giro| / máximo por paso

ZIGZAG_PENALTY = 0.3       # penaliza invertir el sentido del giro

LOST_PENALTY = 0.2         # por paso con el blob perdido

FAILURE_PENALTY = 30.0     # choque, demasiado cerca o blob perdido 3 s

# Espera tras publicar la acción antes de leer los sensores.

ACTION_SETTLE_S = 0.2

# Penaliza el módulo de la velocidad angular (girar sin necesidad).

ANGULAR_PENALTY_WEIGHT = 0.05
 
 