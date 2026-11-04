"""Parámetros de calibración y de diseño del seguimiento en Follow City (Práctica 2)."""

# ------------------------------------------------------------------
# Cámara y detector de blobs
# ------------------------------------------------------------------
IMAGE_WIDTH_PX = 320
IMAGE_HEIGHT_PX = 240
BLOB_SIZE_MAX = 1000.0
TARGET_BLOB_SIZE_NORM = 0.2
TOO_CLOSE_BLOB_SIZE_NORM = 0.35

# Nuevas constantes P2
BLOB_COORD_MAX = 100.0        # el detector entrega x e y en 0-100, no en píxeles

# ------------------------------------------------------------------
# Infrarrojos
# ------------------------------------------------------------------
IR_SATURATION = 1000.0
# Se mantienen (1, 2, 3, 4, 5) porque 0 provoca terminaciones por choque trasero/esquina.
FRONT_IR_INDICES = (1, 2, 3, 4, 5)
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
MIN_LINEAR_SPEED_M_S = 0.03
MAX_LINEAR_SPEED_M_S = 0.1
INITIAL_LINEAR_SPEED_M_S = 0.075
INITIAL_SLOW_STEPS = 5
GUIDE_START_DELAY_S = 7
MAX_ANGULAR_SPEED_RAD_S = 0.35
MAX_ANGULAR_CHANGE_RAD_S = 0.15

# ------------------------------------------------------------------
# Temporización
# ------------------------------------------------------------------
CONTROL_PERIOD_S = 0.1
BLOB_PERIOD_S = 0.2
REAL_STEP_S = 0.4             # duración real de un paso (espera 0,2 s + mensaje nuevo a 5 Hz)

# Duración nominal de los episodios (más cortos para evolución, largos para evaluación)
EPISODE_SECONDS = 20.0        # Antes 30
EVAL_EPISODE_SECONDS = 120.0

LOST_BLOB_SECONDS = 3.0
SENSOR_TIMEOUT_S = 7.0
RESET_SENSOR_TIMEOUT_S = 10.0
RESET_SIM_TIMEOUT_S = 10.0
RESET_ATTEMPTS = 2
ACTION_SETTLE_S = 0.2

# ------------------------------------------------------------------
# Coeficientes para el cálculo de la Calidad (Fitness)
# ------------------------------------------------------------------
CENTER_SIGMA = 0.2
CLOSE_BLOB_PENALTY = 1.0
ZIGZAG_PENALTY = 1.0
ANGULAR_PENALTY_WEIGHT = 0.05
FAILURE_PENALTY_CAL = 20.0    # penalización por fallo, en la escala 0-100 de la calidad