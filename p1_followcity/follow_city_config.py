"""Calibración inicial para el seguimiento de la pelota verde."""

IMAGE_WIDTH_PX = 320
IMAGE_HEIGHT_PX = 240
BLOB_SIZE_MAX = 10000.0
TARGET_BLOB_SIZE_NORM = 0.2
TOO_CLOSE_BLOB_SIZE_NORM = 0.4

IR_SATURATION = 1000.0
FRONT_IR_INDICES = (1, 2, 3)
FRONT_IR_CONTACT_THRESHOLD = 0.85



TILT_ANGLE_DEG = 90.0
TILT_SPEED_DEG_S = 20.0
TILT_TIMEOUT_S = 10.0
MIN_LINEAR_SPEED_M_S = 0.03
MAX_LINEAR_SPEED_M_S = 0.1
INITIAL_LINEAR_SPEED_M_S = 0.075
INITIAL_SLOW_STEPS = 5
AUTO_FORWARD_STEPS = 5
GUIDE_START_DELAY_S = 7
MAX_ANGULAR_SPEED_RAD_S = 0.35
MAX_ANGULAR_CHANGE_RAD_S = 0.15
CONTROL_PERIOD_S = 0.1
BLOB_PERIOD_S = 0.2
EPISODE_SECONDS = 60.0
LOST_BLOB_SECONDS = 3.0
SENSOR_TIMEOUT_S = 3.0
RESET_SENSOR_TIMEOUT_S = 10.0
RESET_ATTEMPTS = 2
TOTAL_TIMESTEPS = 10000
TRAIN_EPISODES = 100
EVAL_EPISODES = 5
MAX_STEPS_PER_EPISODE = max(
    1, int(EPISODE_SECONDS / BLOB_PERIOD_S))
# Recompensa
FOLLOW_REWARD = 1.0          # máximo por paso si está centrado y a la distancia objetivo
FOLLOW_SIGMA_X = 0.3         # anchura de la campana en error_x
FOLLOW_SIGMA_SIZE = 0.08     # anchura de la campana en tamaño normalizado
SMOOTHNESS_WEIGHT = 0.1      # penalización por cambio brusco de giro
FAILURE_PENALTY = 30.0       # choque, demasiado cerca o blob perdido

# Temporización del paso
ACTION_SETTLE_S = 0.1        # tiempo mínimo entre la acción y la observación