"""Controlador de referencia escrito a mano (sin aprendizaje, sin ROS).

Usa exactamente la misma observación y el mismo espacio de acción que la
política SAC, así que sirve para dos cosas:

1. **Demostración inicial** del entrenamiento: antes de que SAC empiece a
   aprender, se rellenan los primeros pasos del buffer de repetición con
   transiciones de este controlador (con ruido), en lugar de acciones
   aleatorias. Cada paso cuesta tiempo real, y las acciones aleatorias apenas
   siguen al guía. Equivale a la "demostración inicial" de la P1, pero con
   las acciones realmente ejecutadas guardadas en el buffer.
2. **Línea base** para la memoria: ``p3_entrenar.py --evaluar --politica
   heuristica`` mide lo que consigue un controlador clásico con la misma
   percepción, para comparar con lo aprendido.

Reglas:
- Giro: *pure pursuit* sobre el punto adelantado del centro del carril
  (componente ``carril_angulo``), más una corrección hacia el guía cuando se
  ve. Con w > 0 a la izquierda (ROS), un punto a la derecha (ángulo > 0) pide
  w < 0.
- Velocidad: proporcional al error de tamaño aparente del guía respecto al
  objetivo (más pequeño = más lejos = más rápido). Se detiene si está
  demasiado cerca, si hay proximidad IR o si no hay carril válido.
"""

import numpy as np

# Índices de la observación (véase follow_city_p3_env.NOMBRES_OBS).
I_VISIBLE, I_GX, I_TAM, I_GX_ULT, I_PERDIDO = 0, 1, 2, 3, 4
I_VALIDEZ, I_DESP, I_ANG, I_SIN_CARRIL, I_IR = 5, 6, 7, 8, 9


class ControladorHeuristico:
    def __init__(self, perfil, ganancia_carril=1.6, ganancia_guia=0.5,
                 ganancia_tamano=2.5, v_crucero_m_s=0.12, v_buscar_m_s=0.05):
        p = perfil.politica
        self.v_min, self.v_max = p.v_min_m_s, p.v_max_m_s
        self.tam_obj = perfil.guia.tamano_objetivo_norm
        self.tam_cerca = perfil.guia.tamano_demasiado_cerca_norm
        self.ir_umbral = 0.8 * perfil.seguridad.ir_umbral_contacto
        self.k_carril, self.k_guia, self.k_tam = ganancia_carril, ganancia_guia, ganancia_tamano
        self.v_crucero, self.v_buscar = v_crucero_m_s, v_buscar_m_s

    def _v_a_accion(self, v):
        v = float(np.clip(v, self.v_min, self.v_max))
        return 2.0 * (v - self.v_min) / max(self.v_max - self.v_min, 1e-9) - 1.0

    def __call__(self, obs):
        obs = np.asarray(obs, np.float64)
        visible = obs[I_VISIBLE] > 0.5
        carril = obs[I_VALIDEZ] > 0.0

        # Giro (acción normalizada: a1 = w / w_max).
        giro = -self.k_carril * obs[I_ANG] if carril else 0.0
        if visible:
            giro += -self.k_guia * obs[I_GX]
        elif not carril:
            giro = -0.5 * np.sign(obs[I_GX_ULT])     # buscar hacia donde se perdió

        # Velocidad.
        if obs[I_IR] >= self.ir_umbral or (visible and obs[I_TAM] >= self.tam_cerca):
            v = 0.0
        elif visible:
            v = self.v_crucero + self.k_tam * (self.tam_obj - obs[I_TAM])
        else:
            v = self.v_buscar if carril else 0.0
        return np.array([self._v_a_accion(v), float(np.clip(giro, -1.0, 1.0))], np.float32)
