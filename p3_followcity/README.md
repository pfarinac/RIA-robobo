# P3 · Seguimiento del guía en la RoboboCity (SAC)

Política de seguimiento aprendida con **SAC** (aprendizaje por refuerzo). Percepción
del guía con `detected_object` y del carril con `camera/image_raw`. El mismo código
funciona en simulación y en el robot real cambiando solo el **perfil**.

## Estado

| Parte | Contenido | Estado |
|---|---|---|
| 0 | Base, comprobación del puente | ✅ `herramientas/comprobar_puente.py` |
| 1 | Perfiles externos sim/real | ✅ `perfiles/`, `perfil.py` |
| 2 | Percepción del guía | ✅ `percepcion_guia.py`, `ros_guia.py`, herramientas de calibración |
| 3 | Percepción del carril (OpenCV) | pendiente (los parámetros ya están en `carril` del perfil) |
| 4–8 | Entorno Gym, SAC, evaluación, transferencia, memoria | pendiente (se reutiliza `p1_followcity`) |

## Estructura

```
p3_followcity/
├── perfiles/perfil_sim.json      # todos los parámetros, con unidades en "_doc"
├── perfiles/perfil_real.json
├── perfil.py                     # carga + validación (unidades obligatorias, coherencia)
├── percepcion_guia.py            # redescripción del guía, sin dependencias de ROS
├── ros_guia.py                   # suscripción a detected_object -> PercepcionGuia
├── herramientas/
│   ├── comprobar_puente.py       # frecuencias, retardos, resolución, servicios
│   ├── inspeccionar_guia.py      # monitor en vivo y captura de calibración
│   └── analizar_calibracion_guia.py  # sim vs real -> escala_tamano + figura
└── tests/test_perfil_y_guia.py   # pruebas sin ROS
```

## Selección del perfil

Todos los scripts aceptan `--perfil sim|real|ruta.json`. Si no se indica, se usa la
variable de entorno `RIA_PERFIL` y, si tampoco existe, `sim`.

```bash
python3 perfil.py --perfil real          # valida y lista parámetros con unidades
export RIA_PERFIL=real                   # alternativa a --perfil
```

Un perfil se rechaza si a algún parámetro le falta su unidad en `_doc` o si los valores
son incoherentes (p. ej. `v_min > v_max`).

## Percepción del guía (2.2)

Observación para la política (5 componentes, `COMPONENTES_OBS_GUIA`):

| Componente | Rango | Significado |
|---|---|---|
| `guia_visible` | {0, 1} | detección nueva, con etiqueta válida, confianza ≥ `confianza_min` y edad ≤ `antiguedad_max_deteccion_s` |
| `guia_x` | [-1, 1] | centro horizontal de la caja (0 centrado, − izquierda); 0 si no visible |
| `guia_tamano` | [0, 1] | `escala_tamano · sqrt(w·h)/sqrt(W·H)` (o altura/ancho según `metrica_tamano`); 0 si no visible |
| `guia_x_ultima` | [-1, 1] | última x válida, para saber hacia dónde se perdió |
| `guia_tiempo_perdido` | [0, 1] | segundos sin verlo / `tiempo_perdido_max_s` |

**Detecciones repetidas:** `readDetectedObject` devuelve la última detección conocida, así
que el puente reenvía objetos antiguos a 5 Hz. Solo se acepta una detección si su
`timestamp` cambia (`usar_timestamp_para_frescura`); si no llega ninguna nueva en
`antiguedad_max_deteccion_s`, el guía pasa a no visible.

## Procedimiento (simulador y real)

```bash
# 1. Puente con los módulos base, camera, object (y sim en simulación)
python3 herramientas/comprobar_puente.py --perfil sim --segundos 10

# 2. Ver qué entrega el detector: resolución real y etiquetas
python3 herramientas/inspeccionar_guia.py --perfil sim
#    -> ajustar guia.ancho_imagen_px / alto_imagen_px y guia.etiquetas_validas

# 3. Calibración a distancias conocidas (repetir en sim y en real)
for d in 0.25 0.35 0.50; do
  python3 herramientas/inspeccionar_guia.py --perfil sim --calibrar --distancia $d \
      --muestras 40 --salida registros/calibracion_guia.csv
done

# 4. Comparar y obtener escala_tamano para el perfil real
python3 herramientas/analizar_calibracion_guia.py registros/calibracion_guia.csv \
    --figura figuras/calibracion_guia.png

# Pruebas sin ROS
python3 tests/test_perfil_y_guia.py
```

## Valores por verificar

- `guia.ancho_imagen_px` / `alto_imagen_px`: 640×480 es provisional. Comprobar con
  `inspeccionar_guia.py` (avisa si alguna coordenada supera el perfil).
- `guia.etiquetas_validas`: vacío = acepta cualquier etiqueta. Fijarlo cuando se sepa la
  etiqueta del guía en el simulador y en el modelo del robot real.
- `camara.tilt_deg`: 90° como en P1; se revisará en la parte 3 para que se vean a la vez
  el carril y el guía.
- `politica.v_min_m_s = 0`: a diferencia de P1, el seguidor debe poder detenerse en las
  paradas del guía (C4).
