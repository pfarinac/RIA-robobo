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
| 3 | Percepción del carril (OpenCV) | ✅ `percepcion_carril.py`, `ros_carril.py`, calibración e inspección |
| 4–8 | Entorno Gym, SAC, evaluación, transferencia, memoria | pendiente (se reutiliza `p1_followcity`) |

## Estructura

```
p3_followcity/
├── perfiles/perfil_sim.json      # todos los parámetros, con unidades en "_doc"
├── perfiles/perfil_real.json
├── perfil.py                     # carga + validación (unidades obligatorias, coherencia)
├── percepcion_guia.py            # redescripción del guía, sin dependencias de ROS
├── ros_guia.py                   # suscripción a detected_object -> PercepcionGuia
├── percepcion_carril.py          # detector de carril (modelo geométrico + Hough), sin ROS
├── ros_carril.py                 # suscripción a image_raw -> PercepcionCarril (bajo demanda)
├── herramientas/
│   ├── comprobar_puente.py       # frecuencias, retardos, resolución, servicios
│   ├── inspeccionar_guia.py      # monitor en vivo y captura de calibración
│   ├── analizar_calibracion_guia.py  # sim vs real -> escala_tamano + figura
│   ├── escena_follow_city.py     # configuración C1–C4 / pelota para -sceneConfig
│   ├── inspeccionar_carril.py    # monitor del carril y captura de imágenes de calibración
│   ├── calibrar_carril.py        # horizonte, ancho por píxel, centro óptico y focal
│   └── medir_latencia_camara.py  # retardo orden -> imagen
├── tests/test_perfil_y_guia.py   # pruebas sin ROS
└── tests/test_carril.py          # pruebas del carril con imágenes sintéticas
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

## Percepción del carril (2.1)

Método clásico (OpenCV) con un **modelo geométrico del suelo**. Con la cámara a altura y tilt
fijos, cada línea del carril cumple en la imagen

    x = A + B·(y − y_h) ± (k/2)·(y − y_h)

con `y_h` el horizonte, `k` el ancho del carril por píxel bajo el horizonte, `A = c_x + f·tanθ`
(punto de fuga: orientación) y `B = k·c0/W` (desplazamiento lateral). Por fotograma solo hay
dos incógnitas (A, B), que se ajustan con **una o las dos líneas**; girar el robot mueve A y
desplazarlo cambia B, así que las dos salidas no se mezclan.

1. ROI y máscara HSV de las líneas blancas.
2. Búsqueda tipo Hough de (A, B) limitada al carril que contiene al robot (|desplazamiento| < 0,55)
   para no confundirlo con el carril o el arcén vecino; si hay modelo del fotograma anterior se
   usa como candidato.
3. Selección de tramos cerca de las líneas predichas y ajuste por mínimos cuadrados robusto.
4. Si se ven las dos líneas, el propio fotograma mide su horizonte y su ancho (corrige el
   **cabeceo** del robot al moverse; sin esto, en marcha se perdía la línea discontinua).

| Componente | Rango | Significado |
|---|---|---|
| `carril_valido` | {0, 1} | validez ≥ `validez_min` |
| `carril_desplazamiento` | [-1, 1] | anchos de carril; > 0 robot a la derecha del centro |
| `carril_angulo` | [-1, 1] | ángulo al punto adelantado del centro (pure pursuit) / `escala_angulo_rad`; > 0 a la derecha |
| `carril_tiempo_sin` | [0, 1] | segundos sin carril válido / `tiempo_max_sin_carril_s` |

También se registran la orientación θ, la validez continua, los lados vistos, el residuo y la
antigüedad de la imagen. Coste: ~7 ms por fotograma (CPU).

**Validación en RoboboSim** (147 imágenes colocando al seguidor con posición conocida, tramos
rectos, fotogramas sueltos sin seguimiento):

| Situación | Desplazamiento medido | Orientación medida |
|---|---|---|
| centrado, alineado | −0,00 ± 0,08 | +2,5° ± 2,3 |
| 6 cm a la izquierda / derecha | −0,26 ± 0,08 / +0,15 ± 0,18 | +0,3° / +2,8° |
| girado 15° izq. / der. | +0,00 / −0,10 | +13,5° / −13,1° |

Ancho de carril estimado: 0,31 m. En marcha (seguimiento entre fotogramas) la validez se
mantiene en 0,93–0,97 viendo las dos líneas.

Calibración en el robot real: capturar en un tramo recto 3 imágenes por posición (centrado;
±4–6 cm; ±10–15°) con `inspeccionar_carril.py --capturar`, y después
`calibrar_carril.py --perfil real --carpeta ... --escribir`.

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

## Hallazgos en RoboboSim 4.4.6 (Follow City, C4, sin pelota)

Medido el 24/11 con el simulador Linux, renderizado por software (sin GPU):

- `camera/image_raw` es **480×640 (vertical)**, `bgr8`. Perfiles actualizados.
- `detected_object`: el guía sale con etiqueta **`robobo`**, confianza ~0,63. **`x, y` son el
  centro de la caja**, en las mismas coordenadas que la imagen (480×640).
- Con el tilt inicial del simulador la caja del guía queda cortada por el borde inferior
  (h = 38 px en vez de ~100). Con **tilt = 105°** se ven el guía entero y las líneas del carril
  (provisional hasta la parte 3).
- El puente reenvía la última detección a 5 Hz: solo ~1 de cada 10 mensajes trae `timestamp`
  nuevo. Sin GPU llega una detección nueva cada ~2,3 s (la cámara va a ~1,7 fps). En vuestro PC
  será más rápido: **medir con `comprobar_puente.py` y fijar `guia.antiguedad_max_deteccion_s`
  ≥ 1,5 × ese periodo** (la herramienta lo calcula y avisa).
- Publicar `cmd_vel` a 10 Hz hace que el simulador avise "You are sending too many messages":
  `seguridad.periodo_reenvio_cmd_vel_s` pasa a 0,2 s.
- El guía no arranca solo: hace falta `sim/reset_simulation` (módulo `sim` del puente, que
  necesita el paquete `robobosim`).
- `herramientas/escena_follow_city.py` genera la configuración C1–C4 / pelota para lanzar el
  simulador sin tocar la interfaz (`-sceneConfig`). C4 = Mode3 con valores por defecto.
- **El guía arranca cuando el seguidor se le acerca**, no con el reinicio.
- **`reset_simulation` rompe la cámara del puente:** el simulador reinicia su servidor de
  streaming y `camera_node` intenta reconectar con el socket viejo ("Transport endpoint is
  already connected"); `start_camera` no lo arregla. Hay que relanzar el puente (o al menos
  el nodo de cámara). **Lo tiene que resolver el entorno de la parte 4.**
- **Latencia orden → imagen ~3,9 s** aquí (sin GPU, cola de fotogramas). Medirla en vuestro PC
  con `medir_latencia_camara.py`: condiciona el periodo de control y la observación.
- **Zona muerta de motores:** a 0,02 m/s el robot simulado no avanza. Medir la velocidad mínima
  efectiva antes de fijar el rango de acción.
- Coordenadas de RoboboSim: con rumbo +z, +x queda a la **izquierda** del robot y la rotación
  positiva gira a la izquierda (solo afecta a scripts que usen las poses privilegiadas).

## Valores por verificar

- `guia.etiquetas_validas` del **perfil real**: etiqueta del modelo preentrenado del robot.
- `camara.resolucion_*` y `guia.*_imagen_px` del **perfil real** (480×640 supuesto).
- `guia.antiguedad_max_deteccion_s` en ambos perfiles, según la frecuencia medida.
- `camara.tilt_deg` = 105°: con él se calibró el carril. Si se cambia, recalibrar (`calibrar_carril.py`).
- Sección `carril` del **perfil real**: umbrales HSV y calibración geométrica.
- `politica.v_min_m_s = 0`: a diferencia de P1, el seguidor debe poder detenerse en las
  paradas del guía (C4).
