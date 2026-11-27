# P3 · Seguimiento del guía en la RoboboCity (SAC)

El Robobo seguidor aprende con **SAC** (aprendizaje por refuerzo) a seguir al guía por el anillo
exterior de la RoboboCity en la configuración **C4** (modo variable con paradas), usando solo sus
sensores: la cámara (guía y carril) y los infrarrojos. El mismo código funciona en RoboboSim y en
el robot real; para pasar de uno a otro basta con cambiar el **perfil** (`--perfil sim` o
`--perfil real`).

Este documento tiene tres partes:

1. [Dónde estamos y qué toca ahora](#1-dónde-estamos-y-qué-toca-ahora)
2. [Qué ha cambiado respecto a la P1 y la P2](#2-qué-ha-cambiado-respecto-a-la-p1-y-la-p2)
3. [Guía de ejecución paso a paso](#3-guía-de-ejecución-paso-a-paso) (terminales y comandos)

Al final está la [referencia técnica](#4-referencia-técnica) con el detalle de cada módulo.

---

## 1. Dónde estamos y qué toca ahora

| Parte | Qué es | Estado |
|---|---|---|
| 0 | Base y comprobación del puente ROS 2 | ✅ hecho |
| 1 | Perfiles externos `perfil_sim.json` / `perfil_real.json` con unidades | ✅ hecho |
| 2 | Percepción del guía con `detected_object` | ✅ hecho y probado en RoboboSim |
| 3 | Percepción del carril con OpenCV desde `camera/image_raw` | ✅ hecho, calibrado en RoboboSim |
| 4 | Entorno Gymnasium (observación, acción, recompensa, parada operativa, registros) | ✅ hecho y probado en RoboboSim |
| 5 | Script de entrenamiento SAC | ⏳ siguiente |
| 6 | Evaluación en C4 y figuras | ⏳ |
| 7 | Transferencia al robot real (calibración, métricas antes/después) | ⏳ |
| 8 | Memoria, vídeo, declaración de IA, `.zip` | ⏳ |

**Ahora os toca ejecutar en vuestro PC los pasos 1 a 6 de la [guía](#3-guía-de-ejecución-paso-a-paso)**
(unos 30–45 minutos). Todo se ha desarrollado y probado con RoboboSim 4.4.6, pero en un entorno
sin GPU donde el simulador va mucho más lento que el tiempo real y la cámara llega con ~4 s de
retraso. Por eso hay tres valores que **solo se pueden medir en vuestro PC** y que hay que poner
en `perfiles/perfil_sim.json` antes de entrenar:

| Qué se mide | Con qué | Qué se ajusta en el perfil |
|---|---|---|
| Periodo entre detecciones nuevas del guía | `comprobar_puente.py` | `guia.antiguedad_max_deteccion_s` ≥ 1,5 × periodo |
| Latencia orden → imagen | `medir_latencia_camara.py` | `episodio.espera_tras_colocar_s` ≥ latencia |
| Que el entorno funcione con vuestro ritmo real | `probar_entorno.py` | nada si los episodios llegan al final |

Mientras tanto, preparo la parte 5 (entrenamiento SAC). Los registros que generéis quedan en
`p3_followcity/registros/`; puedo leerlos directamente desde vuestra carpeta.

---

## 2. Qué ha cambiado respecto a la P1 y la P2

### 2.1. Resumen

| Tema | P1 / P2 | P3 |
|---|---|---|
| Detección del guía | Pelota verde (`color_blobs`) | El propio Robobo guía (`detected_object`), sin pelota |
| Carril | No se usaba | Detector nuevo con OpenCV sobre `camera/image_raw` |
| Parámetros | `follow_city_config.py` (código) | `perfiles/perfil_sim.json` y `perfil_real.json`, con unidades obligatorias |
| Observación | 8 componentes (solo blob y acción) | 12 componentes: guía (5), carril (4), IR (1) y acción anterior (2) |
| Velocidad mínima | 0,03 m/s (no podía pararse) | 0 m/s: puede detenerse en las paradas del guía (C4) |
| IR frontales | Índices 1–5 (incluía un trasero por error) | Índices 0–4 (FrontLL, FrontL, FrontC, FrontR, FrontRR) |
| Reinicio de episodio | `reset_simulation` | Colocar al seguidor detrás del guía con `set_robot_location` |
| Parada operativa | Fin por blob perdido o choque | Contacto, guía sin ver 3 s o carril sin validar 2 s; igual en sim y real |
| Registros | CSV propios por script | `pasos.csv` y `episodios.csv` con todos los campos del apartado 6.4 |

### 2.2. Cambios por parte

**Perfiles (parte 1).** Todo valor ajustable está en `perfiles/*.json`, en secciones `ros`,
`camara`, `carril`, `guia`, `politica`, `seguridad`, `episodio`, `recompensa`, `metricas`,
`aleatorizacion` y (solo sim) `simulacion`. Cada parámetro lleva su unidad en `_doc`, y
`perfil.py` rechaza el perfil si falta una unidad o si hay valores incoherentes. Para cambiar de
entorno no se toca código: `--perfil sim`, `--perfil real` o la variable `RIA_PERFIL`.

**Percepción del guía (parte 2).** `percepcion_guia.py` convierte cada `DetectedObject` en
visible, x ∈ [-1, 1], tamaño aparente ∈ [0, 1], última x conocida y tiempo sin verlo.
- `readDetectedObject` devuelve la **última** detección, así que el puente repite objetos viejos.
  Una detección solo cuenta como nueva si cambia su `timestamp`.
- Se filtra por etiqueta (`robobo` en sim) y confianza.

**Percepción del carril (parte 3).** `percepcion_carril.py` usa OpenCV y un modelo geométrico del
suelo (cámara a altura y tilt fijos).
- Da desplazamiento lateral (en anchos de carril), ángulo al punto adelantado del centro del
  carril, orientación y un **indicador de validez**.
- Funciona con una o con las dos líneas y se corrige solo cuando el robot cabecea.
- Validado en RoboboSim con 147 imágenes de posición conocida. `calibrar_carril.py` lo recalibra
  para el robot real.

**Entorno (parte 4).** `follow_city_p3_env.py`:
- **Acción continua** (v, w) sobre `cmd_vel`. El tilt queda fijo a 105° y fuera de la acción,
  porque la calibración del carril depende de él.
- **Recompensa:** separación de 0,40 m, centrado en el carril, suavidad, pararse cuando el guía
  para, y penalizaciones. En simulación usa la verdad del simulador, que la política nunca ve.
- **Reinicio:** coloca al seguidor 0,30–0,45 m detrás del guía.
- **Métricas:** recorridos sin intervención, % de pasos fuera del carril, contactos y paradas
  superadas.

### 2.3. Cosas que hemos descubierto del simulador y del puente

Afectan a cómo se usa todo esto, así que conviene tenerlas presentes:

- **No uséis `reset_simulation` con la cámara activa.** El simulador reinicia su servidor de
  vídeo y el puente no sabe reconectar: `image_raw` deja de publicar hasta relanzar el puente.
  El entorno no lo usa.
- **El guía solo arranca cuando el seguidor se le acerca**, y **se para a esperar** si el
  seguidor se queda a más de ~1 m.
- **La cámara del simulador es vertical: 480×640.** El detector del guía usa las mismas
  coordenadas, y `x, y` son el centro de la caja.
- **Con el tilt inicial del simulador, la caja del guía sale cortada.** Con 105° se ven bien el
  guía y el carril.
- **No hay que enviar `cmd_vel` más rápido de 5 Hz.** A 10 Hz el simulador avisa de "too many
  messages".
- **El puente convierte v en velocidad de rueda como `int(v / 0,032 · 10)`.** Por debajo de
  ~0,03 m/s la consigna se queda por debajo de 10 y el robot apenas se mueve.
- **Coordenadas de RoboboSim:** con rumbo 0 el robot mira a +z, y +x queda a su izquierda.

---

## 3. Guía de ejecución paso a paso

### 3.1. Qué ventanas hacen falta

| Ventana | Dónde | Para qué | ¿Se queda abierta? |
|---|---|---|---|
| **Simulador** | Windows | RoboboSim en Follow City | Sí, todo el rato |
| **PowerShell** | Windows, carpeta `pc-robobo_new\Windows` | Arrancar el contenedor y abrir terminales | Solo al principio |
| **Terminal A** | Contenedor (`terminal.ps1`) | Puente ROS 2 ↔ simulador | Sí, todo el rato |
| **Terminal B** | Contenedor (`terminal.ps1`) | Nuestros scripts (pruebas, herramientas, entrenamiento) | Sí |
| Terminal C (opcional) | Contenedor (`terminal.ps1`) | Vigilar topics (`ros2 topic hz ...`) | Opcional |

En total: **el simulador y dos terminales del contenedor** (A y B). La tercera solo hace falta
para depurar.

> Dentro del contenedor el repositorio está en `/docker-ros/ws/RIA-robobo`. Los ficheros que
> se generan (registros, figuras, imágenes) aparecen en Windows en
> `pc-robobo_new\Windows\docker-ros\ws\RIA-robobo\p3_followcity\registros`.

### 3.2. Preparación (una sola vez)

**PowerShell**, en `pc-robobo_new\Windows`:

```powershell
.\build.ps1          # solo si la imagen no existe o queréis actualizarla
.\start.ps1
.\terminal.ps1       # abre la Terminal B
```

**Terminal B**, comprobad que están los paquetes de Python que usa la P3:

```bash
python3 -c "import cv2, gymnasium, stable_baselines3, robobopy; print('OK')"
python3 -c "import robobosim; print('robobosim OK')"
```

Si falla `robobosim`, instaladlo. Lo necesita el módulo `sim` del puente, y vuestro Dockerfile no
lo incluye:

```bash
pip install --user --break-system-packages robobosim
```

> Este `pip` se pierde si se reconstruye la imagen. Para que sea permanente, añadid `robobosim` a
> la lista de `pip install` del `Dockerfile` (junto a `robobopy`) y ejecutad `.\build.ps1 -Pip`.

### 3.3. Arranque de cada sesión

**1) Simulador (Windows).** Abrid `Robobo Simulator Win64.exe`, pulsad **World Select** y bajad
hasta **Follow City**. Dentro del mundo:

1. Pulsad el botón de ajustes de arriba a la izquierda (**Scene Settings**).
2. **Follow Mode:** marcad la **tercera** casilla (Mode3 = C4) y dejad vacíos los campos de
   velocidad, tiempos y probabilidad de parada (valores por defecto).
3. Bajad con la barra lateral y **desmarcad Green Ball** (la P3 es sin pelota).
4. **Apply.** Comprobad que el guía ya no lleva la pelota verde.

> C4 es el último modo con los valores por defecto. Para comparar con C1–C3, usad los mismos
> ajustes que en la P1 (C1 crucero a 10, C2 crucero 10–40, C3 variable 10–40) y pasad
> `--configuracion C1`, `C2` o `C3` a los scripts para que quede anotado en los registros.

**2) Contenedor.** En PowerShell, si no está arrancado:

```powershell
.\start.ps1
.\terminal.ps1       # Terminal A
.\terminal.ps1       # Terminal B (en otra ventana de PowerShell)
```

**3) Terminal A: puente ROS 2.** Se deja corriendo:

```bash
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal \
    -p modules:="['camera','object_recognition','sim']"
```

Debe acabar mostrando `Connected to RoboboSim`, `SimNode started` y `Video stream connected`.
Si en algún momento deja de llegar imagen (por ejemplo, tras pulsar el reinicio del simulador),
**Ctrl+C y volved a lanzar este mismo comando**.

**4) Terminal B: carpeta de trabajo.**

```bash
cd /docker-ros/ws/RIA-robobo/p3_followcity
```

Todos los comandos siguientes se ejecutan en la **Terminal B**, desde esta carpeta.

### 3.4. Pasos de comprobación (lo que toca ahora)

**Paso 1. Pruebas sin ROS** (no necesitan simulador; deben salir todas "superadas"):

```bash
python3 perfil.py --perfil sim > /dev/null && echo "perfil sim OK"
python3 perfil.py --perfil real > /dev/null && echo "perfil real OK"
for f in tests/test_*.py; do python3 $f | tail -1; done
```

**Paso 2. Comprobar el puente** (15 s):

```bash
python3 herramientas/comprobar_puente.py --perfil sim --segundos 15
```

Apuntad:
- frecuencia de `image_raw` y su resolución, que debe ser **480x640** y "coincide con el perfil";
- **periodo entre detecciones NUEVAS**. Si pone "DEMASIADO BAJO", subid
  `guia.antiguedad_max_deteccion_s` en `perfiles/perfil_sim.json` al valor recomendado.

**Paso 3. Ver el guía y el carril en vivo** (Ctrl+C para parar cada uno):

```bash
python3 herramientas/inspeccionar_guia.py --perfil sim
python3 herramientas/inspeccionar_carril.py --perfil sim --tilt --salida registros/carril_vivo
```

- En el del guía, la etiqueta debe ser `robobo` y `vis=1` casi siempre con el guía delante.
- En el del carril, con el robot en la carretera, `val=1` y `ambos`. Las imágenes con las
  detecciones dibujadas quedan en `registros/carril_vivo/` (abridlas desde Windows).

**Paso 4. Medir la latencia de la cámara** (el robot gira un poco y vuelve):

```bash
python3 herramientas/medir_latencia_camara.py --perfil sim --repeticiones 3
```

Si la mediana es mayor que 1 s, poned `episodio.espera_tras_colocar_s` a ese valor en
`perfiles/perfil_sim.json`.

**Paso 5. Probar el entorno completo** (unos 2–3 minutos):

```bash
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 60
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 60 --politica aleatoria --sin-check
```

Debéis ver:
- `check_env: OK salvo determinismo (esperado en tiempo real)`;
- cada episodio empieza con `guia_visib=+1.00` y `carril_val` > 0,5;
- los pasos tardan ~0,25 s;
- en el simulador, el seguidor aparece detrás del guía al empezar cada episodio.

Con la política recta es normal que acabe saliéndose del carril; lo importante es que no haya
errores y que los episodios no terminen en el primer paso.

**Paso 6. Revisar los registros**:

```bash
ls registros/pruebas/
column -s, -t < $(ls -td registros/pruebas/*/ | head -1)episodios.csv | less -S
```

`episodios.csv` debe tener una fila por episodio con `motivo_fin`, salidas de carril, contactos y
paradas. Si algo no cuadra, decídmelo: puedo leer los registros directamente desde vuestra
carpeta.

### 3.5. Robot real (más adelante, parte 7)

Lo mismo con `--perfil real`. En la **Terminal A** el puente apunta a la IP del móvil del Robobo y
**sin** el módulo `sim`:

```bash
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=<IP_DEL_MOVIL> \
    -p modules:="['camera','object_recognition']"
```

Calibraciones (Terminal B):

```bash
# Guía a distancias conocidas: repetir para 0.25, 0.35 y 0.50 m, en sim y en real
python3 herramientas/inspeccionar_guia.py --perfil real --calibrar --distancia 0.25 \
    --muestras 40 --salida registros/calibracion_guia.csv
python3 herramientas/analizar_calibracion_guia.py registros/calibracion_guia.csv \
    --figura figuras/calibracion_guia.png        # -> guia.escala_tamano del perfil real

# Carril: en un tramo recto, 3 fotos por posición (centrado; ±4-6 cm; girado ±10-15°)
python3 herramientas/inspeccionar_carril.py --perfil real --tilt --capturar registros/carril_real \
    --lateral 0.0 --rot 0 --muestras 3
python3 herramientas/calibrar_carril.py --perfil real --carpeta registros/carril_real --escribir
```

Convención: `--lateral` > 0 con el robot a la **derecha** del centro, `--rot` > 0 girado a la
**derecha**.

### 3.6. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `No llegan imágenes de .../image_raw` | La cámara del puente se cayó (p. ej. tras un reinicio del simulador) | Ctrl+C en la Terminal A y relanzar el puente |
| `ModuleNotFoundError: robobosim` al lanzar el puente | Falta el paquete | `pip install --user --break-system-packages robobosim` |
| `No llegan las poses del simulador` | Puente sin el módulo `sim` | Lanzar con `modules:="['camera','object_recognition','sim']"` |
| El guía no aparece o lleva pelota verde | Ajustes de escena sin aplicar | Scene Settings → Mode3, sin Green Ball → Apply |
| `guia_visib` parpadea entre 0 y 1 | Detecciones más lentas que `antiguedad_max_deteccion_s` | Paso 2 y subir ese valor |
| Los episodios terminan enseguida con `carril_perdido` | Latencia de cámara mayor que la espera tras colocar | Paso 4 y subir `espera_tras_colocar_s` |
| `Perfil ... no válido` | Parámetro sin unidad o valor incoherente | Leer el mensaje: dice qué clave falla |
| PowerShell no deja ejecutar scripts | Política de ejecución | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |

---

## 4. Referencia técnica

### Estructura de la carpeta

```
p3_followcity/
├── perfiles/perfil_sim.json      # todos los parámetros, con unidades en "_doc"
├── perfiles/perfil_real.json
├── perfil.py                     # carga + validación (unidades obligatorias, coherencia)
├── percepcion_guia.py            # redescripción del guía, sin dependencias de ROS
├── ros_guia.py                   # suscripción a detected_object -> PercepcionGuia
├── percepcion_carril.py          # detector de carril (modelo geométrico + Hough), sin ROS
├── ros_carril.py                 # suscripción a image_raw -> PercepcionCarril (bajo demanda)
├── follow_city_p3_env.py         # entorno Gymnasium (mismo código en sim y real)
├── recompensa_metricas.py        # recompensa, parada operativa y métricas de recorrido (sin ROS)
├── sim_privilegiado.py           # SOLO sim: poses, rastro del guía, colocación, paradas
├── registro.py                   # CSV de pasos y de episodios
├── herramientas/
│   ├── comprobar_puente.py       # frecuencias, retardos, resolución, servicios
│   ├── inspeccionar_guia.py      # monitor en vivo y captura de calibración
│   ├── analizar_calibracion_guia.py  # sim vs real -> escala_tamano + figura
│   ├── escena_follow_city.py     # configuración C1–C4 / pelota para -sceneConfig
│   ├── inspeccionar_carril.py    # monitor del carril y captura de imágenes de calibración
│   ├── calibrar_carril.py        # horizonte, ancho por píxel, centro óptico y focal
│   ├── medir_latencia_camara.py  # retardo orden -> imagen
│   └── probar_entorno.py         # check_env + episodios de prueba con registros
├── tests/test_perfil_y_guia.py   # pruebas sin ROS
├── tests/test_carril.py          # pruebas del carril con imágenes sintéticas
├── tests/test_recompensa_metricas.py
└── tests/test_sim_privilegiado.py
```

### Selección del perfil

Todos los scripts aceptan `--perfil sim|real|ruta.json`. Si no se indica, se usa la
variable de entorno `RIA_PERFIL` y, si tampoco existe, `sim`.

```bash
python3 perfil.py --perfil real          # valida y lista parámetros con unidades
export RIA_PERFIL=real                   # alternativa a --perfil
```

Un perfil se rechaza si a algún parámetro le falta su unidad en `_doc` o si los valores
son incoherentes (p. ej. `v_min > v_max`).

### Percepción del guía (2.2)

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

### Percepción del carril (2.1)

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

### Entorno de la política (2.3, parte 4)

`FollowCityP3Env(perfil, modo, configuracion, carpeta_registros)`: mismo código en sim y real.

**Observación** (12 componentes, todas normalizadas; las de posición valen 0 si el dato no es válido):

| # | Componente | Rango | Origen |
|---|---|---|---|
| 0–4 | guía: visible, x, tamaño, x última, tiempo perdido | ver 2.2 | `detected_object` |
| 5 | `carril_validez` | [0, 1] | indicador continuo de validez |
| 6–7 | `carril_desplazamiento`, `carril_angulo` | [-1, 1] | `image_raw` |
| 8 | `carril_tiempo_sin` | [0, 1] | s sin carril / `tiempo_max_sin_carril_s` |
| 9 | `ir_frontal` | [0, 1] | máximo de los 5 IR frontales (índices 0–4 de `base/ir`) |
| 10–11 | `v_anterior`, `w_anterior` | [0, 1], [-1, 1] | acción del paso anterior (compensa el retardo de la cámara) |

**Acción** continua [-1, 1]²: `v ∈ [v_min, v_max]` con `v_min = 0` (puede detenerse en C4) y
`w ∈ [-w_max, w_max]` con cambio máximo `dw_max` por paso. Se envía `(v·escala_v, w·escala_w)`
por `cmd_vel`, reenviada cada 0,2 s durante el periodo de control (0,25 s).

**PAN/TILT fuera del espacio de acción.** El tilt se fija a 105° al arrancar: la geometría
del carril (horizonte, ancho por píxel) y el tamaño aparente del guía están calibrados para
esa inclinación. Moverlo cambiaría el significado de esas observaciones y obligaría a añadir
el ángulo a la observación y a recalibrar en cada posición.

**Recompensa** (por paso, pesos en `recompensa`): separación (campana en 0,40 m) + centrado
en el carril − fuera de carril − guía no visible − |Δw| + premio por quedarse quieto con el
guía parado y cerca (C4) − penalización al terminar. En simulación usa la distancia y el
desplazamiento **verdaderos** (`usar_privilegiada`); la política nunca los ve.

**Parada operativa** (sim y real): contacto (IR frontal ≥ umbral o, en sim, < 0,20 m),
guía sin ver 3 s o carril sin validar 2 s → se detiene el robot y termina el episodio. En
sim, además, el episodio termina si el guía se aleja a más de 1 m (el guía se para a esperar).

**Reinicio.** En sim se coloca al seguidor 0,30–0,45 m detrás del guía, con ±3 cm y ±8° de
variación, mediante `set_robot_location` (no `reset_simulation`, que rompe la cámara del
puente). Después se espera a tener 3 imágenes seguidas con guía visible y carril válido, lo
que cubre la latencia de la cámara. En real se detiene el robot y se espera a que el operador
lo coloque.

**Verdad en simulación** (`sim_privilegiado.py`): el guía recorre el centro del carril, así
que su rastro de posiciones es la línea central. El desplazamiento lateral real es la
distancia con signo del seguidor a ese rastro (fuera de carril si supera 0,155 m). El guía se
considera parado si su velocidad media en 2 s se mantiene por debajo de 5 mm/s durante 1,5 s.

**Registros** (`registros/<etiqueta>_<modo>_<C>_<fecha>/`):
- `pasos.csv`: marcas temporales, configuración, perfil, observación normalizada, acción y
  `cmd_vel` enviada, detección del guía, validez y salidas del carril, IR, recompensa,
  contacto, salida de carril (verdad y estimada) y verdad del simulador.
- `episodios.csv`: completado sin intervención, motivo de fin, % de pasos fuera de carril
  (verdad y estimada) y número de salidas, contactos, paradas del guía y paradas superadas
  (verdad en sim; estimadas por percepción en real), separación media y mínima.

**Probar el entorno** (simulador abierto en C4 sin pelota y puente lanzado):

```bash
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 40
python3 herramientas/probar_entorno.py --perfil sim --politica aleatoria --sin-check
```

`check_env` de Gymnasium pasa todas las comprobaciones salvo la de determinismo, que no puede
cumplir un robot en tiempo real.

**Antes de entrenar en vuestro PC** (lo que mi entorno sin GPU no puede medir):
1. `medir_latencia_camara.py --perfil sim`: si la latencia es > 0,5 s, subir
   `episodio.espera_tras_colocar_s` a la latencia y revisar `periodo_control_s`.
2. `comprobar_puente.py --perfil sim`: periodo entre detecciones nuevas →
   `guia.antiguedad_max_deteccion_s` ≥ 1,5 × ese periodo.
3. Velocidad mínima efectiva: el puente convierte v a velocidad de rueda como
   `int(v/0,032·10)`, así que por debajo de ~0,03 m/s la consigna es < 10 y el robot apenas
   se mueve. Con `probar_entorno.py --politica recta` y `priv_distancia_m` se puede medir.

### Hallazgos en RoboboSim 4.4.6 (Follow City, C4, sin pelota)

Medido el 24/11 con el simulador Linux, renderizado por software (sin GPU):

- `camera/image_raw` es **480×640 (vertical)**, `bgr8`. Perfiles actualizados.
- `detected_object`: el guía sale con etiqueta **`robobo`**, confianza ~0,63. **`x, y` son el
  centro de la caja**, en las mismas coordenadas que la imagen (480×640).
- Con el tilt inicial del simulador la caja del guía queda cortada por el borde inferior
  (h = 38 px en vez de ~100). Con **tilt = 105°** se ven el guía entero y las líneas del carril
  (valor definitivo: con él está calibrado el carril).
- El puente reenvía la última detección a 5 Hz: solo ~1 de cada 10 mensajes trae `timestamp`
  nuevo. Sin GPU llega una detección nueva cada ~2,3 s (la cámara va a ~1,7 fps). En vuestro PC
  será más rápido: **medir con `comprobar_puente.py` y fijar `guia.antiguedad_max_deteccion_s`
  ≥ 1,5 × ese periodo** (la herramienta lo calcula y avisa).
- Publicar `cmd_vel` a 10 Hz hace que el simulador avise "You are sending too many messages":
  `seguridad.periodo_reenvio_cmd_vel_s` pasa a 0,2 s.
- `herramientas/escena_follow_city.py` genera la configuración C1–C4 / pelota para lanzar el
  simulador sin tocar la interfaz (`-sceneConfig`). C4 = Mode3 con valores por defecto.
- **El guía arranca cuando el seguidor se le acerca**, no con el reinicio.
- **`reset_simulation` rompe la cámara del puente:** el simulador reinicia su servidor de
  streaming y `camera_node` intenta reconectar con el socket viejo ("Transport endpoint is
  already connected"); `start_camera` no lo arregla. Hay que relanzar el puente (o al menos
  el nodo de cámara). El entorno lo evita reiniciando con `set_robot_location`.
- **Latencia orden → imagen ~3,9 s** aquí (sin GPU, cola de fotogramas). Medirla en vuestro PC
  con `medir_latencia_camara.py`: condiciona el periodo de control y la observación.
- **Zona muerta de motores:** a 0,02 m/s el robot simulado no avanza. Medir la velocidad mínima
  efectiva antes de fijar el rango de acción.
- Coordenadas de RoboboSim: con rumbo +z, +x queda a la **izquierda** del robot y la rotación
  positiva gira a la izquierda (solo afecta a scripts que usen las poses privilegiadas).

### Valores por verificar

- `guia.etiquetas_validas` del **perfil real**: etiqueta del modelo preentrenado del robot.
- `camara.resolucion_*` y `guia.*_imagen_px` del **perfil real** (480×640 supuesto).
- `guia.antiguedad_max_deteccion_s` en ambos perfiles, según la frecuencia medida.
- `camara.tilt_deg` = 105°: con él se calibró el carril. Si se cambia, recalibrar (`calibrar_carril.py`).
- Sección `carril` del **perfil real**: umbrales HSV y calibración geométrica.
- `politica.v_min_m_s = 0`: a diferencia de P1, el seguidor debe poder detenerse en las
  paradas del guía (C4).
