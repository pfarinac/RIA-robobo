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
| 5 | Script de entrenamiento SAC (`p3_entrenar.py`) | ✅ hecho y probado en RoboboSim |
| 6 | Evaluación en C4 y figuras | ⏳ siguiente (tras entrenar) |
| 7 | Transferencia al robot real (calibración, métricas antes/después) | ⏳ |
| 8 | Memoria, vídeo, declaración de IA, `.zip` | ⏳ |

**Ahora toca entrenar** (sección [3.5](#35-entrenamiento-parte-5)). Las comprobaciones de los pasos
1–6 ya se hicieron el 28/11 y los perfiles están ajustados con vuestras medidas:

| Medida (28/11, vuestro PC) | Valor | Ajuste aplicado |
|---|---|---|
| Detecciones nuevas del guía | cada 0,51 s | `guia.antiguedad_max_deteccion_s` = 0,9 s |
| Latencia orden → imagen | 0,21 s | (cubierta por la espera de 3 imágenes válidas al empezar) |
| Velocidad del guía en C4 | hasta ~0,17 m/s | `politica.v_max_m_s` = 0,25 (antes 0,10), `w_max` = 0,6 rad/s |
| Velocidad real del seguidor frente a la enviada (28/11) | ~2× (0,25 enviado → ~0,50 m/s) | `politica.escala_v` = 0,5 y `escala_w` = −0,5: la acción queda en m/s reales |
| Signo del giro (28/11) | `angular.z` > 0 gira a la **derecha** en RoboboSim (al revés que ROS) | `escala_w` negativa: en la política, w > 0 sigue siendo izquierda. En el robot real hay que comprobarlo (parte 7) |

> **Cambio del 28/11 (tarde): reinicio desde la posición inicial.** Cada episodio empieza
> ahora con `reset_simulation`, como en el anexo B de la P2: guía y seguidor vuelven a la
> posición inicial de la escena y el seguidor parte parado. Como el reinicio corta las
> conexiones del puente con el simulador, **el puente lo lanza y lo relanza el propio script**
> (la Terminal A ya solo hace falta para las herramientas de comprobación). Las métricas siguen los parámetros del anexo B. Detalles
> en [2.3](#23-cosas-que-hemos-descubierto-del-simulador-y-del-puente).

El orden recomendado es:
1. Probar el script con un entrenamiento corto (5 minutos).
2. Medir la línea base heurística.
3. Lanzar el entrenamiento largo (150 episodios, unas 2–3 h, reanudable).

---|---|---|
| Periodo entre detecciones nuevas del guía | `comprobar_puente.py` | `guia.antiguedad_max_deteccion_s` ≥ 1,5 × periodo |
| Que el entorno funcione con vuestro ritmo real | `probar_entorno.py` | nada si los episodios llegan al final |

Después viene la parte 5 (script de entrenamiento SAC). Los registros que generéis quedan en
`p3_followcity/registros/`, así que se pueden revisar directamente desde la carpeta.

---

## 2. Qué ha cambiado respecto a la P1 y la P2

### 2.1. Resumen

| Tema | P1 / P2 | P3 |
|---|---|---|
| Detección del guía | Pelota verde (`color_blobs`) | El propio Robobo guía (`detected_object`), sin pelota |
| Carril | No se usaba | Detector nuevo con OpenCV sobre `camera/image_raw` |
| Parámetros | `follow_city_config.py` (código) | `perfiles/perfil_sim.json` y `perfil_real.json`, con unidades obligatorias |
| Observación | 8 componentes (blob, tiempo sin verlo y velocidades) | 12 componentes: guía (5), carril (4), IR (1) y acción anterior (2) |
| Velocidad mínima | 0,03 m/s en la P1 (no podía pararse) | 0 m/s: puede detenerse en las paradas del guía (C4) |
| IR frontales | Índices 1–5 (incluía un trasero por error) | Índices 0–4 (FrontLL, FrontL, FrontC, FrontR, FrontRR) |
| Reinicio de episodio | `reset_simulation` | `reset_simulation` (posición inicial), con el puente lanzado y relanzado por el script |
| Fin de episodio | Fin por blob perdido o choque | Anexo B: colisión (< 0,12 m), guía perdido (> 0,90 m o sin verlo 3 s); además, carril sin validar 2 s. Igual en sim y real (en real, contacto por IR) |
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

**Entrenamiento (parte 5).** `p3_entrenar.py` mantiene la estructura de `p1_entrenar.py`: modos
`--calibrar`, `--entrenar` y `--evaluar`, SAC con hiperparámetros reducidos, un callback que
cuenta episodios y se detiene al alcanzar el objetivo, guardado del modelo si se interrumpe y
evaluación determinista al terminar. Cambios respecto a la P1:

| P1 | P3 |
|---|---|
| `follow_city_config.py` | `--perfil sim` o `--perfil real` |
| SAC o PPO | Solo SAC (el de la P3) |
| 5 pasos fijos de "avance recto" al empezar cada episodio | `--pasos-demostracion`: antes de aprender se llenan los primeros pasos del buffer con un **controlador heurístico** (pure pursuit del carril + distancia al guía) con ruido, guardando la acción ejecutada |
| Un solo `.zip` al final | **Puntos de control** cada `--guardar-cada` episodios (modelo + buffer de repetición + `estado.json`) y **`--continuar`** para reanudar |
| Resumen con recompensa y motivo de fin | Métricas del enunciado: recorridos sin intervención, % fuera de carril, contactos y paradas superadas; `pasos.csv` y `episodios.csv` |
| — | `--evaluar --politica heuristica`: la misma evaluación con el controlador heurístico, como **línea base** para la memoria |

### 2.3. Cosas que hemos descubierto del simulador y del puente

Afectan a cómo se usa todo esto, así que conviene tenerlas presentes:

- **`reset_simulation` corta las conexiones del puente.** Con RoboboSim 4.4.6 y el puente
  actual, tras el reinicio dejan de funcionar `cmd_vel` y `move_tilt` (el simulador cierra la
  conexión de control) y `image_raw` deja de publicar (cierra la de vídeo). Lo engañoso es que
  `base/ir` y `detected_object` **siguen publicando el último valor**, así que parece vivo.
  Por eso el entorno, tras cada reinicio, fija el tilt y espera una imagen nueva: si algo no
  responde, relanza el puente (`puente.py`, unos segundos) y sigue. Si en vuestro PC el
  reinicio no rompe nada, no se relanza nunca. El número de relanzamientos queda en
  `registros/.../puente.log`.
- **En vuestro PC el reinicio casi siempre deja el puente vivo, pero no siempre:** en el
  primer entrenamiento (28/11) falló tras 79 reinicios. Por eso para entrenar es mejor dejar
  que el script gestione el puente (sin `--puente-externo`): si un reinicio lo rompe, lo
  relanza y sigue.
- **SAC actualiza la red al final de cada episodio**, con el robot parado. Con una
  actualización por paso (como en la P1), en el contenedor cada paso pasaba de 0,26 s a
  ~0,9 s: el robot recibía órdenes a ~1 Hz y los episodios acababan por carril o guía
  perdidos en 3–4 pasos. Además PyTorch usa 1 hilo (`--hilos-torch`).
- **Reinicio = posición inicial de la escena.** El seguidor empieza a ~0,76 m del guía (0,68 m
  de separación medida al punto de referencia), parado.
- **El guía solo arranca cuando el seguidor se le acerca**, y **se para a esperar** si el
  seguidor se queda a más de ~1 m.
- **La cámara del simulador es vertical: 480×640.** El detector del guía usa las mismas
  coordenadas, y `x, y` son el centro de la caja.
- **Con el tilt inicial del simulador, la caja del guía sale cortada.** Con 105° se ven bien el
  guía y el carril.
- **El guía es blanco, como las líneas.** Con el guía delante, el detector de carril tomaba su
  cuerpo por la línea discontinua y el robot giraba hacia donde no debía. Ahora se excluye la
  zona de la caja del guía (ampliada) al buscar las líneas.
- **El tamaño del guía se mide por el ancho de la caja, no por el área.** El detector solo encuadra
  el cuerpo blanco. Por debajo de ~0,35 m su parte inferior se sale de la imagen con cualquier
  tilt (se probaron 90°, 95°, 100° y 105°), así que la altura y el área dejan de crecer y el
  robot no sabía si estaba cerca o lejos. El ancho crece hasta ~0,32 m (≈ 0,16 / distancia):
  objetivo 0,40 (a 0,40 m), demasiado cerca 0,48.
- **Tilt de 105°:** es el que más mira hacia abajo de los probados. Es el que menos corta al guía
  de cerca y el que mejor ve el carril.
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
| **Terminal A** | Contenedor (`terminal.ps1`) | Puente ROS 2 ↔ simulador, **solo para las herramientas** de 3.4 y para el robot real | Mientras se usan las herramientas |
| **Terminal B** | Contenedor (`terminal.ps1`) | Nuestros scripts (pruebas, herramientas, entrenamiento) | Sí |

**`p3_entrenar.py` y `probar_entorno.py` lanzan su propio puente** (y lo relanzan tras cada
reinicio si hace falta). Antes de usarlos, **cerrad el puente de la Terminal A** (Ctrl+C): si
hay otro puente en marcha, el script se para y lo avisa. Las herramientas de comprobación
(`comprobar_puente.py`, `inspeccionar_*.py`, `medir_latencia_camara.py`) sí necesitan el puente
de la Terminal A.

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

**3) Terminal A: puente ROS 2 (solo para las herramientas de 3.4).** Para entrenar o evaluar
no hace falta: el script lanza el suyo.

```bash
ros2 run robobo_ros2 robobo_container --ros-args -p ip:=host.docker.internal \
    -p modules:="['camera','object_recognition','sim']"
```

Debe acabar mostrando `Connected to RoboboSim`, `SimNode started` y `Video stream connected`.
Si en algún momento deja de llegar imagen (por ejemplo, tras pulsar el reinicio del simulador),
**Ctrl+C y volved a lanzar este mismo comando**. **Cerradlo (Ctrl+C) antes de lanzar
`probar_entorno.py` o `p3_entrenar.py`.**

**4) Terminal B: carpeta de trabajo.**

```bash
cd /docker-ros/ws/RIA-robobo/p3_followcity
```

Todos los comandos siguientes se ejecutan en la **Terminal B**, desde esta carpeta.

### 3.4. Pasos de comprobación (hechos el 28/11; repetir si cambia el PC o el simulador)

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

Si la mediana es mayor que 1 s, avisad: habría que revisar `seguridad.periodo_control_s`.

**Paso 5. Probar el entorno completo** (unos 3–4 minutos). **Antes, Ctrl+C en la Terminal A**:
el script lanza su propio puente.

```bash
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 60
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 60 --politica aleatoria --sin-check
```

Debéis ver:
- `check_env: OK salvo determinismo (esperado en tiempo real)`;
- cada episodio empieza con `guia_visib=+1.00` y `carril_val` > 0,5;
- los pasos tardan ~0,25 s;
- en el simulador, **cada episodio empieza con la escena reiniciada**: los dos robots vuelven a
  la posición inicial y el seguidor arranca parado. Si sale `Tras el reinicio: ...; se relanza
  el puente`, es lo esperado: el puente se relanza solo y sigue.

Con la política recta es normal que acabe saliéndose del carril; lo importante es que no haya
errores y que los episodios no terminen en el primer paso.

**Paso 6. Revisar los registros**:

```bash
ls registros/pruebas/
column -s, -t < $(ls -td registros/pruebas/*/ | head -1)episodios.csv | less -S
```

`episodios.csv` debe tener una fila por episodio con `motivo_fin`, salidas de carril, contactos y
paradas. Si algo no cuadra, los registros de esta carpeta tienen todo lo necesario para
revisarlo.

### 3.5. Entrenamiento (parte 5)

Con el simulador en C4 sin pelota, todo en la **Terminal B** desde `p3_followcity`, y **sin
ningún puente lanzado en la Terminal A**: el script lanza el suyo y lo relanza cuando un
reinicio lo deja sin conexión. Así un entrenamiento largo no necesita a nadie delante.

**Sí: este es el entrenamiento en simulación que pide el enunciado** (SAC, RoboboCity, C4 sin
pelota). Cada episodio termina por colisión, guía perdido, carril perdido o tiempo, y el
siguiente empieza con la escena reiniciada a la posición inicial.

**1) Comprobar la observación** (no mueve el robot; Ctrl+C para salir):

```bash
python3 p3_entrenar.py --calibrar --perfil sim
```

Con el seguidor detrás del guía, debe salir `guía vis=1` y `carril val` > 0,5 con `ambos`.

**2) Prueba corta del script** (unos 5 minutos; crea `modelos/p3_sac_seed1`, que luego se puede
borrar):

```bash
python3 tests/test_entrenar_falso.py          # sin ROS: ciclo completo con un entorno falso
python3 p3_entrenar.py --entrenar --perfil sim --seed 1 --episodios-entrenamiento 3 \
    --pasos-demostracion 60 --guardar-cada 1 --episodios-evaluacion 1
```

Debéis ver los mensajes `Demostración heurística`, `episodio 1..3`, `[punto de control ...]`,
`Modelo guardado` y la `EVALUACIÓN TRAS ENTRENAMIENTO`. Es normal que estos episodios acaben
pronto (`contacto`, `carril_perdido`): el modelo apenas ha aprendido.

**3) Línea base heurística** (para comparar en la memoria; unos 15 minutos). Con `--imagenes 4`
se guarda 1 imagen de cada 4 pasos, con el carril y el guía dibujados, en
`registros/evaluacion/heuristica_.../imagenes/`, para ver qué percibe el robot en marcha:

```bash
python3 p3_entrenar.py --evaluar --politica heuristica --perfil sim --episodios 5 --imagenes 4
```

**4) Entrenamiento largo.** Unas 2–3 horas para 150 episodios de hasta 60 s. Se lanza en
segundo plano con `nohup`, para que siga aunque se cierre la terminal, y con la salida a un
fichero:

```bash
mkdir -p registros
nohup python3 p3_entrenar.py --entrenar --perfil sim --seed 2026 \
    --episodios-entrenamiento 150 > registros/entrenamiento_seed2026.log 2>&1 &
tail -f registros/entrenamiento_seed2026.log      # Ctrl+C deja de mirar, NO para el entrenamiento
```

Cada línea `episodio N` resume ese episodio: pasos, recompensa, motivo de fin, % fuera de
carril, contactos, paradas y separación media. Si los episodios se alargan y acaban por
`completado`, está aprendiendo.

**5) Si se corta** (corte de luz, el puente se cae, Ctrl+C...): se guarda un punto de control
cada 10 episodios y también al interrumpir. Para seguir donde se quedó:

```bash
python3 p3_entrenar.py --continuar modelos/p3_sac_seed2026 --episodios-entrenamiento 150
```

Para pararlo a propósito: `pkill -INT -f p3_entrenar.py`. Guarda antes de salir.

**6) Evaluar el modelo** entrenado (determinista, recorridos de 120 s como en el anexo B; en C4
el anexo pide dos recorridos como mínimo):

```bash
python3 p3_entrenar.py --evaluar --perfil sim --episodios 5
# otro modelo o punto de control:
python3 p3_entrenar.py --evaluar --perfil sim --modelo modelos/p3_sac_seed2026/modelo_ep0100.zip
```

**Dónde queda cada cosa:**

| Qué | Dónde |
|---|---|
| Modelo final | `modelos/p3_sac_seed2026.zip` (es el que usa `--evaluar` por defecto) |
| Puntos de control, buffer y estado | `modelos/p3_sac_seed2026/` |
| Pasos y episodios del entrenamiento | `registros/entrenamiento/sac_seed2026_entrenamiento_C4_<fecha>/` |
| Pasos y episodios de cada evaluación | `registros/evaluacion/<modelo>_evaluacion_C4_<fecha>/` |
| Curvas de TensorBoard | `p3_logs/sac_seed2026/` |
| Salida del puente (y sus relanzamientos) | `registros/<entrenamiento o evaluacion>/.../puente.log` o `registros/puente.log` |

> TensorBoard: el contenedor no publica ningún puerto, así que no se puede abrir en el
> navegador de Windows tal cual. Las figuras de la memoria se generarán a partir de
> `episodios.csv`, que tiene todo lo necesario (parte 6). Si queréis verlo en vivo, añadid
> `ports: ["6006:6006"]` al servicio en `docker-compose.yml` y lanzad
> `tensorboard --logdir p3_logs --bind_all`.

### 3.6. Robot real (más adelante, parte 7)

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

### 3.7. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| `Ya hay un puente robobo_ros2 en marcha` | El puente de la Terminal A sigue abierto | Ctrl+C en la Terminal A y relanzar el script |
| `El puente no responde tras 3 lanzamientos` | Simulador cerrado, fuera de Follow City o colgado | Abrir el simulador en Follow City (C4, sin pelota) y `--continuar` |
| `No llegan imágenes de .../image_raw` (con la Terminal A) | La cámara del puente se cayó (p. ej. tras un reinicio del simulador) | Ctrl+C en la Terminal A y relanzar el puente |
| `ModuleNotFoundError: robobosim` al lanzar el puente | Falta el paquete | `pip install --user --break-system-packages robobosim` |
| `No llegan las poses del simulador` | Puente sin el módulo `sim` | Lanzar con `modules:="['camera','object_recognition','sim']"` |
| El guía no aparece o lleva pelota verde | Ajustes de escena sin aplicar | Scene Settings → Mode3, sin Green Ball → Apply |
| `guia_visib` parpadea entre 0 y 1 | Detecciones más lentas que `antiguedad_max_deteccion_s` | Paso 2 y subir ese valor |
| Los episodios terminan enseguida con `carril_perdido` | La cámara no ve el carril al empezar | Abrid `registros/.../imagenes` (con `--imagenes 4`) y pasádmelas |
| `Perfil ... no válido` | Parámetro sin unidad o valor incoherente | Leer el mensaje: dice qué clave falla |
| `Ya existe modelos/p3_sac_seed2026` | Ya hay un entrenamiento con esa semilla | `--continuar` para seguirlo, o `--sobrescribir` para empezar de cero |
| El entrenamiento se para con `pérdida de datos ROS` | Se cayó la cámara o el puente | Relanzar el puente y `--continuar` |
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
├── sim_privilegiado.py           # SOLO sim: poses, reset_simulation, separación, rastro, paradas
├── puente.py                     # SOLO sim: lanza y relanza el puente robobo_ros2
├── registro.py                   # CSV de pasos y de episodios
├── controlador_heuristico.py     # controlador clásico: demostración y línea base (sin ROS)
├── p3_entrenar.py                # --calibrar / --entrenar / --continuar / --evaluar
├── modelos/                      # modelos SAC y puntos de control
├── p3_logs/                      # TensorBoard
├── registros/                    # pasos.csv y episodios.csv de cada ejecución
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
├── tests/test_sim_privilegiado.py
└── tests/test_entrenar_falso.py  # ciclo de p3_entrenar.py con un entorno falso (necesita SB3)
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

**Separación (anexo B):** distancia del centro del seguidor al punto 0,08 m detrás del centro
del guía (donde iba la pelota verde; la nota del anexo permite usar la posición del guía).
Banda objetivo 0,20–0,45 m; objetivo de la recompensa 0,32 m (= 0,40 m centro a centro).
**Desviación:** ángulo entre el rumbo del seguidor y la dirección a ese punto.

**Recompensa** (por paso, pesos en `recompensa`): separación (campana en 0,32 m) + centrado
en el carril − fuera de carril − guía no visible − |Δw| + premio por quedarse quieto con el
guía parado y cerca (C4) − penalización al terminar. En simulación usa la distancia y el
desplazamiento **verdaderos** (`usar_privilegiada`); la política nunca los ve.

**Fin de episodio** (sim y real, anexo B): colisión (IR frontal ≥ umbral o, en sim,
separación < 0,12 m), guía perdido (sin verlo 3 s o, en sim, separación > 0,90 m) o carril
sin validar 2 s → se detiene el robot y termina el episodio (no completado). Completado: llegar
al tiempo máximo (60 s entrenando, 120 s evaluando) sin nada de lo anterior.

**Reinicio.** En sim, `reset_simulation`: ambos robots a la posición inicial de la escena, el
seguidor parado y el guía esperando a que se acerque. Después se fija el tilt, se comprueba que
el robot responde y llegan imágenes (si no, se relanza el puente) y se espera a tener 3
imágenes seguidas con guía visible y carril válido. `simulacion.modo_reinicio = "colocar"`
deja el reinicio antiguo con `set_robot_location`, solo para depurar. En real se detiene el
robot y se espera a que el operador lo coloque.

**Verdad en simulación** (`sim_privilegiado.py`): el guía recorre el centro del carril, así
que su rastro de posiciones es la línea central. El desplazamiento lateral real es la
distancia con signo del seguidor a ese rastro (fuera de carril si supera 0,155 m). El guía se
considera parado (anexo B) si su velocidad media en 1 s está por debajo de 0,01 m/s durante
más de 0,5 s; la espera inicial antes de arrancar no cuenta como parada. Parada superada: no
hay colisión durante la parada y el seguidor sigue sin fallo 10 s después de que el guía
reanude.

**Registros** (`registros/<etiqueta>_<modo>_<C>_<fecha>/`):
- `pasos.csv`: marcas temporales, configuración, perfil, observación normalizada, acción y
  `cmd_vel` enviada, detección del guía, validez y salidas del carril, IR, recompensa,
  contacto, salida de carril (verdad y estimada) y verdad del simulador.
- `episodios.csv`: completado sin intervención, motivo de fin, % de pasos fuera de carril
  (verdad y estimada) y número de salidas, contactos, paradas del guía y paradas superadas
  (verdad en sim; estimadas por percepción en real), % de pasos en la banda 0,20–0,45 m,
  separación media y mínima, desviación media y % de pasos con desviación < 15°.

**Probar el entorno** (simulador abierto en C4 sin pelota; el script lanza el puente):

```bash
python3 herramientas/probar_entorno.py --perfil sim --episodios 2 --pasos 40
python3 herramientas/probar_entorno.py --perfil sim --politica aleatoria --sin-check
```

`check_env` de Gymnasium pasa todas las comprobaciones salvo la de determinismo, que no puede
cumplir un robot en tiempo real.

**Antes de entrenar en vuestro PC** (lo que mi entorno sin GPU no puede medir):
1. `medir_latencia_camara.py --perfil sim`: si la latencia es > 0,5 s, revisar
   `periodo_control_s`.
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
- **`reset_simulation` corta las conexiones del puente** (control y vídeo; los topics de IR y
  detecciones siguen republicando el último valor). `camera_node` intenta reconectar con el
  socket viejo ("Transport endpoint is already connected"). El entorno lo resuelve relanzando
  el puente tras el reinicio cuando no responde (`puente.py`).
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
