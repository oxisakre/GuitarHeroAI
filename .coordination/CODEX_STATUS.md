# Estado de Codex

Actualizado: 2026-09-22

## TERMINADO: práctica PPO de sostenidos múltiples, registro 220640

**Todos los archivos LIBRES. Ningún entrenamiento sigue corriendo.**
Modelo recomendado para probar: **`IA_guitarristaAcordes.zip`** (último paso de
la segunda tanda). `IA_guitarristaCanciones.zip` sigue intacto. No modifiqué
visión, seguimiento, control de teclas, calibración ni los modelos existentes.

### Diagnóstico reproducido

El usuario confirmó el comando con `--model IA_guitarristaCanciones.zip
--control --debug`. El registro `debug/20260922_220640/eventos.jsonl` tiene
14.615 cuadros. Hay 33 keyDown en carriles vacíos de nuestra fila 19 mientras
hay al menos dos colas. Las acciones PPO registradas piden esas teclas: no es
el controlador inventando pulsaciones ni prueba de otro fallo de OpenCV.
Ejemplo a los 41,34 s desde el primer cuadro: fila19 `[0,2,2,0,0]`, held estimado
`[0,1,1,0,0]`, PPO pide `[1,1,1,0,0]`: agrega verde durante roja+amarilla.
La imagen correspondiente es `pista_25.png`. A los 42,13 s agrega roja durante
amarilla+azul. Consultar el mismo modelo con esos tableros reproduce ambos.

Reconstruí los tableros completos desde `detalles_seguimiento`, `observed`,
altura 800 y latencia 30 ms: **0 discrepancias con fila19 en los 14.615 cuadros**.
La memoria held se estima con las acciones registradas; el log no guarda held
ni si hubo nueva inferencia. Las acciones recalculadas coinciden en el 99,877%
de los cuadros. NO es un replay exacto del teclado ni una medición de aciertos
del juego. Contar solicitudes por cuadro NO equivale a contar pulsaciones.

### Cambios y entrenamiento

- `main.py`: nuevo modo optativo `note_style="sustain"`: mezcla episodios
  normales con práctica de acordes de 1-4 teclas (predominan 2-3), mayor
  frecuencia de sostenidos y colas de 4-40 filas. Las colas de un acorde de
  práctica comparten duración. Conserva silencios y las recompensas originales.
  Los modos `song` y `dense` existentes conservan su generación.
- `train.py`: acepta `--note-style sustain` y `--learning-rate` opcional,
  aplicado tanto a modelos nuevos como al reanudar. Sin el argumento conserva
  el comportamiento anterior.
- `play.py`: permite visualizar el modo `sustain`.
- `tests/test_environment.py`: reproducibilidad y continuidad de colas incluyen
  el nuevo modo; regresión nueva verifica acordes largos, silencios, episodios
  normales y que todas las notas/sostenidos se pueden tocar correctamente.
- `examples/evaluate_live_policy.py`: comparación sin teclado real sobre
  observaciones reconstruidas del log y simulaciones con semillas fijas. Falla
  si fila19 reconstruida no coincide (altura/latencia erróneas). Explicita los
  límites del replay; `--output` guarda el informe JSON.
- `README.md`: comandos de práctica, comparación y del modelo recomendado.

Dos entrenamientos acotados de 100.000 pasos pedidos (100.352 efectivos por
bloques PPO), partiendo ambos del modelo del usuario, semilla 6109, gamma 0.5,
evaluación cada 20.000 pasos en tres episodios. Se usó un hilo CPU.

1. `IA_guitarristaSostenidos*.zip`: tasa original 0.0003, ~62 s. **Experimental,
   no recomendado**: el último paso mejora el log, pero empeora otras canciones.
   Los checkpoints y resultados se conservaron para auditar.
2. `IA_guitarristaAcordes*.zip`: tasa **0.00003**, ~59 s. Mejora consistente en
   log y otras semillas. Se recomienda `IA_guitarristaAcordes.zip`, aunque el
   entrenador por recompensa imprime automáticamente el sufijo `_mejor`.
   Ambos se midieron; el último tiene menos solicitudes indebidas en este log.

Reproducir la segunda tanda con otro nombre de salida:

```powershell
./.venv/Scripts/python.exe train.py --resume IA_guitarristaCanciones.zip --note-style sustain --steps 100000 --seed 6109 --eval-episodes 3 --learning-rate 0.00003 --output IA_practicaNueva
```

### Validación

En el replay, modelo original → `IA_guitarristaAcordes.zip`:

- Solicitudes de teclas vacías: **129 → 8**; durante colas múltiples: 127 → 8.
- Cabezas no solicitadas: **0 → 0**; colas con held=1 soltadas: **0 → 0**.
- Estos conteos usan los mismos tableros y held reconstruido, no el puntaje.

Simulación final con semillas **20260922..20260929**, ocho episodios de 1000
pasos por modo, distintos de entrenamiento y selección (original → nuevo):

| Métrica | Canciones normales | Modo de práctica |
| --- | --- | --- |
| Cabezas perdidas | 11 → 7 | 26 → 5 |
| Pulsaciones indebidas | 100 → 33 | 324 → 35 |
| Sostenidos cortados | 2 → 2 | 8 → 2 |
| Ticks de cola perdidos | 17 → 17 | 141 → 24 |

Resultados guardados: `acordes_validacion.json`, `acordes_comparacion.json`,
`sostenidos_baseline.json`, `sostenidos_comparacion.json`,
`sostenidos_checkpoints.json`, todos en `.coordination/`. Salidas de entrenamiento:
`acordes_training.txt` y `sostenidos_training.txt`. No hubo entrenamiento con
acciones etiquetadas del registro ni reglas que filtren las acciones de PPO.

`python -B -m unittest discover -s tests -q`: **87/87**, 6,424 s. `git diff
--check`: sin errores (solo advertencias LF/CRLF). Sin pantalla en vivo ni teclas
reales; puntaje real pendiente. No prometo que todos los errores estén resueltos.

### Para Claude: activación del poder, todavía pendiente

El usuario también informa un fallo inmediatamente al activar el poder. No dio
el segundo de activación. Las capturas `soltada_46..48` muestran poder ya activo
(122,3–125,9 s desde el inicio del log), `soltada_49` ya muestra colores normales
(128,8 s). No está demostrada la causa de esa transición. Las primeras 60
imágenes de repulsas se agotan a los 58,7 s; no hay una secuencia de capturas
completa del cambio. No ajusté umbrales ni asociación sin una reproducción.
Para continuar, pedir el momento aproximado o una grabación del cambio de color
y comprobar ocultamientos, saltos de posición y tiempos de captura. El modelo
ve celdas 0/1/2, no colores: activar poder no cambia directamente su entrada,
salvo por cambios en lo que detecta/proyecta la visión.

Comando recomendado al usuario:

```powershell
./.venv/Scripts/python.exe guitar_flash.py --model IA_guitarristaAcordes.zip --control --debug
```

---

## Histórico: terminada la mejora visual de sostenidos

El usuario autorizó implementarla y pidió dejar una entrega completa antes de
agotar el contexto. Implementación y validación terminadas; no queda trabajo
en curso. **Todos los archivos de esta tanda quedan LIBRES.**

### Archivos modificados

- `guitarflash/vision.py`: separación del filtro de cabezas y la recuperación
  visual de colas confirmadas, tolerancia de grosor y huecos pequeños.
- `guitar_flash.py`: `procesar_captura` comunica al detector los carriles que
  ya tienen una nota confirmada con cola.
- `tests/test_vision.py`: tres regresiones de grosor y separación de colas.
- `tests/test_sustain_effects.py`: archivo nuevo, seis pruebas de integración
  de brillo, sostenidos y rechazo de otras figuras.
- `.coordination/CODEX_STATUS.md`: esta entrega.

### Comportamiento nuevo y límites

1. `max_tail_width_ratio` pasa de .14 a .18 por defecto (14 a 18 px cuando cada
   carril mide 100 px). Las cabezas mantienen sus umbrales anteriores.
2. `_tail_runs` une hasta dos filas vacías dentro de una cola uniforme, sin
   atravesar una cabeza ni un hueco grande. No inventa píxeles en la máscara:
   conserva el hueco para las comprobaciones del seguimiento.
3. `detect(..., sustain_lanes=())` admite contexto opcional de carriles 0 a 4.
   Se obtiene de `track.moving` y `track.tail_top is not None`, no de una simple
   mancha detectada ni de haber solicitado una tecla.
4. Sólo en esos carriles se añade a la máscara de colas una detección con
   saturación mínima 70 y brillo mínimo 55, grosor hasta el 24% del carril y
   longitud mínima del 8% de la altura (64 px por defecto). Debe pasar cerca del
   centro, conservar grosor uniforme y no atravesar las bandas de cabezas.
   Esto tolera que el sostenido se vuelva más pálido, oscuro o ancho al tocarlo.
5. El filtro tolerante **no crea cabezas ni atribuye cola a una cabeza nueva**.
   La identificación inicial usa el filtro estricto y el ancho de cola de .18.
   Guías totalmente grises/blancas y trazos cortos pálidos quedan fuera.

Los nuevos umbrales opcionales de configuración son
`active_tail_saturation_min` y `active_tail_value_min`. No cambié el archivo de
calibración, el modelo, la latencia, la asociación de notas ni el control de
teclas. No hay nuevos argumentos obligatorios de consola: basta reiniciar el
programa con el comando habitual para cargar la nueva versión.

### Comparación con imágenes reales

Se guardó el baseline anterior en memoria y se compararon 102 imágenes
rectificadas (`pista_*.png` y `soltada_*_pista.png`) de `debug/20260918_031645`:

- **660 cabezas antes y después, exactamente los mismos carriles y posiciones.**
- Cabezas con cola: 16 a 18. Las dos asociaciones nuevas se inspeccionaron:
  cola azul real en `soltada_07_pista.png`, naranja real en `soltada_38_pista.png`.
- En `soltada_11_pista.png`, las colas de carriles 2 y 4 pasan de 13 y 10
  segmentos a **un segmento continuo cada una**.
- Filas activas de cola: 3513 a 4167; segmentos totales: 67 a 49. Estos totales
  NO son una medida de precisión: no se etiquetaron visualmente todos los
  segmentos nuevos y cortos. La igualdad de cabezas es respecto al baseline,
  no una certificación de que las 660 sean todas notas verdaderas.

### Pruebas y pendientes

`python -B -m unittest discover -s tests -q`: **75/75 aprobadas**, 4,760 s.
`git diff --check` sin errores. Sin pantalla en vivo ni teclas reales.

Las regresiones verifican una cola que varía de 11 a 18 px; huecos de dos filas;
separación de segmentos distantes; un sostenido de 10 s que cambia de 12 px y
color vivo a 20 px y S=95 al llegar a la línea, manteniéndose hasta su final;
menor brillo; carriles vecinos sin tolerancia; guías blancas/grises y trazos
pálidos cortos de cartel; ausencia de notas nuevas a partir de figuras pálidas.

**Pendiente real:** medir puntaje con misma canción/dificultad/poder. No se
implementó OCR ni reconocimiento general de carteles; un cartel grande todavía
puede ocultar notas. Tampoco se garantiza seguir una cola completamente blanca
o una cola que ya nace fuera del filtro estricto y nunca se confirma. La mejora
demostrada es la continuidad de colas en estos casos, no recuperar los 49k.
Las futuras pruebas deberían guardar el log versión 2 (`--debug`) de esta
ejecución; el último registro disponible es de la versión anterior.

---

## Revisión que motivó la mejora (histórico)

Revisión terminada. No se aplicaron nuevos cambios de código en esta tanda;
sólo se actualiza este informe. El usuario compartió capturas de carteles,
llamas y notas con el poder activado, y preguntó por usar una versión más simple.

Hallazgo en `debug/20260918_031645/soltada_11_pista.png`, ya rectificada: las
colas de los carriles 2 y 4 varían entre 11 y 15 px de ancho en filas 219:680.
`NoteDetector` admite hasta 14 px con la configuración predeterminada
(`max_tail_width_ratio=.14`). Rechaza por ese límite 34 y 63 de esas 461 filas,
respectivamente. El resultado contiene 13 y 10 segmentos de cola. Una prueba
sólo en memoria con `.18` produce un segmento continuo por carril en la misma
imagen. Es evidencia de fragilidad del grosor, NO prueba de que explique la
soltada o la caída de puntaje. No se guardó ese ajuste en la configuración.

Prueba de color: aumentar sólo el brillo V de 150 a 255 con saturación 255
conserva una cola de 12 px; bajar la saturación a 100 (blanquearla) o el brillo
a 80 elimina la cola. No hay que confundir "más brillante" con "menos saturada"
o "más gruesa". El filtro actual requiere S>=130 y V>=90 para cabezas y colas.

Prioridad sugerida: tolerancia al grosor y a huecos de una cola ya identificada,
comprobada con imágenes reales antes/después de mantenerla, sin relajar a ciegas
el filtro global de cabezas. El cartel puede ocultar notas; comprobar su efecto
en una secuencia, sin deducir pulsaciones falsas a partir de una sola foto.
Las llamas de la cuarta captura aportada no demuestran por sí solas que estén
fuera de la banda excluida: con rectificación aproximada sus puntas quedan
dentro de esa banda. El porcentaje de exclusión se aplica después de rectificar.

El registro más reciente disponible, `20260918_031645`, todavía usa el formato
anterior (sin `captured`, detalles de cola ni `version: 2`), de modo que no
confirma el rendimiento de la última entrega de Codex. Para comparar versiones,
recalibrar cada una y mantener canción, dificultad y uso del poder iguales.
Una pista opaca con menos efectos facilita la percepción; el nombre o antigüedad
de la versión por sí solos no garantizan mejor resultado.

---

## Entrega anterior de código (2026-09-18)

## Estado actual: tanda terminada, con cambios de código autorizados

El usuario pidió mejorar el código después de observar una caída de puntaje.
Leí la cuarta tanda de Claude y conservé sus correcciones de capturas idénticas
y asociación ordenada. No cambié el modelo, los umbrales de color, la latencia
ni los márgenes de asociación de cabezas.

### Archivos modificados y liberados

- `guitarflash/tracking.py`: recuperación acotada de una cola recortada arriba
  después de una interrupción de detección.
- `tests/test_tracking.py`: cinco regresiones de recuperación y finalización.
- `guitarflash/debug.py`: tiempos exactos, datos completos de detecciones,
  estado de notas/colas y filas de cola por carril, conservando campos anteriores.
- `guitar_flash.py`: pasa `captured` y `result.tail_mask` al registro.
- `tests/test_debug.py`: prueba de reconstrucción a partir de 140 cuadros del log.
- `.coordination/CODEX_STATUS.md`: esta entrega.

**Todos estos archivos quedan LIBRES. No queda trabajo de Codex en curso.**

### Fallo corregido: la cola no se recuperaba tras desaparecer 350 ms

Es distinto del descarte de capturas idénticas ya corregido por Claude: aquí
`tracker.update` se ejecuta en todos los cuadros. Al perder la máscara durante
350 ms, una cola recortada en y=0 acumulaba un desplazamiento estimado de 150 px.
Cuando reaparecía, la puerta de 100 px la rechazaba para siempre.

Reproducción a 400 px/s: últimas detecciones de cabeza en
`(t,y)=(0,680),(.025,690),(.05,700),(.075,710)`, cola en filas 0:704 del carril 4;
máscara vacía entre t=.5 y t=.85 y nuevamente visible después. Antes, la fila19
pasaba a vacío en **t=2.575 s**, aunque la cola continuaba visible. Ahora mantiene
el sostenido durante los casi 10 s de esa prueba y conserva su identidad.

Implementación: cada Track guarda `last_tail_seen` y `tail_clipped`. Un extremo
recortado (y<=2) se recupera sólo si también estaba recortado antes, su última
observación está dentro de `max_unseen` (0,45 s por defecto), y su segmento
continúa hasta al menos el 75% de la altura. Así no se ancla la nota vieja a una
cola corta de otra nota que entra arriba. Para extremos conocidos, dentro de
la imagen, se conserva el margen geométrico habitual.

Pruebas nuevas: recuperación tras 350 ms; rechazo después de 500 ms a 100 y
400 px/s; final normal de la cola recuperada; rechazo de una cola nueva corta;
rechazo de segmentos desconectados. Este arreglo no pretende tolerar oclusiones
indefinidas ni demostrar qué provocó la caída de 49k a 39k en el juego real.

### Registro versión 2, activado con el mismo --debug

Se mantienen `t`, `detecciones`, `notas`, `enviadas`, etc., para no romper los
analizadores existentes. Se añaden:

- `captured` y `observed`: instantes exactos de captura y cálculo del tablero;
  el antiguo `t` sigue redondeado.
- `velocidad_exacta` y `detalles_detecciones`: carril (1 a 5), y sin redondear,
  altura, indicador de sostenido y extremo de cola.
- `detalles_seguimiento`: id, carril, posición, última observación, velocidad,
  confirmación, extremo de cola, `last_tail_seen` y `tail_clipped`.
- `colas_por_carril`: cinco listas de intervalos de filas `[inicio, fin)` con
  cola. Representan exactamente la presencia de píxeles que consulta el tracker
  en la franja central de cada carril. `None` significa dato no registrado;
  listas vacías significan ausencia observada de cola.

La prueba reconstruye detecciones y máscaras a partir del JSON de 140 cuadros
con posiciones fraccionarias y una oclusión: obtiene los mismos tableros,
identidades y velocidades usando la misma configuración inicial. No se guarda
video ni se leen aciertos del juego; los registros viejos siguen teniendo las
limitaciones descritas en la revisión anterior.

### Validación y siguiente comparación

`python -B -m unittest discover -s tests -q`: **66/66 pruebas aprobadas**, en
3,538 s. Se usó el Python incluido en Codex con las dependencias de `.venv`.
`git diff --check` sin errores. Pruebas sin teclado real ni ventanas.

Para comprobar el puntaje, hace falta otra partida comparable con la misma
canción, dificultad, calibración y uso del poder. El comando habitual con
`--debug` sirve; no cambiaron sus opciones. No afirmo que este cambio recupere
los 49k: el efecto demostrado es evitar la pérdida de una cola en el caso
reproducible y mejorar los datos para investigar los fallos restantes.

---

## Historial: revisión anterior, sin cambios de código

Esta actualización documenta la revisión posterior a la tercera tanda de Claude
y a la vuelta de la ventana inicial de 120 a 64 px. El usuario pidió revisar y
comentar, y ahora autorizó dejar estos hallazgos en el buzón.

**Único archivo editado por Codex en esta entrega:** `.coordination/CODEX_STATUS.md`.
No se modificaron código, pruebas, configuración ni modelos. No hay archivos de
implementación reservados por Codex ni trabajo suyo en curso.

Validación independiente: `python -B -m unittest discover -s tests -q`:
**58/58 pruebas aprobadas**, en 2,706 s. Se usó el Python incluido en Codex con
`PYTHONPATH=.venv/Lib/site-packages` y escritura de bytecode desactivada.
Las reproducciones adicionales se ejecutaron por stdin, sin guardar scripts,
capturar pantalla ni enviar teclas reales.

### 1. Sostenido largo: saltarse imágenes idénticas pierde una cola visible

En `guitar_flash.py:239-242`, si la captura coincide con la anterior, tampoco se
ejecuta `tracker.update`. Una vez que la cabeza salió del recorte, una cola larga
que todavía entra por arriba puede producir una imagen idéntica durante segundos.
Su extremo visible necesita seguir anclado al borde superior mediante
`guitarflash/tracking.py:166-174`; en cambio, `observation` lo extrapola hasta
salir del tablero y hace desaparecer el sostenido.

Reproducción integral con imágenes sintéticas, detector y seguimiento reales:
velocidad 400 px/s, cola de 4000 px (10 s), capturas a 40 Hz, fondo inmóvil.
La cabeza cruza la línea aproximadamente en t=1,7475 s; el final verdadero de
la cola sería t=11,7475 s. Con el descarte actual desaparece del tablero en
**t=3,925 s**, unos 2,2 s después del inicio del sostenido. Actualizando el
seguimiento también en las imágenes idénticas, permanece visible hasta t=6 s,
que fue el final de esta comparación.

```python
import cv2
import numpy as np
from guitarflash.vision import NoteDetector
from guitarflash.tracking import BoardTracker

corners = [(0, 0), (499, 0), (499, 799), (0, 799)]
for skip_identical in (True, False):
    detector = NoteDetector()
    tracker = BoardTracker(initial_speed=400)
    last = None
    seen_hold = False
    release_at = None
    for i in range(241):
        t = i / 40
        y = 100 + int(400 * t)
        frame = np.full((800, 500, 3), 15, np.uint8)
        cv2.line(frame, (350, y - 4000), (350, y), (240, 0, 0), 7)
        cv2.ellipse(frame, (350, y), (30, 11), 0, 0, 360, (240, 0, 0), -1)
        if not skip_identical or last is None or not np.array_equal(frame, last):
            last = frame.copy()
            result = detector.detect(frame, corners)
            tracker.update(result.heads, result.tail_mask, t)
        board, _ = tracker.observation(t)
        seen_hold |= board[19, 3] == 2
        if seen_hold and board[19, 3] == 0 and release_at is None:
            release_at = t
    print(skip_identical, release_at, int(board[19, 3]))
# True: 3.925, 0; False: None, 2.
```

Sugerencia: separar la supresión de detecciones repetidas de la actualización
temporal de colas. Agregar una regresión del bucle con una cola visible y capturas
idénticas. No basta con el test que llama a `update` en cada paso.
Esto demuestra el fallo bajo esa condición; no confirma que sea la causa del
sostenido de 10 s del usuario, cuyo registro útil sigue pendiente.

### 2. Ráfagas: ocho notas todavía pueden producir nueve llegadas

Reproducción con código actual: 8 notas separadas 110 px, velocidad 1700 px/s,
juego a 30 FPS, captura a 60 FPS con jitter de ±8 ms, semilla 15 e inicialización
a 400 px/s. Resultado: **9 identidades llegando a la fila de toque**.

```python
import numpy as np
from guitarflash.tracking import BoardTracker
from guitarflash.vision import Detection

tracker = BoardTracker(initial_speed=400)
rng = np.random.default_rng(15)
mask = np.zeros((800, 500), np.uint8)
arrivals = set()
for i in range(90):
    t = i / 60 + float(rng.uniform(-.008, .008))
    base = 30 + 1700 * int(t * 30) / 30
    heads = [Detection(1, base - k * 110) for k in range(8)
             if 0 <= base - k * 110 < 704]
    tracker.update(heads, mask, t)
    _, tokens = tracker.observation(t)
    arrivals.update(token for token in tokens if token is not None)
print(len(arrivals))  # 9; hay 8 notas reales.
```

En t=0,356538 s, una pista aún no confirmada tiene una velocidad estimada de
2634,54 px/s a partir de sólo dos posiciones. Su predicción gana por cercanía y
se queda con la cabeza de la nota vecina: salta de y=100 a y=266,67. Se crea otra
identidad para su propia cabeza. Revisar el uso de velocidad antes de confirmar
la pista (`tracking.py:82`) y la asociación por distancia (`:106-118`).

Restaurar 64 px no resuelve este caso. Además, `max(25, 64, speed*elapsed*2.5)`
no impone un máximo de 64 px. Tampoco basta con cambiar `max` por `min`: aquí
gana una predicción prematura con un error de sólo 19 px. Hace falta probar la
combinación de ráfagas y jitter, preservando el orden y la identidad de las notas.
El test de una sola nota con seis semillas no cubre esta situación.

Limitación: es una reproducción sintética; no demuestra que explique las 70
erradas de la partida real.

### 3. Cartel «1000 Notas Acertadas» y límites del diagnóstico

El usuario aportó un recorte del cartel sobre la pista. En ese recorte las letras
claras quedan mayormente por debajo del mínimo de saturación 130. Esto no prueba
que el cartel completo sea inocuo: puede tapar notas y su aparición/desaparición
puede interferir en el seguimiento. Falta una secuencia antes/durante/después
para confirmar si genera falsos positivos, pérdidas o cambios de identidad.

Verifiqué `debug/20260917_212808/eventos.jsonl`: 2422 pulsaciones; en la fila19
registrada, 2409 tienen cabeza, 11 cola y 2 vacío. Esos números son correctos,
pero **fila19 es la estimación del propio sistema, no la verdad del juego**.
No permiten afirmar que los destellos ya no generan notas falsas, que las 70
erradas sean todas de tiempo, ni que sólo se hayan perdido dos notas reales.
Igualmente, 2422 = 2352 aciertos + 70 erradas no demuestra una pulsación por cada
nota física: un total puede contener pulsaciones duplicadas y notas omitidas.

Los replays también siguen limitados: el registro no guarda `captured`, si se
ejecutó `tracker.update`, `tail_top` ni la máscara de colas. No reproducen
exactamente el bucle ni los sostenidos. Registrar esos datos ayudaría a comparar
cambios sin atribuir diferencias al seguimiento cuando cambiaron sus entradas.

Prioridad sugerida: corregir el caso de cola inmóvil, conservar el caso conjunto
de ráfagas+jitter como regresión y comprobar la aparición del cartel mediante
una secuencia real. No se aplicó ninguna de estas correcciones en esta revisión.

---

## Registro de la primera revisión (histórico, no estado actual)

Las reservas de archivos y cifras de pruebas que siguen corresponden a la
primera revisión. El estado vigente y los resultados nuevos están arriba.

ESPERANDO / REVISIÓN EN PARALELO SIN EDITAR CÓDIGO DE CLAUDE.

Claude tiene temporalmente la propiedad de:

- `guitarflash/tracking.py`
- pruebas y herramientas relacionadas con duplicados y registros reales

Codex no editará esos archivos mientras Claude marque el trabajo en curso.

## Lo que vi

Leí el cambio actual: el seguimiento exige movimiento consistente antes de
confirmar una nota y amplió el emparejamiento de notas ya confirmadas. La causa
descrita (error de predicción de 13 px frente a un margen de 12 px) es coherente
con la creación de un segundo `Track`.

La reducción de duplicados en registros reales es una señal útil, pero antes de
cerrar conviene medir también falsos negativos. Casos que sugiero conservar como
regresión:

- velocidad máxima y FPS mínimo reales: una nota con pocas observaciones no debe
  descartarse sólo porque no alcanzó varias muestras estables;
- dos notas próximas del mismo carril no deben fusionarse;
- acordes y sostenidos no deben perder identidad ni cortar la cola;
- una figura fija del fondo no debe alcanzar `moving=True`;
- la misma imagen repetida por diferencia entre FPS del juego y captura no debe
  duplicar ni inflar la velocidad;
- comparar `IDs confirmados que llegan a fila 19` contra notas reales conocidas,
  además del porcentaje de duplicados.

## Para Claude

Cuando termines esta tanda, escribí `.coordination/CLAUDE_STATUS.md` indicando:

- archivos que modificaste;
- comando de pruebas y resultado;
- métricas antes/después por cada registro real;
- cantidad de notas verdaderas perdidas, si el registro permite conocerla;
- cualquier revisión concreta que quieras que haga Codex.

Después marcá esos archivos como `LIBRES`. Codex hará una revisión independiente
y ejecutará la suite completa sin cambiar tu solución mientras siga EN CURSO.

## Verificación independiente

Ejecuté la suite completa sobre el estado actual con el Python incluido en Codex
y las dependencias locales:

`python -m unittest discover -s tests -v`

Resultado: **54 pruebas aprobadas de 54** en 2,713 s. No envié teclas reales ni
abrí la interfaz gráfica. `tracking.py` continúa reservado para Claude.

Nota de entorno: `.venv\\Scripts\\python.exe` devolvió `Unable to create process`;
las mismas dependencias funcionaron usando el Python incluido en Codex mediante
`PYTHONPATH=.venv\\Lib\\site-packages`. Esto no parece relacionado con el cambio
de seguimiento, pero conviene comprobar el comando normal antes de entregarlo.
