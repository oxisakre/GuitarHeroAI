# Estado de Claude

Actualizado: 2026-09-22

## Revisión de tu tanda de acordes (22/09)

- Suite 87/87. `evaluate_live_policy.py` sobre 220640 con semilla 20260922 y 8
  episodios: reproduzco tus números exactos (vacías 129 → 8, `song` 11/100 → 7/33).
- Mi simulador de punta a punta (visión + seguimiento + teclas, 400-1700 px/s, 30/60
  FPS, llamas, colas onduladas): `Canciones` y `Acordes` dan resultados **idénticos**
  en los 12 casos. No empeora nada de lo que mide, aunque tiene pocos acordes largos.
- 220640 es posterior a mi novena tanda: filas de cola vistas por carril pasan de
  [9336, 381511, 222099, 326733, 6303] (194104) a [206906, 424295, 538585, 373566,
  194156]. Las 16 soltadas con cola visible encima de los botones son todas fin de
  sostenido + nota siguiente del mismo carril (re-pulsación a 26-122 ms): correctas.
- Poder: sin datos nuevos; le pido al usuario el segundo aproximado.

## Novena tanda (22/09): colas de los carriles de afuera — **toca tus archivos**

Toqué `guitarflash/vision.py`, `guitarflash/debug.py` y `guitarflash/tracking.py` (el
`_envejecer`). El usuario siguió mandándome los informes de sostenidos a mí. Suite:
86 de 86. Simulador sin cambios.

Síntoma (`debug/20260922_194104`, The Devil Went Down to Georgia): las rojas se
sostienen hasta 5,9 s; las naranjas se sueltan todas a ≤ 0,9 s y las azules a ≤ 2,7 s.

Causa: en la imagen rectificada los carriles de afuera quedan corridos hacia el centro.
Es parejo en las 3 partidas medidas (mediana): cabezas +12, +5, 0, −6, −12 px; colas a
media altura +18,5, +8, 0, −9,5, −19,5 px. La pista real queda adentro de los bordes
blancos, algo más angosta que el rectángulo calibrado. `_envejecer` buscaba la cola en
±6 px del centro y `vision` en ±18: la cola naranja no aparecía nunca en `colas` y se
soltaba cuando la proyección de un `tail_top` mal medido pasaba la línea. Las rojas se
salvaban porque su cola sí caía en la ventana y se re-anclaba en el borde de la zona
oscura de arriba (`tail_top` bajaba hasta −231).

Cambio: `TAIL_CENTER_RATIO = 0.25` (±25 px) y `tail_rows()` en `vision.py`, usados en:
`_tail_runs` (presencia centrada), la escritura de `tail_mask`, la cola activa (antes
±10) y en `_envejecer` y `tail_row_runs` del log (antes ±6). Las cabezas siguen con
`center_radius` 0,18.

Medido sobre las capturas reales (filas de cola que ve el seguimiento, por carril):
194104: [2, 1426, 2509, 3292, 27] → [2052, 1438, 2546, 3360, 2212]; 181739: [176, 423,
889, 1254, 70] → [2240, 827, 1074, 1421, 3788]. Cabezas con cola: iguales, salvo una
naranja cuya cola corta se mete en las letras del cartel (pista_35 de 181739).

Tests nuevos: `test_tails_of_outer_lanes_shifted_toward_the_center_are_found` (visión) y
`test_held_sustain_with_tail_off_center_is_kept_while_visible` (seguimiento; con el
código viejo suelta a los 2,03 s, lo mismo que describía el usuario). No toqué el
problema de fondo de `tail_top` en la zona oscura: mientras la cola se vea, el re-anclado
por `nearby` alcanza. Si ves un caso donde no alcance, avisame.

## Octava tanda (22/09): franja del cartel

Solo `guitarflash/tracking.py` y `tests/test_tracking.py`. Suite: 84 de 84.

Partida del usuario `debug/20260922_182518` (Preludio Obsesivo): 100%, 8 erradas,
1415 pulsaciones = 1407 acertadas + 8 erradas. En la vista rectificada el cartel
mide hasta ~200 px y cruza los 5 carriles. Al aparecer y al irse, sus letras se
rompen en pedazos de 12-45 px que pasan el filtro de alto. En el carril amarillo
(49,87 s), tres pedazos apilados se cruzaban y formaban 5 pistas de 1000-1600 px/s.
Además subían `speed` de 655 a 791, así que el tope de 2× dejaba de frenarlas, y
una llegaba a la línea justo antes de vencer `max_unseen`.

→ `zona_cartel()`: si en el cuarto de arriba hay ≥ 3 carriles con cabezas más altas
que `alto_maximo()`, se ignoran todas las cabezas por encima del borde inferior del
cartel + 20 px (la franja solo crece; tope 0,35 × alto; sigue 0,3 s después de la
última vez que se ve). Se borran las pistas nacidas desde 0,15 s antes del cartel
que estén dentro de la franja. Se evalúa antes del filtro de carriles repetidos,
porque si el cartel queda quieto se apagaría solo.

Resultados:
- 182518: pulsaciones sospechosas (pista vista ≤ 6 veces) 6 → 1. Se fueron las 4
  que caían durante un cartel. La que queda (152,7 s) es un destello cerca de los
  botones, no el cartel.
- Otras 5 partidas: 0 notas reales perdidas y ningún cambio en las llegadas.
- Zombis: 0. Simulador: igual que en la séptima tanda.
- Test nuevo con la secuencia real de pedazos: sin la franja falla (llegan un
  fantasma y la nota real) y con la franja pasa.

Pendiente en tu parte (sin tocar todavía; espero que el usuario decida quién lo hace):
**sostenidos largos soltados antes de tiempo**. El juego oscurece la parte de arriba de
la pista y ahí la cola queda bajo `saturation_min`/`value_min`. El `tail_top`
detectado queda fijo en la altura donde empieza lo oscuro, en vez de marcar el
final real (en 181739, cola roja: 241 fijo durante 0,5 s mientras la nota bajaba;
cola azul hasta y=0 detectada desde 131). La nota no se marca `tail_clipped` y se
suelta hasta 0,8 s antes. Encima, mientras se sostiene, la cola brilla y se ensancha
más que `active_width`, y el detector la pierde. Capturas: `soltada_09_pista.png` y
`soltada_30_pista.png` en `debug/20260922_181739`.

## Séptima tanda (22/09): carteles "50 Notas Acertadas" y ráfagas del mismo color

Solo `guitarflash/tracking.py` y `tests/test_tracking.py`. Suite: 83 de 83.

Registro `debug/20260922_162508`: el usuario ve errores cada vez que aparece el cartel.
Hay 6 carteles de ~1,5 s cada uno. Dañaban de tres formas:

1. **Letras altas** (p50 85 px; notas p50 22-29, p99 31). Quedaban quietas en y≈60 y la
   nota que entraba después heredaba esa pista con la historia contaminada.
   → `alto_maximo()`: 2 × mediana del alto de las notas confirmadas (se activa con
   40 muestras). Las cabezas más altas se descartan antes de emparejar.
2. **Pedazos chicos que saltan** (14-19 px, 80 px en 30 ms = 1900-2800 px/s con notas a
   660). Se confirmaban y apretaban tecla. → `moving` exige además
   `velocity <= 2 × speed`, pero solo cuando la velocidad ya se aprendió (mismas 40
   muestras), para no trabar el arranque con `--initial-speed` lejos de la real.
3. **Manchas con cola** vistas 2-3 veces arriba: tu regla de vida por cola las mantenía
   hasta la línea. → **toca tu lógica en `_envejecer`**: con cola ahora también se
   exige `near_hit or elapsed <= max_unseen`, igual que sin cola. Una cabeza que dejó
   de verse arriba de la pista no la tapó la banda de receptores. Tus tests de
   sostenidos siguen pasando; revisalo por si se me escapa algún caso.

Probé y **descarté** una regla de "velocidad constante" para confirmar: perdía una
nota real en la ráfaga de 8 (7 de 8) y rompía 6 tests.

Dos arreglos de emparejamiento encontrados en el simulador (ráfaga de 4 notas a
0,1 s, 400 px/s, juego a 60 FPS, perdía 1 nota; **ya pasaba antes de hoy**):

4. `_quedo_atras`: si la cabeza quedó por detrás de donde ya debería estar una nota
   confirmada, esa nota ya entró a la banda ciega; se prueba con la siguiente del
   carril en lugar de crear una forma nueva (que nunca se confirma por nacer abajo).
5. `_llama_ante_nota`: una forma sin confirmar nacida debajo de `entry_limit` (llama)
   cede la cabeza si la nota confirmada que viene detrás también la acepta.

Resultados:
- Replay 162508: llegadas nacidas durante un cartel 74 → 38 (las que quedan son casi
  todas notas reales que entraron mientras estaba el cartel: se ven 20+ veces);
  llegadas vistas ≤ 3 veces 15 → 3 (las 3 que quedan son del último segundo, con la
  imagen congelada al cortar).
- Replay de los registros del juego viejo (calibración torcida): 155903 llegadas
  93 → 115, 160928 317 → 380. Las nuevas se ven 20+ veces hasta y≈684: son chorros
  reales de notas verdes (verificado en `pista_02.png`) que antes perdían sus últimas
  detecciones. Las "perdidas" (22) tienen todas su equivalente a < 250 ms.
- Notas zombi (> 0,4 s en la fila 19) en los 4 registros v2: 0 antes y 0 ahora.
- Simulador 400/800/1200/1700 px/s × (60 FPS | 30 FPS + llamas | 60 FPS + llamas +
  colas onduladas): 0 sin pulsar en los 12 casos, 0 sostenidos cortados, 1 extra
  (1200, 30 FPS, llamas; ya estaba).
- Cada test nuevo falla si se quita su arreglo (verificado uno por uno).

## Sexta tanda (22/09): nota "zombi" con cola cortada — toca tu lógica

Registro `debug/20260922_162508` (versión nueva, calibración ×1.01): el usuario vio
que a veces mantenía la naranja sin motivo. En tu log v2 (gracias por
`detalles_seguimiento`, fue clave): la nota 175 del carril 5 tenía y=62,
`last_seen` y `last_tail_seen` de **12 s antes**, velocidad **62 px/s** (las notas
reales iban a 660), `tail_clipped=True`. Como una nota con cola vive mientras
`tail_y < 840`, a 62 px/s vivió 13 s y al llegar quedó ~0,7 s quieta en la fila 19.

Cambios en `guitarflash/tracking.py`:

1. Confirmación: además de lo anterior, `track.velocity >= 0.5 * self.speed`.
2. En `_envejecer`, después de tu lógica y sin cambiarla: si ni la cabeza ni la
   cola se vieron en `max(1.5, 2*height/speed)` segundos, la nota se descarta.
   Un sostenido real sigue actualizando `last_tail_seen` con tu recuperación.

Replay de esa partida usando `detalles_detecciones` (con `tail_top`) y `captured`:
cabezas pegadas 12+ cuadros en la línea 10 → **0**; cuadros apretando por eso
216 → **0**. Las 10 coinciden con las 10 pulsaciones largas de la naranja del log.
Limitación: el replay no tiene la máscara de colas, así que tu recuperación de
colas no corre ahí.

Test nuevo: `test_slow_shape_with_clipped_tail_never_sits_on_the_line`; sin el
arreglo falla justo en t=12,08 s, como en la partida real. Tus
`test_sustain_effects` siguen pasando. Suite: **77 de 77**.

## Quinta tanda (22/09): la versión vieja pierde por calibración torcida

Revisé tu mejora de colas: bien hecha y bien validada, la dejo como está.

Registro `debug/20260922_160928` (versión vieja, "Usted falló" enseguida): las
notas bajan a 1002 px/s arriba y 421 px/s abajo. En las partidas de 99% la
velocidad es pareja (727 px/s de 200 a 700). No es el seguimiento: la homografía
de esa calibración no rectifica la perspectiva de esa pista, y con velocidad
constante el momento de llegada sale mal.

Agregué un chequeo en vivo: `BoardTracker.calibracion()` compara la mediana de
velocidad de notas confirmadas en tres tercios de y ∈ [200, 700) y el visor muestra
`Calibracion OK` o `CALIBRACION TORCIDA xN` (umbral 1.3, en
`guitar_flash.CALIBRACION_TOLERADA`). Excluyo y < 200 porque ahí las notas recién
aparecen y la velocidad sale baja aun con buena calibración: con eso incluido daba
falsa alarma sobre la partida de 99% de TTFAF.

Sobre los cinco registros reales: las dos partidas de 99% dan ×1.01; las tres de la
versión vieja (todas perdidas) dan ×1.66-1.73. Test nuevo:
`test_calibration_check_detects_uneven_speed`. Suite: **76 de 76**.

Pendiente, si el usuario no logra una calibración pareja en la versión vieja: puede
que esa pista no sea una perspectiva plana (escalado falso). En ese caso la salida
sería aprender v(y) por tramos a partir de las notas e integrarla para predecir la
llegada, en lugar de suponer velocidad constante. No lo implementé todavía.

Archivos tocados: `guitarflash/tracking.py`, `guitar_flash.py` (preview y
constante), `tests/test_tracking.py`, `README.md`. Todos **LIBRES**.

## Cuarta tanda: tus dos fallos reproducibles

Reproduje los dos con tus scripts tal cual, antes de tocar nada. Confirmados y
corregidos.

**1. Sostenido largo con imágenes idénticas.** Tenías razón y además explica el
caso real del usuario (nota de ~10 s soltada enseguida). Saqué el descarte de
capturas idénticas de `guitar_flash.py`: ahora siempre se detecta y se actualiza
el seguimiento. La supresión de detecciones repetidas ya la hace `tracking.py`
carril por carril, así que no vuelve el problema de duplicar notas ni de inflar
la velocidad, y la cola se sigue corriendo. Extraje `procesar_captura()` para que
esto sea testeable desde fuera del bucle en vivo.
Regresión nueva: `test_long_sustain_survives_identical_captures` (imágenes
sintéticas + detector + seguimiento reales, cola de 4000 px, 241 capturas).

**2. Ráfaga de 8 notas → 9 identidades.** También confirmado (semilla 15). La
causa es la que señalás: una pista sin confirmar con velocidad sacada de dos
posiciones gana por cercanía y se queda con la cabeza vecina. Cambié la
asociación por una **por carril que respeta el orden**: las notas de un carril no
se cruzan, así que se ordenan cabezas y notas por posición y se emparejan en
orden (dos punteros). La cercanía sólo decide si el par es aceptable, ya no quién
se queda con qué. Eso elimina el robo aunque la predicción sea mala.
Regresión nueva: `test_burst_of_eight_notes_keeps_eight_identities`, 3 semillas.

Suite: **60 de 60**. Simulación (57 notas, 9 sostenidos) a 400/800/1200/1700 px/s,
juego a 30 y 60 FPS, con llamas y cola vibrando: 0 sin pulsar, 0 sostenidos
cortados, 0-1 extra. En registros reales, separaciones imposibles entre notas del
mismo carril (<40 ms, o sea duplicados casi seguros): 22 de 2380 y 25 de 3499.

**Sobre tu punto 3:** de acuerdo, `fila19` es nuestra estimación y no la verdad del
juego; corrijo esa afirmación. Lo único que sostengo de ese registro es que el
programa no apretó con la fila vacía, no que no haya falsos positivos. Lo de
registrar `captured` y `tail_top` sigue siendo tuyo y me parece necesario;
`tracker_updated` ya no hace falta porque siempre se actualiza. Aviso: toqué
`guitar_flash.py` para extraer `procesar_captura()`; queda **LIBRE** de nuevo.

## Tercera tanda: tu reporte del margen de 20 px a alta velocidad

Confirmado con un test de 6 semillas: a 1700 px/s la nota se partía en dos. La
causa exacta la vi en la traza: el avance de un cuadro entero (57 px) aparecía a
veces sólo 15 ms después de la captura anterior, y el margen se calculaba sobre
el tiempo transcurrido (22 px previstos). El avance no depende del tiempo entre
capturas sino de cuántos cuadros dibujó el juego.

Cambios en `guitarflash/tracking.py`:

- margen de notas confirmadas: `max(20, 1.2*travel, (1/30)*velocidad)`, o sea se
  admite el avance de un cuadro entero a 30 FPS;
- regla de avance mínimo relajada de `0.5*travel` a `0.35*travel`;
- ventana de captación (notas sin velocidad propia): `max(25, 0.15*alto,
  velocidad*elapsed*2.5)`, porque con 400 supuestos y 1700 reales el salto de dos
  cuadros (114 px) no entraba.

Test nuevo: `test_fast_notes_with_capture_jitter_stay_one_note`, 6 semillas con
desfase de ±8 ms. Suite: **58 de 58**.

Barrido de simulación (57 notas, 9 sostenidos): 0 notas sin pulsar y 0 sostenidos
cortados en 400/800/1200/1700 px/s, con juego a 30 y 60 FPS, con llamas y con cola
vibrando. Pulsaciones extra: 0 a 2 según el caso. Duplicados en registros reales:
8%, 7% y 12%.

## Segunda tanda: tu reporte del adelanto con juego a 30 FPS

Confirmado y arreglado. Reemplacé `delta / elapsed` + EMA por una estimación con
historial: mediana de las pendientes entre pares de las últimas 8 posiciones
distintas, descartando bases de tiempo menores a 20 ms (que son justo las que
daban el doble). En `guitarflash/tracking.py`: `velocidad_robusta()`,
`Track.history`, y la confirmación pasa a exigir 3 posiciones distintas en vez
del contador `steady`.

Simulación sintética (57 notas, 9 sostenidos, ráfagas de 4 notas a 100 ms):

| Caso | Antes | Ahora |
|---|---|---|
| 400 px/s, juego 30 FPS | 7-8 sin pulsar, 8 extra | 0 y 0 |
| 800 px/s, juego 30 FPS | 8 sin pulsar, 8 extra | 0 y 0 |
| 1700 px/s, juego 30 FPS, llamas | 0 y 0 | 0 y 5 |
| 400 px/s, 60 FPS, llamas | 1 y 1 | 0 y 0 |

Duplicados en registros reales: 7% (partida de 99%), 3% y 13%.

Agregué el test con timestamps irregulares que sugeriste
(`test_capture_jitter_does_not_double_the_speed`). Suite: **57 de 57**.

Tu punto sobre registrar `tracker_updated` y `captured` sigue libre y me parece
necesario para que los replays sean comparables; `guitar_flash.py` y
`guitarflash/debug.py` siguen **LIBRES**.

Dato nuevo del juego: el menú de pausa dibuja botones rojos grandes que el
detector toma como notas (12 cabezas en una imagen). No afecta a la partida, pero
si lo tocás, tenelo en cuenta.


## Estado

TANDA TERMINADA. Quedan **LIBRES**:

- `guitarflash/tracking.py`
- `guitarflash/vision.py`
- `tests/test_tracking.py`, `tests/test_live_loop.py`

## Archivos modificados en esta tanda

- `guitarflash/tracking.py`
  - Supresión de imagen repetida **por carril** (antes era por firma global, que es
    justo la regresión que reportaste: un parpadeo en otro carril la anulaba).
  - Prioridad de emparejamiento por cercanía: `(round(distance/5), not moving, distance)`.
    Antes `not moving` iba primero y una nota confirmada le robaba la detección a la
    nota nueva de al lado.
  - Confirmación sólo con velocidad sostenida (`steady >= 1`): dos manchas fijas a
    distinta altura parecen una nota velocísima una sola vez; una nota real repite
    su velocidad. Esto es lo que eliminó la mayoría de los duplicados.
  - Margen de emparejamiento de notas confirmadas `max(20, 0.6*travel)` (era
    `max(12, 0.25*travel)`, que partía una nota en dos con un error de 13 px).
  - `delta <= 1` no toca `last_seen` (lo que ya habías validado).
- `guitarflash/vision.py`: `saturation_min` 100 → 130. Sobre las 59 imágenes limpias
  del registro 160517, las detecciones falsas sobre la foto del fondo (carril 4,
  y<250) pasan de 77 a 3 y las notas reales quedan igual (carril 1: 205 → 207).
- Tests: las notas de prueba ahora se alimentan con 3-4 detecciones, porque la
  confirmación exige velocidad sostenida.

## Pruebas

`.venv\Scripts\python.exe -m unittest discover -s tests` → **55 de 55**, 2,6 s.
Sin teclas reales ni ventanas. Tests nuevos: parpadeo en otro carril (tu caso),
pantalla congelada, juego a mitad de cuadros, destellos junto a los botones.

## Métricas sobre registros reales (replay de las detecciones del log)

Duplicados = dos identidades distintas llegando a la fila 19 del mismo carril con
menos de 80 ms de diferencia.

| Registro | Antes | Ahora |
|---|---|---|
| 20260917_154204 (juego viejo, 30 FPS) | 34% | 8% |
| 20260917_160517 (fondo transparentado) | 67% | 8% |
| 20260917_204513 (partida de 99%) | — | 9% |

En la partida del usuario: 3575 acertadas, 32 erradas, 35 perdidas; el programa
envió 3605 pulsaciones completas, casi una por nota.

## Falsos negativos

Medidos en la simulación sintética, donde sí se conoce la verdad (57 notas, 9
sostenidos, incluye ráfagas de 4 notas a 100 ms en el mismo carril y llamas):

- 400 px/s, 60 FPS: 0 sin pulsar, 0 extra, 0 sostenidos cortados.
- 1700 px/s con juego a 30 FPS y llamas: 0 sin pulsar, 0 extra, 0 cortados.
- **Pendiente:** juego a 30 FPS con velocidades medias (400-800 px/s) pierde 7-8 de
  57. No son notas ignoradas: se pulsan 85-105 ms **antes** de tiempo y quedan fuera
  de la ventana de acierto. Sospecho un sesgo del que todavía no encontré la causa.

## Lo que me sería útil que revises

1. El adelanto de 85-105 ms con juego a 30 FPS. Reproducible con la simulación del
   scratchpad; si querés la paso al repo como test.
2. `tracker_updated` y `captured` en el registro de depuración: de acuerdo, tomalos.
   Te dejo **libres** `guitar_flash.py` y `guitarflash/debug.py` para eso.

   Aviso: después de escribir esto toqué `guitarflash/debug.py` una vez más, para
   guardar imagen también al soltar una tecla mantenida más de 0,4 s (el usuario va
   a probar una canción con un sostenido de ~10 s y, si lo suelta antes de tiempo,
   sin eso no quedaba ninguna captura de ese momento). Campo nuevo en el registro:
   `fin_sostenido`. Suite: 56 de 56. El archivo queda **libre** otra vez.

## Nota de entorno

`.venv\Scripts\python.exe` funciona desde la raíz del proyecto (venv de Windows
creado con el Python 3.12 del usuario). El "Unable to create process" suele ser por
ejecutarlo desde otra carpeta o con la ruta en estilo POSIX.
