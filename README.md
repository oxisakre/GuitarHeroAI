# IA para Guitar Flash

PPO aprende en un simulador de cinco carriles. OpenCV transforma la pantalla
en observaciones para ese modelo; el adaptador de teclado ejecuta las acciones
que elige PPO. Se usa **Sólo tocar**. La activación del poder es manual.

**Estado actual:** integración inicial con calibración, detección, seguimiento,
vista previa y control de teclas. Se verificó con imágenes y pruebas sintéticas;
todavía falta medir su rendimiento en una canción real. La inferencia en el
navegador **no entrena** el modelo. El programa no lee el puntaje de Guitar Flash.

## Instalación (Windows, Python 3.10 o posterior)

Desde esta carpeta:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

El `.venv` usa el Python 3.12 instalado en Windows. Si se rompe o cambiás de
Python, borrá la carpeta `.venv` y repetí estos dos comandos. Los comandos de
abajo usan ese entorno sin necesidad de activar scripts de PowerShell.

## 1. Calibrar la pantalla

Abrí Guitar Flash en el navegador y mantené fijo su tamaño y zoom. Usá el modo
Sólo tocar. La calibración funciona sobre una captura congelada:

```powershell
.\.venv\Scripts\python.exe guitar_flash.py --calibrate --keys asdfg
```

Después de ejecutarlo tenés 5 segundos para dejar el juego a la vista, sin la
consola encima (`--calibrate-delay 10` da más tiempo). La captura se abre
reducida en la ventana `Calibracion 1/2`, con las instrucciones arriba:

1. Arrastrá un rectángulo alrededor de la pista completa y confirmá con Enter.
2. En `Calibracion 2/2`, marcá cuatro puntos en orden: borde superior izquierdo,
   superior derecho, extremo derecho de la línea de toque y extremo izquierdo
   de esa línea.
3. Los puntos inferiores deben estar a la altura del **centro de los botones**,
   en los límites externos de los cinco carriles. Los superiores deben abarcar
   los mismos cinco carriles en una zona visible de la pista.
4. Enter guarda `guitarflash_config.json`. R reinicia. Escape, cerrar la ventana
   o Ctrl+C en la consola cancelan. Si los puntos no forman una pista válida, el
   motivo aparece arriba y podés marcarlos de nuevo con R.

`--monitor 2` permite elegir otro monitor. `--keys` recibe las cinco letras o
números configurados en el juego, en orden verde, rojo, amarillo, azul, naranja.
No se cambia la configuración del juego automáticamente. Si cambia la posición,
el tamaño o el zoom del navegador, volvé a calibrar.

## 2. Comprobar la detección

```powershell
.\.venv\Scripts\python.exe guitar_flash.py
```

Ubicá el visor **fuera de la región capturada**. Muestra la pista rectificada,
las cabezas candidatas y la matriz: blanco = cabeza, verde = cola. El seguimiento
necesita varias capturas en movimiento, por lo que una imagen pausada no activa
notas en la matriz. La banda de receptores se excluye del detector; las notas se
proyectan hasta la línea usando su movimiento. Q o Escape cierran el visor.

Arriba a la derecha del visor aparece, a los pocos segundos de tocar, **Calibracion
OK** o **CALIBRACION TORCIDA**. Las notas bajan a velocidad pareja; si el programa
las ve más rápidas arriba que abajo, las esquinas no siguen la perspectiva de la
pista y el momento de apretar sale mal. En las partidas reales de 99% daba ×1.01;
en todas las que perdían enseguida, entre ×1.66 y ×1.73. Si aparece torcida,
recalibrá con los cuatro puntos sobre los bordes exteriores de la pista.

Las llamas y destellos que aparecen sobre los botones al acertar o mantener una
nota larga no cuentan como notas. Sólo se confirma lo que entra por el 70%
superior de la pista, y una nota confirmada sólo avanza: una llama que queda
detrás de ella no la "hace volver" (eso soltaba los sostenidos y los volvía a
apretar).

El detector busca color saturado y geometría, y asigna la tecla por carril.
Acepta círculos, estrellas y notas azules; los brillos, temas y resoluciones
pueden requerir ajuste. En el JSON, `detector` permite cambiar, por ejemplo:

```json
{"saturation_min": 100, "value_min": 90, "hit_exclusion_ratio": 0.08}
```

Un umbral de saturación menor admite colores más apagados, pero también fondo.
Una banda excluida mayor evita receptores y destellos, pero obliga a extrapolar
más tiempo. Esto debe comprobarse visualmente, incluidas las estrellas y el poder.

## 3. Consultar el modelo y luego conectar las teclas

`IA_guitarristaCanciones.zip` es el modelo entrenado con canciones (`song`,
`gamma` 0.5, 420.000 pasos). En el simulador acierta el 99.6% de las notas y sólo
aprieta al aire en el 0.2% de las celdas vacías; `IA_guitarristaMultiple.zip`
lo hacía en el 31%.

Vista previa de las decisiones del modelo, sin enviar teclas:

```powershell
.\.venv\Scripts\python.exe guitar_flash.py --model IA_guitarristaCanciones.zip
```

Control del juego:

```powershell
.\.venv\Scripts\python.exe guitar_flash.py --model IA_guitarristaCanciones.zip --control
```

Para la mejora de **sostenidos simultáneos** del 22/09, hay una copia entrenada
`IA_guitarristaAcordes.zip`. El original se conserva. Probarla con:

```powershell
.\.venv\Scripts\python.exe guitar_flash.py --model IA_guitarristaAcordes.zip --control --debug
```

Sobre las observaciones del registro `20260922_220640`, las solicitudes de
teclas vacías pasan de 129 a 8. En ocho canciones simuladas de práctica distintas
del entrenamiento, las pulsaciones indebidas bajan de 324 a 35 y los sostenidos
cortados de 8 a 2. Es una mejora de PPO, sin reglas que sustituyan sus acciones.
Falta comprobar el puntaje real y el fallo justo al activar el poder.
Los detalles y límites están en `.coordination/CODEX_STATUS.md`.

Enfocá el juego y pulsá **F8** para activar o pausar. **F9** detiene el programa
incluso con el navegador enfocado. Cambiar de ventana pausa el control y libera
las teclas. También se liberan al salir o ante una excepción. El programa empieza
pausado y sólo controla las cinco teclas configuradas; no activa el poder.
El foco se comprueba por ventana: antes de cambiar de pestaña en el mismo
navegador, pausá con F8.

El controlador mantiene las teclas de sostenidos y genera una nueva pulsación
cuando PPO vuelve a elegir una cabeza distinta del mismo carril. Para que el
juego note esa nueva pulsación, la tecla queda suelta al menos `--min-release-ms`:
si ve venir otra cabeza al mismo carril, la suelta antes y la vuelve a apretar
cuando llega. Esta traducción debe validarse con el comportamiento del juego: el
simulador acepta cabezas con la tecla mantenida. No hay un segundo jugador
automático por reglas.

Opciones de ajuste:

- `--fps 60`: frecuencia base de captura, no garantía de rendimiento.
- `--decision-hz 60`: consultas por segundo al modelo; con 60 consulta en cada
  captura. Captura y consulta se aceleran si las notas cruzan una celda muy
  rápido, hasta donde permita el equipo. No se recuperan notas que pasaron
  durante una interrupción larga.
- `--latency-ms 30`: anticipa las posiciones observadas para compensar lo que
  tardan la pantalla, la captura y la tecla en llegar al juego. En una simulación
  sin esos retrasos, 0 aprieta unos 10 ms tarde y 30 unos 18 ms antes. Probá la
  misma canción con 0, 30 y 60 y quedate con el que dé mejor porcentaje: si
  toca tarde, subilo; si toca antes de tiempo, bajalo.
- `--min-release-ms 25`: tiempo mínimo con la tecla suelta antes de volver a
  apretarla. Si el juego no registra notas seguidas en el mismo carril, subilo.
- `--initial-speed 400`: velocidad inicial en píxeles rectificados por segundo;
  se actualiza con el movimiento observado.
- `--debug`: con `--control`, guarda en `debug\<fecha>\` las detecciones y teclas
  de cada decisión (`eventos.jsonl`) y una imagen de la vista cada vez que una
  tecla se suelta y se vuelve a apretar en menos de 0,4 s. Sirve para ver por
  qué corta un sostenido o aprieta de más.

El simulador aún avanza una fila por acción. El adaptador maneja el reloj real,
pero la transferencia del modelo original no está garantizada. No se implementó
todavía entrenamiento con retardos aleatorios ni aprendizaje dentro del navegador.

## Entrenar y evaluar

El simulador corrige la recompensa de los sostenidos (+2), usa semillas
reproducibles y registra aciertos, errores y sostenidos.

Tiene tres modos de generación (`--note-style` en `train.py` y `play.py`):

- `song` (por defecto): cada episodio sortea una canción con intro y silencios,
  tramos de distinta densidad, acordes y sostenidos. La pista queda vacía
  alrededor del 13% del tiempo y cerca de 1 de cada 4 notas tiene sostenido.
- `dense`: el generador original. La pista casi nunca está vacía y 9 de cada 10
  notas tienen sostenido. Un modelo entrenado sólo así nunca recibe el castigo
  por tocar al aire en silencio, por lo que en una canción real aprieta teclas
  sin notas.
- `sustain`: mezcla episodios de `song` con práctica de acordes sostenidos
  de 4 a 40 filas, frecuentemente simultáneos. Conserva silencios y las mismas
  recompensas. Entrena al propio PPO; no filtra sus acciones durante el juego.

Para practicar sostenidos desde un modelo existente y guardar una copia nueva:

```powershell
.\.venv\Scripts\python.exe train.py --resume IA_guitarristaCanciones.zip --note-style sustain --steps 100000 --seed 6109 --eval-episodes 3 --learning-rate 0.00003 --output IA_practicaNueva
```

Para comparar modelos sin abrir el juego ni enviar teclas:

```powershell
.\.venv\Scripts\python.exe examples/evaluate_live_policy.py --log debug/20260922_220640/eventos.jsonl --models IA_guitarristaCanciones.zip IA_guitarristaAcordes.zip --output artifacts/comparacion_sostenidos.json
```

La comparación usa los tableros reconstruidos del registro v2 y también
canciones simuladas con semillas fijas. Cuenta **solicitudes por cuadro**, no
errores que haya confirmado Guitar Flash. La memoria `held` se estima desde
las acciones registradas; no es un replay exacto del teclado. Si se cambió
altura de la pista o latencia, indicar `--height` y `--latency-ms` correspondientes.
`--learning-rate` permite practicar con cambios pequeños en un modelo que ya
juega bien. Más pasos o más recompensa no garantizan menos errores: comparar
también cabezas perdidas, teclas extra y sostenidos cortados, con otras semillas.

El entrenamiento nuevo recibe `board` (20×5) y `held` (cinco indicadores de
sostenido válido) mediante `MultiInputPolicy`:

```powershell
.\.venv\Scripts\python.exe train.py --steps 500000 --seed 42
```

Los modelos nuevos usan `gamma` 0.5 en vez del 0.99 de PPO. Cada pulsación se
premia o castiga en el mismo paso, así que no hace falta mirar lejos; con 0.99
el retorno suma unos 100 pasos de notas aleatorias y ese ruido tapa el −1 por
tocar al aire. Con las mismas canciones de evaluación, a los 140.000 pasos 0.5
llegaba al 99.6% del puntaje de un jugador perfecto y 0.99 al 51%. Se puede
cambiar con `--gamma`.

Guarda un archivo nuevo en `models/`, checkpoints y evaluaciones separadas, y
el mejor modelo según recompensa de evaluación. Cada evaluación toca las mismas
canciones (semilla fija), así los checkpoints se comparan en igualdad de
condiciones. El modelo original se conserva. A unos 1.400 pasos por segundo,
500.000 pasos tardan unos 7 minutos contando las evaluaciones.

Conviene elegir el nombre con `--output`. Por ejemplo, `--output IA_nueva` crea
en la carpeta del proyecto:

- `IA_nueva_mejor.zip`: el de mejor evaluación, el que conviene usar en
  `guitar_flash.py --model`.
- `IA_nueva.zip`: el modelo del último paso.
- `IA_nueva_datos\checkpoints\`: una copia cada 20.000 pasos (Git la ignora).
- `ppo_guitar_hero_logs\IA_nueva_1`: sólo las curvas para TensorBoard.

Sin `--output`, usa `models\guitar_hero_<fecha y hora>`. Nunca pisa un modelo
existente. Al terminar, o al cortarlo con Ctrl+C, guarda lo aprendido e imprime
los comandos exactos para verlo en `play.py` y probarlo en Guitar Flash.

### Ver cómo entrena

- **Consola:** cada 2.048 pasos imprime una tabla (`ep_rew_mean` es la
  recompensa media de los últimos episodios) y cada 20.000 pasos una línea
  `Eval ... episode_reward=`. Con la semilla por defecto, un jugador perfecto
  saca unos 5.156 en esas canciones de evaluación.
- **Gráficos:** mientras entrena, en otra terminal:

  ```powershell
  .\.venv\Scripts\tensorboard.exe --logdir ppo_guitar_hero_logs
  ```

  y abrí http://localhost:6006. Las curvas `eval/mean_reward` y
  `rollout/ep_rew_mean` deberían subir y estabilizarse. Ahí también están
  `IA_guitarristaCanciones_gamma_0.5` (el entrenamiento de ese modelo),
  `comparacion_gamma_0.99` y `referencia_jugador_perfecto` (5.156, el máximo en
  las canciones de evaluación).
- **Verla jugar:** cualquier checkpoint se puede abrir en el simulador, incluso
  con el entrenamiento en marcha:

  ```powershell
  .\.venv\Scripts\python.exe play.py --model IA_nueva_datos\checkpoints\guitar_hero_200000_steps.zip
  ```

Para continuar uno anterior:

```powershell
.\.venv\Scripts\python.exe train.py --resume IA_guitarristaCanciones.zip --steps 500000 --output IA_guitarristaCanciones_2
.\.venv\Scripts\python.exe play.py --model IA_guitarristaCanciones_2_mejor.zip
```

No conviene reanudar `IA_guitarristaMultiple.zip` para quitarle las pulsaciones
al aire. Con la pista vacía su probabilidad de apretar tres teclas es 1.0: como
nunca prueba no apretar, nunca descubre que así evita el −1. Tras casi 700.000
pasos de canciones seguía igual. Para eso hay que entrenar un modelo nuevo.

Reanudar un modelo antiguo conserva su observación de matriz sola. Para aprender
con `board/held` hay que crear uno nuevo. Tanto `play.py` como `guitar_flash.py`
aceptan ambos formatos. En el juego real, `held` es una **estimación a partir de
las acciones enviadas**, ya que aún no se leen aciertos reales del juego.

## Imágenes y pruebas sin controlar el escritorio

Para analizar una captura propia:

```powershell
.\.venv\Scripts\python.exe guitar_flash.py --image captura.png --calibrate --config imagen.json
.\.venv\Scripts\python.exe guitar_flash.py --image captura.png --config imagen.json --output artifacts/deteccion.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`examples/guitarflash_stars.json` y `examples/guitarflash_sustains.json` contienen
las calibraciones provisionales de las dos imágenes compartidas. Son sólo para
esas imágenes: el programa impide usarlas directamente para controlar la pantalla.
Una foto permite validar geometría y candidatos, pero no velocidad ni precisión
de las pulsaciones. Las pruebas no envían teclas reales ni abren ventanas.
