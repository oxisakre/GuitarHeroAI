"""Seguimiento geométrico; la elección de teclas queda a cargo de PPO."""
import collections
import itertools
from dataclasses import dataclass, field

import numpy as np

from .vision import tail_rows

HISTORIA = 8
BASE_MINIMA = 0.02
CUADRO_LENTO = 1 / 30
# Las notas reales miden entre p50 y p99 casi lo mismo (26 y 31 px en el registro
# real); el doble ya no es una nota.
ALTO_TOLERADO = 2.0
# El cartel "50 Notas Acertadas" cruza toda la pista arriba. Mientras aparece y se
# va, sus letras se rompen en pedazos del tamaño de una nota: se ignora su franja
# mientras está y un rato después. Las notas se toman cuando salen por debajo.
CARTEL_CARRILES = 3
CARTEL_ALTURA = 0.25       # sólo se busca en el cuarto de arriba de la pista
CARTEL_TOPE = 0.35         # la franja ignorada nunca baja de acá
CARTEL_MARGEN = 20.0
CARTEL_GRACIA = 0.3


def velocidad_robusta(historia):
    """Pendiente por mediana de pares, ignorando bases de tiempo cortas.

    Las capturas no están sincronizadas con los cuadros del juego: a veces todo
    el salto de un cuadro cae entre dos capturas muy juntas y ese par solo daría
    el doble de velocidad. Con varias muestras y la mediana, eso no manda.
    """
    pendientes = [(y2 - y1) / (t2 - t1)
                  for (t1, y1), (t2, y2) in itertools.combinations(historia, 2)
                  if t2 - t1 >= BASE_MINIMA]
    return float(np.median(pendientes)) if pendientes else 0.0


@dataclass
class Track:
    identifier: int
    lane: int
    y: float
    last_seen: float
    first_y: float
    first_seen: float
    velocity: float = 0.0
    observations: int = 1
    moving: bool = False
    history: list = field(default_factory=list)
    tail_top: float | None = None
    last_tail_seen: float | None = None
    tail_clipped: bool = False


class BoardTracker:
    """Proyecta notas confirmadas en movimiento al tablero 20x5 del modelo.

    La homografía quita la perspectiva. Su velocidad se estima en píxeles
    rectificados/segundo. No se decide qué nota tocar dentro de esta clase.
    """

    def __init__(self, height=800, width=500, initial_speed=400.0,
                 max_unseen=0.45, latency_ms=0.0, entry_ratio=0.7):
        self.height, self.width = height, width
        self.speed = float(initial_speed)
        self.max_unseen = max_unseen
        self.latency = latency_ms / 1000.0
        # Las notas entran por arriba de la pista. Lo que aparece por primera vez
        # cerca de los botones (llamas al acertar o sostener) nunca se confirma.
        self.entry_limit = height * entry_ratio
        self.tracks = []
        self._next_id = 1
        self._last_heads = {}
        # Velocidad medida en el tercio de arriba, medio y abajo de la pista.
        # Con la perspectiva bien calibrada las tres dan lo mismo.
        self._por_tercio = [collections.deque(maxlen=300) for _ in range(3)]
        # Alto de las notas confirmadas: todas miden parecido. Carteles como
        # "50 Notas Acertadas" miden tres veces más (85 px contra 26 en el
        # registro real) y sus letras parecían notas entrando por arriba.
        self._altos = collections.deque(maxlen=400)
        # (hasta cuándo, borde inferior, cuándo empezó) del cartel en pantalla.
        self._cartel = None

    def alto_maximo(self):
        """Alto máximo aceptable para una nota, o None mientras se aprende."""
        if len(self._altos) < 40:
            return None
        return ALTO_TOLERADO * float(np.median(self._altos))

    def reset(self):
        self.tracks.clear()
        self._last_heads = {}
        self._cartel = None

    def zona_cartel(self, heads, timestamp, limite):
        """Borde inferior de la franja tapada por el cartel, o None si no hay."""
        altas = [head for head in heads
                 if head.height > limite and head.y < CARTEL_ALTURA * self.height]
        if len({head.lane for head in altas}) >= CARTEL_CARRILES:
            borde = max(head.y + head.height / 2 for head in altas) + CARTEL_MARGEN
            if self._cartel is None or timestamp > self._cartel[0]:
                desde, anterior = timestamp, 0.0
            else:
                _, anterior, desde = self._cartel
            # El cartel crece y se inclina: la franja sólo se agranda.
            borde = min(max(anterior, borde), CARTEL_TOPE * self.height)
            self._cartel = (timestamp + CARTEL_GRACIA, borde, desde)
        if self._cartel is not None and timestamp <= self._cartel[0]:
            return self._cartel[1]
        return None

    def calibracion(self):
        """Cuánto cambia la velocidad entre arriba y abajo (1 = pareja), o None
        si todavía no hay suficientes notas medidas en los tres tercios."""
        if any(len(tercio) < 30 for tercio in self._por_tercio):
            return None
        medianas = [float(np.median(tercio)) for tercio in self._por_tercio]
        return max(medianas) / min(medianas)

    def update(self, heads, tail_mask, timestamp):
        # Carril por carril: si otro carril parpadea (un destello, el fondo), esta
        # imagen repetida no debe pasar por nueva. Sin información nueva, emparejar
        # inventa notas y velocidades del doble.
        signature = {}
        for head in heads:
            signature.setdefault(head.lane, []).append(round(head.y, 1))
        repetidos = {lane for lane, ys in signature.items() if ys == self._last_heads.get(lane)}
        self._last_heads = signature
        limite = self.alto_maximo()
        # El cartel se busca en la imagen completa: quieto, sus carriles se
        # descartan por repetidos y la franja se apagaría sola.
        borde = None if limite is None else self.zona_cartel(heads, timestamp, limite)
        heads = [head for head in heads if head.lane not in repetidos]
        if limite is not None:
            if borde is not None:
                heads = [head for head in heads if head.y > borde]
                # Los primeros pedazos llegan un par de cuadros antes que las
                # letras enteras: lo que nació ahí desde entonces no es una nota.
                desde = self._cartel[2] - 0.15
                self.tracks = [track for track in self.tracks
                               if not (track.first_seen >= desde and track.y <= borde)]
            heads = [head for head in heads if head.height <= limite]
        emparejados, nuevas = [], []
        for lane in range(5):
            cabezas = sorted((head for head in heads if head.lane == lane), key=lambda h: -h.y)
            # Las notas de un carril no se cruzan: la que se vio más abajo sigue
            # más abajo. Emparejar respetando ese orden evita que una nota con la
            # velocidad mal estimada se quede con la cabeza de su vecina.
            seguidas = sorted((track for track in self.tracks if track.lane == lane
                               and timestamp - track.last_seen <= self.max_unseen),
                              key=lambda track: -track.y)
            indice_cabeza = indice_nota = 0
            while indice_cabeza < len(cabezas):
                cabeza = cabezas[indice_cabeza]
                if indice_nota >= len(seguidas):
                    nuevas.append(cabeza)
                    indice_cabeza += 1
                elif seguidas[indice_nota].y > cabeza.y + 2:
                    indice_nota += 1          # esa nota va por delante de esta cabeza
                elif self._llama_ante_nota(seguidas, indice_nota, cabeza, timestamp):
                    indice_nota += 1
                elif self._acepta(seguidas[indice_nota], cabeza, timestamp):
                    emparejados.append((seguidas[indice_nota], cabeza))
                    indice_cabeza += 1
                    indice_nota += 1
                elif self._quedo_atras(seguidas[indice_nota], cabeza, timestamp):
                    # Esa nota ya bajó a la banda de los botones y no se ve: esta
                    # cabeza es la siguiente de la ráfaga. Antes se la tomaba como
                    # nota nueva sin confirmar y en ráfagas muy pegadas se perdía.
                    indice_nota += 1
                else:
                    nuevas.append(cabeza)
                    indice_cabeza += 1
        for track, head in emparejados:
            self._actualizar(track, head, timestamp)
        for head in nuevas:
            self.tracks.append(Track(
                self._next_id, head.lane, float(head.y), timestamp,
                float(head.y), timestamp, tail_top=head.tail_top,
                last_tail_seen=timestamp if head.tail_top is not None else None,
                tail_clipped=head.tail_top is not None and head.tail_top <= 2,
                history=[(timestamp, float(head.y))]))
            self._next_id += 1
        self._envejecer(tail_mask, timestamp)

    def _llama_ante_nota(self, seguidas, indice, head, timestamp):
        """¿Una forma nacida junto a los botones le disputa la cabeza a una nota?

        Lo que aparece por primera vez abajo (una llama) nunca se confirma. Si
        la nota que viene detrás también acepta esta cabeza, es de la nota: si no,
        la llama se la quedaba y la nota se perdía en ráfagas del mismo color.
        """
        track = seguidas[indice]
        if track.moving or track.first_y < self.entry_limit or indice + 1 >= len(seguidas):
            return False
        siguiente = seguidas[indice + 1]
        return siguiente.moving and self._acepta(siguiente, head, timestamp)

    def _quedo_atras(self, track, head, timestamp):
        """¿La detección está por detrás de donde ya tiene que estar esta nota?"""
        travel = (track.velocity or self.speed) * (timestamp - track.last_seen)
        return track.moving and head.y < track.y + 0.35 * travel - 2

    def _acepta(self, track, head, timestamp):
        """¿Esta detección puede ser esta nota?"""
        elapsed = timestamp - track.last_seen
        travel = (track.velocity or self.speed) * elapsed
        distance = abs(head.y - (track.y + travel))
        if track.moving:
            # Una nota confirmada sólo avanza: una detección que quedó por detrás
            # (una llama en el botón cuando la nota ya entró en la banda ciega) no
            # es ella; tomarla la haría "volver atrás" y soltaría el sostenido.
            # El margen acompaña a la velocidad porque el avance no depende del
            # tiempo entre capturas sino de cuántos cuadros dibujó el juego: entre
            # dos capturas muy juntas puede caber un cuadro entero.
            ritmo = CUADRO_LENTO * (track.velocity or self.speed)
            return (head.y >= track.y + 0.35 * travel - 2
                    and distance <= max(20.0, 1.2 * travel, ritmo))
        # Sólo una forma aún no confirmada puede seguir quieta. Su velocidad todavía
        # no es de fiar, así que se acepta un salto grande, pero no tanto como para
        # saltar por encima de la nota siguiente en una ráfaga del mismo carril.
        distance = min(distance, abs(head.y - track.y))
        return distance <= max(25.0, 0.08 * self.height, self.speed * elapsed * 2.5)

    def _actualizar(self, track, head, timestamp):
        delta = head.y - track.y
        if delta <= 1.0:
            # La nota está donde estaba: es el mismo cuadro del juego visto otra
            # vez. Actualizarle la hora haría que el próximo movimiento pareciera
            # ocurrir en menos tiempo y la velocidad saldría al doble.
            return
        track.observations += 1
        track.history.append((timestamp, head.y))
        del track.history[:-HISTORIA]
        velocity = velocidad_robusta(track.history)
        if 15 < velocity < self.height * 8:
            track.velocity = velocity
            # Dos manchas quietas a distinta altura parecen una nota velocísima,
            # pero sólo una vez. Una nota real se ve bajar varias veces seguidas.
            # Todas las notas bajan a la misma velocidad: algo que va a menos de la
            # mitad (una forma fija que tiembla arriba de la pista) no es una nota.
            # Tampoco lo que va a más del doble: pedazos del cartel "50 Notas
            # Acertadas" saltaban 80 px en 30 ms (2800 px/s con notas a 660).
            # Ese tope recién vale cuando la velocidad del juego ya se aprendió.
            aprendida = len(self._altos) >= 40
            track.moving |= (len(track.history) >= 3 and head.y - track.first_y >= 4
                             and track.first_y < self.entry_limit
                             and track.velocity >= 0.5 * self.speed
                             and (not aprendida or track.velocity <= 2.0 * self.speed))
            if track.moving:
                self._altos.append(head.height)
            if track.moving:
                # Un destello que salta de lugar no debe cambiar la velocidad con
                # la que se proyectan todas las notas: cada nota la corrige 10%.
                sample = min(max(track.velocity, 0.5 * self.speed), 2.0 * self.speed)
                self.speed = 0.9 * self.speed + 0.1 * sample
                if len(track.history) >= 4:
                    # Arriba de todo las notas recién aparecen y su velocidad sale
                    # baja aunque la calibración esté bien: se mide de ahí para abajo.
                    medio = (track.history[0][1] + track.history[-1][1]) / 2
                    desde, hasta = 0.25 * self.height, 0.875 * self.height
                    if desde <= medio < hasta:
                        self._por_tercio[int(3 * (medio - desde) / (hasta - desde))].append(track.velocity)
        if track.tail_top is not None:
            track.tail_top += delta
        track.y, track.last_seen = head.y, timestamp
        if head.tail_top is not None:
            track.tail_top = float(head.tail_top)
            track.last_tail_seen = timestamp
            track.tail_clipped = head.tail_top <= 2

    def _envejecer(self, tail_mask, timestamp):
        """Corre las colas con el tiempo y descarta lo que ya pasó o no era nota."""
        kept = []
        for track in self.tracks:
            elapsed = timestamp - track.last_seen
            speed = track.velocity or self.speed
            y = track.y + speed * elapsed
            tail_y = None if track.tail_top is None else track.tail_top + speed * elapsed
            # Después de pasar la cabeza seguimos observando el extremo de la
            # cola. Esto evita inventar su duración si salía por arriba.
            if track.moving and tail_y is not None and y > self.height * 0.85:
                rows = np.flatnonzero(tail_rows(tail_mask, track.lane, self.width))
                if rows.size:
                    gaps = np.diff(rows) > 3
                    starts = rows[np.r_[True, gaps]]
                    ends = rows[np.r_[gaps, True]]
                    index = int(np.argmin(abs(starts - tail_y)))
                    candidate = float(starts[index])
                    clipped = candidate <= 2
                    # Una cola que sigue entrando por arriba puede quedar oculta
                    # brevemente. Su extremo real aún no se conoce: recuperarla
                    # en y=0 no debe depender sólo de la posición extrapolada.
                    # Debe llegar sin cortes hasta abajo para no tomar la cola de
                    # otra nota que acaba de entrar en el mismo carril.
                    connected = ends[index] >= self.height * 0.75
                    recovered = (track.tail_clipped and clipped and connected
                                 and track.last_tail_seen is not None
                                 and timestamp - track.last_tail_seen <= self.max_unseen)
                    nearby = abs(candidate - tail_y) < max(60, speed * 0.25)
                    if recovered or (nearby and not clipped):
                        track.tail_top = candidate - speed * elapsed
                        track.last_tail_seen = timestamp
                        track.tail_clipped = clipped
                        tail_y = candidate
            # La cabeza deja de verse antes de la línea por la banda de
            # receptores. A baja velocidad puede tardar más de .45 s en
            # cruzarla; la vida útil ahí debe depender de la geometría.
            near_hit = track.moving and track.y >= self.height * 0.75
            if tail_y is not None:
                # Una cabeza con cola que se deja de ver arriba de la pista no se
                # tapó con los receptores: era una mancha (el cartel de aciertos)
                # y no puede seguir bajando sola hasta la línea.
                alive = (near_hit or elapsed <= self.max_unseen) and tail_y < self.height + self.height / 20
            else:
                alive = (near_hit or elapsed <= self.max_unseen) and y < self.height + self.height / 20
            # Una falsa detección inmóvil no queda retenida indefinidamente.
            if not track.moving:
                alive = elapsed <= self.max_unseen and timestamp - track.first_seen < 2
            # Si nadie vio ni su cabeza ni su cola en más tiempo del que tarda una
            # nota real en cruzar toda la pista, no es una nota: una "cola cortada
            # por arriba" que no se vuelve a ver la mantenía viva 13 s y la hacía
            # quedarse quieta en la línea, apretando la tecla sin motivo.
            visto = max(track.last_seen, track.last_tail_seen or track.last_seen)
            if timestamp - visto > max(1.5, 2 * self.height / max(self.speed, 1.0)):
                alive = False
            if alive:
                kept.append(track)
        self.tracks = kept

    def observation(self, timestamp):
        board = np.zeros((20, 5), dtype=np.uint8)
        tokens = [None] * 5
        # Fila 19 comienza exactamente en la línea calibrada, no una celda
        # antes. Las cabezas se extrapolan unos píxeles fuera del recorte.
        cell = (self.height - 1) / 19
        for track in self.tracks:
            if not track.moving:
                continue
            elapsed = timestamp - track.last_seen + self.latency
            speed = track.velocity or self.speed
            y = track.y + speed * elapsed
            row = int(np.floor(y / cell))
            if track.tail_top is not None:
                top = track.tail_top + speed * elapsed
                first = max(0, int(np.floor(top / cell)))
                last = min(20, row)
                if first < last:
                    board[first:last, track.lane] = 2
            if 0 <= row < 20:
                board[row, track.lane] = 1
                if row == 19:
                    tokens[track.lane] = track.identifier
        return board, tokens


def update_held_estimate(held, board, action):
    """Memoria de acciones, no confirmación de aciertos por parte del juego."""
    result = np.asarray(held, dtype=np.int8).copy()
    for lane in range(5):
        if not action[lane] or board[19, lane] == 0:
            result[lane] = 0
        elif board[19, lane] == 1:
            result[lane] = 1
    return result
