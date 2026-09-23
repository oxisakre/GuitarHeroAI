import unittest

import numpy as np

from guitarflash.tracking import BoardTracker, update_held_estimate
from guitarflash.vision import Detection


def near_line_tracker(**kwargs):
    """Estos tests hacen aparecer las notas cerca de la línea para ser breves;
    el filtro de entrada por arriba se prueba aparte."""
    return BoardTracker(entry_ratio=1.0, **kwargs)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((800, 500), np.uint8)

    def test_flicker_near_receptors_never_becomes_a_note(self):
        # Llamas al acertar: manchas que aparecen y saltan cerca de los botones.
        tracker = BoardTracker(initial_speed=400)
        rng = np.random.default_rng(3)
        for index in range(240):
            timestamp = index / 60
            flames = [Detection(lane, float(rng.uniform(680, 735)))
                      for lane in range(5) if rng.random() < 0.6]
            tracker.update(flames, self.mask, timestamp)
            board, tokens = tracker.observation(timestamp)
            self.assertFalse(board.any())
            self.assertEqual(tokens, [None] * 5)
        self.assertEqual(tracker.speed, 400)

    def test_game_drawing_at_half_the_capture_rate_does_not_duplicate_notes(self):
        # El juego dibuja a 30 cuadros y la captura va a 60: cada imagen llega dos veces.
        tracker = BoardTracker(initial_speed=1600)
        for index in range(40):
            timestamp = index / 60
            y = 40 + 1600 * (index // 2) / 30          # sólo avanza en cuadros nuevos
            tracker.update([Detection(1, y)], self.mask, timestamp)
        self.assertEqual(len(tracker.tracks), 1, "Cada nota debe seguir siendo una sola.")
        self.assertAlmostEqual(tracker.tracks[0].velocity, 1600, delta=200)
        self.assertAlmostEqual(tracker.speed, 1600, delta=200)

    def test_flicker_in_another_lane_does_not_split_notes(self):
        # Cuatro notas del mismo carril a 123 px, juego a 30 cuadros y captura a 60,
        # con dos manchas del fondo alternando en otro carril: la alternancia no debe
        # hacer que las imágenes repetidas se procesen como nuevas.
        tracker = BoardTracker(initial_speed=560)
        llegadas = set()
        for index in range(160):
            timestamp = index / 60
            base = 60 + 560 * (index // 2) / 30          # el juego sólo avanza cada 2 capturas
            notas = [Detection(3, base - 123 * n) for n in range(4) if 0 <= base - 123 * n < 736]
            destello = [Detection(0, 52.0 if index % 2 else 91.0)]
            tracker.update(notas + destello, self.mask, timestamp)
            _, tokens = tracker.observation(timestamp)
            llegadas |= {token for token in tokens if token is not None}
            self.assertIsNone(tokens[0], "El destello no es una nota.")
        self.assertEqual(len(llegadas), 4, "Cada nota debe llegar a la línea una sola vez.")

    def test_capture_jitter_does_not_double_the_speed(self):
        # Juego a 30 cuadros y captura a 60 con desfase irregular: el salto de un
        # cuadro entero cae a veces entre dos capturas separadas por pocos ms, y
        # ese par solo daría el doble de velocidad.
        tracker = BoardTracker(initial_speed=560)
        rng = np.random.default_rng(7)
        for index in range(30):
            timestamp = index / 60 + float(rng.uniform(-0.005, 0.005))
            y = 100 + 560 * (int(timestamp * 30) / 30)     # sólo cambia cada cuadro del juego
            tracker.update([Detection(2, y)], self.mask, timestamp)
        self.assertAlmostEqual(tracker.tracks[0].velocity, 560, delta=90)
        self.assertAlmostEqual(tracker.speed, 560, delta=120)

    def test_fast_notes_with_capture_jitter_stay_one_note(self):
        # A 1700 px/s la nota avanza 57 px por cuadro del juego. Según cuándo caiga
        # la captura, el avance visible entre dos capturas va de 57 a 114 px: no debe
        # tomarse como una nota nueva.
        for semilla in range(6):
            with self.subTest(semilla=semilla):
                tracker = BoardTracker(initial_speed=400)
                rng = np.random.default_rng(semilla)
                identidades = set()
                for index in range(40):
                    timestamp = index / 60 + float(rng.uniform(-0.008, 0.008))
                    y = 30 + 1700 * (int(timestamp * 30) / 30)
                    if y >= 736:
                        break
                    tracker.update([Detection(1, y)], self.mask, timestamp)
                    identidades |= {t.identifier for t in tracker.tracks}
                self.assertEqual(len(identidades), 1, "La misma nota no debe partirse en dos.")

    def test_burst_of_eight_notes_keeps_eight_identities(self):
        # Ráfaga del mismo carril a 1700 px/s, juego a 30 cuadros y captura a 60 con
        # desfase: una nota con la velocidad recién estimada no debe quedarse con la
        # cabeza de su vecina y hacer que aparezca una novena identidad.
        for semilla in (15, 3, 21):
            with self.subTest(semilla=semilla):
                tracker = BoardTracker(initial_speed=400)
                rng = np.random.default_rng(semilla)
                llegadas = set()
                for index in range(90):
                    timestamp = index / 60 + float(rng.uniform(-0.008, 0.008))
                    base = 30 + 1700 * int(timestamp * 30) / 30
                    notas = [Detection(1, base - k * 110) for k in range(8)
                             if 0 <= base - k * 110 < 704]
                    tracker.update(notas, self.mask, timestamp)
                    _, tokens = tracker.observation(timestamp)
                    llegadas.update(token for token in tokens if token is not None)
                self.assertEqual(len(llegadas), 8)

    def pasar_notas(self, tracker, posicion):
        """Notas que entran por arriba cada 0.3 s; posicion(t) da su altura a los t s."""
        for index in range(600):
            timestamp = index / 60
            notas = []
            for n in range(int(timestamp / 0.3) + 1):
                y = posicion(timestamp - n * 0.3)
                if 0 <= y < 704:
                    notas.append(Detection(n % 5, y))
            tracker.update(notas, self.mask, timestamp)

    def test_calibration_check_detects_uneven_speed(self):
        pareja = BoardTracker(initial_speed=600)
        self.pasar_notas(pareja, lambda t: 10 + 600 * t)
        self.assertLess(pareja.calibracion(), 1.2)
        # Perspectiva mal corregida: arriba van rápido y abajo frenan (como en el
        # registro real con la calibración corrida: de 1000 a 420 px/s).
        torcida = BoardTracker(initial_speed=600)
        self.pasar_notas(torcida, lambda t: 800 * (1 - np.exp(-1.5 * t)))
        self.assertGreater(torcida.calibracion(), 1.5)

    def test_slow_shape_with_clipped_tail_never_sits_on_the_line(self):
        # Registro real: una forma arriba del carril naranja, con "cola cortada" y
        # 62 px/s, quedó 13 s viva y 0,7 s quieta en la línea apretando la tecla.
        tracker = BoardTracker(initial_speed=660)
        for index in range(900):                      # 15 s a 60 capturas por segundo
            timestamp = index / 60
            notas = [Detection(0, 30 + 660 * (timestamp % 1.2))] if timestamp % 1.2 < 1.0 else []
            if timestamp < 0.5:
                notas.append(Detection(4, 50 + 62 * timestamp, has_tail=True, tail_top=0.0))
            tracker.update(notas, self.mask, timestamp)
            board, tokens = tracker.observation(timestamp)
            self.assertEqual(board[19, 4], 0, f"Nada debe llegar a la línea naranja (t={timestamp:.2f}).")
        self.assertFalse([t for t in tracker.tracks if t.lane == 4])

    def _partida(self, tracker, hasta, extra=lambda timestamp: []):
        """Notas de 26 px a 660 px/s en los carriles 2 a 5, una cada 0,25 s.
        Deja el carril 1 libre para lo que prueba cada test."""
        tokens_carril_1 = set()
        for index in range(int(hasta * 60)):
            timestamp = index / 60
            notas = [Detection(1 + n % 4, 30 + 660 * (timestamp - n * 0.25), height=26.0)
                     for n in range(int(timestamp / 0.25) + 1)
                     if 30 + 660 * (timestamp - n * 0.25) < 704]
            tracker.update(notas + extra(timestamp), self.mask, timestamp)
            _, tokens = tracker.observation(timestamp)
            if tokens[0] is not None:
                tokens_carril_1.add(tokens[0])
        return tokens_carril_1

    def test_banner_letters_are_not_taken_as_notes(self):
        # Registro real: las letras de "50 Notas Acertadas" miden ~85 px (las
        # notas ~26). Quedaban quietas arriba y la nota que entraba después
        # heredaba esa pista con la historia equivocada.
        tracker = BoardTracker(initial_speed=660)
        cartel = lambda t: [Detection(0, 60 + 2 * (t - 5.0), height=85.0)] if t >= 5.0 else []
        self._partida(tracker, 5.5, cartel)
        self.assertIsNotNone(tracker.alto_maximo())
        self.assertFalse([t for t in tracker.tracks if t.lane == 0])

    def test_letter_pieces_under_the_banner_never_reach_the_line(self):
        # Secuencia real del carril amarillo (Preludio Obsesivo, 49,87 s): tres
        # pedazos de letra apilados se cruzaban, formaban notas de 1000-1600 px/s,
        # subían la velocidad del juego y una llegaba a la línea: tecla errada.
        # Por debajo del cartel sale después una nota real.
        pedazos = [[(67, 18)], [(72, 22), (103, 13)], [(36, 12), (77, 23), (107, 16)],
                   [(38, 16), (79, 25), (110, 16)], [(41, 23), (84, 26), (114, 17)],
                   [(42, 33), (88, 28), (117, 19)], [(49, 32), (92, 36), (121, 17)],
                   [(52, 34), (92, 39), (125, 20)], [(57, 31), (108, 63)],
                   [(62, 36), (101, 35), (133, 22)], [(68, 34), (119, 61)],
                   [(72, 32), (122, 61)], [(100, 98)], [(98, 94)], [(95, 92)], [(91, 94)],
                   [(90, 87)], [(54, 32), (100, 53)], [(47, 33), (96, 53)]]
        inicio, salida, y_salida = 5.0, 5.0 + 18 / 60, 175.0
        golpe = salida + (799 - y_salida) / 660

        def cartel(t):
            cuadro = round((t - inicio) * 60)
            cabezas = []
            if 0 <= cuadro < len(pedazos):
                cabezas += [Detection(0, y, height=h) for y, h in pedazos[cuadro]]
            if 2 <= cuadro < 90:                          # las letras enteras, en otros carriles
                cabezas += [Detection(lane, 60.0 + 8 * lane, height=92.0) for lane in (1, 3, 4)]
            if t >= salida and y_salida + 660 * (t - salida) < 704:
                cabezas.append(Detection(0, y_salida + 660 * (t - salida), height=26.0))
            return cabezas

        tracker = BoardTracker(initial_speed=660, latency_ms=30)
        primera = {}
        for index in range(int(7.5 * 60)):
            timestamp = index / 60
            notas = [Detection(1 + n % 4, 30 + 660 * (timestamp - n * 0.25), height=26.0)
                     for n in range(int(timestamp / 0.25) + 1)
                     if 30 + 660 * (timestamp - n * 0.25) < 704]
            tracker.update(notas + cartel(timestamp), self.mask, timestamp)
            _, tokens = tracker.observation(timestamp)
            if tokens[0] is not None:
                primera.setdefault(tokens[0], timestamp)
        self.assertEqual(len(primera), 1, f"Una sola nota real en el carril 1: {primera}")
        self.assertAlmostEqual(next(iter(primera.values())), golpe - 0.03, delta=0.04)

    def test_banner_fragments_faster_than_notes_are_not_confirmed(self):
        # Registro real: pedazos del cartel saltaban 80 px en 30 ms (2800 px/s)
        # cuando las notas bajaban a 660 y llegaban a la línea como notas.
        tracker = BoardTracker(initial_speed=660)
        saltos = {300: 43.0, 301: 116.0, 302: 123.0, 303: 200.0}
        pedazo = lambda t: ([Detection(0, saltos[round(t * 60)], height=16.0)]
                            if round(t * 60) in saltos else [])
        llegadas = self._partida(tracker, 7.0, pedazo)
        self.assertFalse(llegadas, "Un pedazo del cartel no debe apretar una tecla.")

    def test_speed_limit_waits_until_the_game_speed_is_learned(self):
        # Al arrancar la velocidad inicial puede estar lejos de la real: no se
        # descarta una nota por ir rápida antes de haberla aprendido.
        tracker = near_line_tracker(initial_speed=400)
        for index in range(6):
            tracker.update([Detection(0, 100 + 1500 * index / 60)], self.mask, index / 60)
        self.assertTrue(tracker.tracks[0].moving)

    def test_tailed_blob_lost_at_the_top_does_not_reach_the_line(self):
        # Registro real: una mancha del cartel con "cola" se vio tres veces arriba,
        # desapareció y la cola la mantenía viva hasta la línea: tecla sin nota.
        tracker = BoardTracker(initial_speed=660)
        mancha = lambda t: ([Detection(0, 90 + 660 * (t - 5.0), height=40.0, has_tail=True,
                                       tail_top=50 + 660 * (t - 5.0))]
                            if 5.0 <= t < 5.06 else [])
        llegadas = self._partida(tracker, 7.5, mancha)
        self.assertFalse(llegadas)

    def test_tight_same_lane_burst_keeps_every_note(self):
        # A 400 px/s, cuatro notas cada 0,1 s quedan a 40 px. La primera entra a
        # la banda de los botones y deja de verse; la segunda se comparaba sólo
        # con ella, se rechazaba y se creaba una nota nueva que nunca se confirma.
        tracker = BoardTracker(initial_speed=400)
        golpes = [2.0 + 0.1 * n for n in range(4)]
        llegadas, huerfanas = [], set()
        for index in range(int(2.6 * 55)):
            timestamp = index / 55
            cuadro = int(timestamp * 60) / 60              # el juego dibuja a 60 cuadros
            cabezas = [Detection(0, 799 - 400 * (golpe - cuadro), height=22.0) for golpe in golpes
                       if 0 <= 799 - 400 * (golpe - cuadro) < 704]
            tracker.update(cabezas, self.mask, timestamp)
            huerfanas |= {t.identifier for t in tracker.tracks if t.first_y >= tracker.entry_limit}
            _, tokens = tracker.observation(timestamp + 0.03)
            if tokens[0] is not None and tokens[0] not in llegadas:
                llegadas.append(tokens[0])
        self.assertEqual(len(llegadas), 4)
        self.assertFalse(huerfanas, "Cada cabeza de la ráfaga tiene que quedar con su nota.")

    def test_flame_at_the_buttons_does_not_steal_the_next_note(self):
        # Una llama junto a los botones queda como forma sin confirmar. Cuando la
        # nota pasa por ahí, la cabeza es de la nota y no de la llama.
        tracker = BoardTracker(initial_speed=400)
        for index in range(int(1.8 * 60)):
            timestamp = index / 60
            cabezas = []
            y = 799 - 400 * (1.8 - timestamp)
            if y < 704:
                cabezas.append(Detection(0, y, height=22.0))
            if 1.40 <= timestamp < 1.45:
                cabezas.append(Detection(0, 692.0, height=22.0))
            tracker.update(cabezas, self.mask, timestamp)
        nota = next(t for t in tracker.tracks if t.moving)
        self.assertGreater(nota.y, 690)

    def test_frozen_screen_with_coloured_shapes_creates_no_notes(self):
        # Cartel de "Usted falló": formas de colores fijas repetidas en cada captura.
        tracker = BoardTracker(initial_speed=1600)
        shapes = [Detection(1, 320), Detection(1, 524), Detection(1, 598), Detection(3, 643)]
        for index in range(120):
            timestamp = index / 60
            tracker.update(shapes, self.mask, timestamp)
            board, tokens = tracker.observation(timestamp)
            self.assertFalse(board.any())
            self.assertEqual(tokens, [None] * 5)
        self.assertEqual(tracker.speed, 1600)
        self.assertFalse(tracker.tracks, "Las formas quietas no deben quedar retenidas.")

    def test_note_entering_from_the_top_is_confirmed(self):
        tracker = BoardTracker(initial_speed=400)
        for timestamp in (0.0, 0.05, 0.1):
            tracker.update([Detection(2, 150 + 400 * timestamp)], self.mask, timestamp)
        self.assertTrue(tracker.tracks[0].moving)
        board, _ = tracker.observation(0.1)
        self.assertEqual(board[int(190 / (799 / 19)), 2], 1)

    def test_stationary_receivers_do_not_become_notes(self):
        tracker = BoardTracker()
        for timestamp in (0, 0.05, 0.1, 0.2):
            tracker.update([Detection(2, 720)], self.mask, timestamp)
        board, tokens = tracker.observation(0.2)
        self.assertFalse(board.any())
        self.assertEqual(tokens, [None] * 5)

    def test_confirmed_head_reaches_last_row_with_stable_token(self):
        tracker = near_line_tracker(initial_speed=400)
        for timestamp, y in ((0, 670), (0.05, 690), (0.1, 710), (0.15, 730)):
            tracker.update([Detection(0, y)], self.mask, timestamp)
        tracker.update([], self.mask, 0.33)
        board, tokens = tracker.observation(0.33)
        self.assertEqual(board[19, 0], 1)
        self.assertIsNotNone(tokens[0])
        later_board, later_tokens = tracker.observation(0.35)
        self.assertEqual(later_board[19, 0], 1)
        self.assertEqual(tokens[0], later_tokens[0])

    def test_slow_head_survives_excluded_receptor_band(self):
        tracker = near_line_tracker(initial_speed=100)
        for timestamp, y in ((0, 700), (0.1, 710), (0.2, 720), (0.3, 730)):
            tracker.update([Detection(1, y)], self.mask, timestamp)
        # Las cabezas dejan de detectarse antes de y=736. A velocidad lenta
        # tardan más de max_unseen en cruzar esa banda hasta la línea y=799.
        for timestamp in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            tracker.update([], self.mask, timestamp)
        board, tokens = tracker.observation(1.0)
        self.assertEqual(board[19, 1], 1)
        self.assertIsNotNone(tokens[1])

    def test_sustain_remains_after_head_leaves_image(self):
        tracker = near_line_tracker(initial_speed=400)
        for timestamp, y in ((0, 680), (0.05, 700), (0.1, 720), (0.15, 740)):
            tracker.update([Detection(3, y, has_tail=True, tail_top=y - 500)], self.mask, timestamp)
        tracker.update([], self.mask, 0.45)
        board, tokens = tracker.observation(0.45)
        self.assertEqual(board[19, 3], 2)
        self.assertIsNone(tokens[3])
        tracker.update([], self.mask, 1.7)
        self.assertFalse(tracker.observation(1.7)[0].any())

    def test_held_sustain_with_tail_off_center_is_kept_while_visible(self):
        # Naranja larga: la cabeza se ve hasta la banda de botones y después sólo
        # queda la cola, 19 px corrida hacia el centro. Arriba el juego oscurece la
        # pista: la cola se ve desde y≈110 siempre, aunque siga más arriba.
        # Mientras se vea, se sostiene.
        tracker = near_line_tracker(initial_speed=400)
        for timestamp, y in ((0, 600), (0.05, 620), (0.1, 640), (0.15, 660), (0.2, 680)):
            tracker.update([Detection(4, y, has_tail=True, tail_top=110.0)], self.mask, timestamp)
        mascara = np.zeros_like(self.mask)
        for index in range(1, 60):
            timestamp = 0.2 + index / 30
            mascara[:] = 0
            mascara[110:704, 428:434] = 255
            tracker.update([], mascara, timestamp)
            board, _ = tracker.observation(timestamp)
            if timestamp > 0.2 + (799 - 680) / 400 + 0.05:     # la cabeza ya pasó la línea
                self.assertGreater(board[19, 4], 0, f"soltó a los {timestamp:.2f} s")

    def test_tail_clipped_above_image_does_not_get_fixed_duration(self):
        tracker = near_line_tracker(initial_speed=400)
        mask = self.mask.copy()
        mask[:735, 346:354] = 255
        for timestamp, y in ((0, 680), (0.025, 690), (0.05, 700), (0.075, 710)):
            tracker.update([Detection(3, y, has_tail=True, tail_top=0)], mask, timestamp)
        for timestamp in np.arange(0.1, 3, 0.05):
            tracker.update([], mask, float(timestamp))
        self.assertEqual(tracker.observation(2.95)[0][19, 3], 2)

    def _clipped_sustain(self, speed=400):
        tracker = near_line_tracker(initial_speed=speed)
        mask = self.mask.copy()
        mask[:704, 346:354] = 255
        for timestamp in (0, 0.025, 0.05, 0.075):
            y = 680 + speed * timestamp
            tracker.update([Detection(3, y, has_tail=True, tail_top=0)], mask, timestamp)
        return tracker, mask

    def test_clipped_sustain_recovers_after_brief_occlusion(self):
        tracker, mask = self._clipped_sustain()
        identifier = tracker.tracks[0].identifier
        for index in range(4, 400):
            timestamp = index * 0.025
            hidden = 0.5 <= timestamp < 0.85
            tracker.update([], self.mask if hidden else mask, timestamp)
            if timestamp >= 0.5:
                self.assertEqual(tracker.observation(timestamp)[0][19, 3], 2)
        self.assertEqual([track.identifier for track in tracker.tracks], [identifier])

    def test_clipped_sustain_does_not_recover_after_grace_expires(self):
        # A velocidad baja, la posición aún cae dentro del margen geométrico;
        # eso no debe permitir recuperar una cola después de agotar la gracia.
        for speed in (100, 400):
            with self.subTest(speed=speed):
                tracker, mask = self._clipped_sustain(speed)
                for index in range(4, 400):
                    timestamp = index * 0.025
                    hidden = 0.5 <= timestamp < 1.0
                    tracker.update([], self.mask if hidden else mask, timestamp)
                self.assertFalse(tracker.tracks)
                self.assertEqual(tracker.observation(timestamp)[0][19, 3], 0)

    def test_recovered_sustain_still_ends_when_its_tail_passes(self):
        tracker, mask = self._clipped_sustain()
        for index in range(4, 160):
            timestamp = index * 0.025
            frame_mask = mask.copy()
            # Se recupera del cartel y luego el extremo de la cola entra en la
            # imagen y baja hasta salir: no debe quedar anclado arriba para siempre.
            if 0.5 <= timestamp < 0.85:
                frame_mask[:] = 0
            elif timestamp >= 1.0:
                frame_mask[:int(400 * (timestamp - 1.0))] = 0
            tracker.update([], frame_mask, timestamp)
            if 0.5 <= timestamp <= 2.9:
                self.assertEqual(tracker.observation(timestamp)[0][19, 3], 2)
        self.assertFalse(tracker.tracks)
        self.assertEqual(tracker.observation(timestamp)[0][19, 3], 0)

    def test_new_short_tail_above_does_not_extend_old_sustain(self):
        # También cubre una cola nueva que aparece antes de superar el margen
        # geométrico habitual: estar cerca de y=0 no basta para asociarla.
        for hidden_until in (0.5, 0.85):
            with self.subTest(hidden_until=hidden_until):
                tracker, mask = self._clipped_sustain()
                upper_tail = self.mask.copy()
                upper_tail[:300, 346:354] = 255
                for index in range(4, 160):
                    timestamp = index * 0.025
                    frame_mask = mask if timestamp < 0.5 else self.mask
                    if timestamp >= hidden_until:
                        frame_mask = upper_tail
                    tracker.update([], frame_mask, timestamp)
                self.assertFalse(tracker.tracks)

    def test_disconnected_tail_segments_do_not_recover_clipped_sustain(self):
        tracker, mask = self._clipped_sustain()
        for index in range(4, 34):
            timestamp = index * 0.025
            tracker.update([], mask if timestamp < 0.5 else self.mask, timestamp)
        last_seen = tracker.tracks[0].last_tail_seen
        disconnected = mask.copy()
        disconnected[250:400] = 0
        tracker.update([], disconnected, 0.85)
        self.assertEqual(tracker.tracks[0].last_tail_seen, last_seen)
        for index in range(35, 160):
            tracker.update([], self.mask, index * 0.025)
        self.assertFalse(tracker.tracks)

    def test_latency_advances_observation_without_selecting_keys(self):
        tracker = near_line_tracker(initial_speed=400, latency_ms=50)
        for timestamp, y in ((0, 710), (0.05, 730), (0.1, 750)):
            tracker.update([Detection(4, y)], self.mask, timestamp)
        board, tokens = tracker.observation(0.2)
        self.assertEqual(board[19, 4], 1)
        self.assertIsNotNone(tokens[4])

    def test_estimated_hold_requires_head_and_continuous_action(self):
        held = np.zeros(5, np.int8)
        board = np.zeros((20, 5), np.uint8)
        board[19, :2] = (1, 2)
        held = update_held_estimate(held, board, [1, 1, 0, 0, 0])
        self.assertEqual(held.tolist(), [1, 0, 0, 0, 0])
        board[19, :2] = 2
        held = update_held_estimate(held, board, [1, 0, 0, 0, 0])
        self.assertEqual(held.tolist(), [1, 0, 0, 0, 0])
        held = update_held_estimate(held, board, [0, 0, 0, 0, 0])
        self.assertFalse(held.any())


if __name__ == "__main__":
    unittest.main()
