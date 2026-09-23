"""Integración de visión y tablero con secuencias, sin teclado ni políticas."""

import unittest

import cv2
import numpy as np

from guitarflash.tracking import BoardTracker
from guitarflash.vision import NoteDetector


class PerceptionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.detector = NoteDetector()
        self.tracker = BoardTracker(initial_speed=400.0)
        self.corners = [(0, 0), (499, 0), (499, 799), (0, 799)]
        self.step = 1 / 40

    @staticmethod
    def frame(notes):
        """notes: (carril, centro vertical, longitud de cola en píxeles)."""
        frame = np.full((800, 500, 3), 15, dtype=np.uint8)
        colors = [(0, 220, 0), (0, 0, 230), (0, 230, 230), (240, 0, 0), (0, 150, 255)]
        for lane, y, tail_length in notes:
            center = lane * 100 + 50
            y = int(round(y))
            if tail_length:
                cv2.line(frame, (center, y - tail_length), (center, y), colors[lane], 7)
            cv2.ellipse(frame, (center, y), (30, 11), 0, 0, 360, colors[lane], -1)
        return frame

    def update(self, notes, timestamp):
        result = self.detector.detect(self.frame(notes), self.corners)
        self.tracker.update(result.heads, result.tail_mask, timestamp)
        board, tokens = self.tracker.observation(timestamp)
        return result, board, tokens

    def test_two_heads_cross_blind_band_once_each_and_static_shape_is_ignored(self):
        seen = {}
        vanished_before_hit = False
        for index in range(101):
            timestamp = index * self.step
            notes = [(1, 370 + 400 * timestamp, 0), (1, 120 + 400 * timestamp, 0), (4, 350, 0)]
            result, board, tokens = self.update(notes, timestamp)
            self.assertFalse(board[:, 4].any(), "Una forma inmóvil no debe entrar al tablero de PPO.")
            self.assertIsNone(tokens[4])
            if tokens[1] is not None:
                seen.setdefault(tokens[1], []).append(index)
                self.assertEqual(board[19, 1], 1)
                # En este momento la visión ya no ve esa cabeza: el seguimiento
                # ha de cruzar la banda excluida sin crear una identidad nueva.
                self.assertFalse(any(head.lane == 1 and head.y > self.detector.detection_bottom for head in result.heads))
                track = next(track for track in self.tracker.tracks if track.identifier == tokens[1])
                self.assertGreater(timestamp - track.last_seen, 0.1)
                vanished_before_hit = True
            self.assertFalse(np.any(board == 2), "Las notas simples no deben crear colas.")
        self.assertTrue(vanished_before_hit)
        self.assertEqual(len(seen), 2, "Cada nota física debe generar una única identidad en la fila 19.")
        starts = []
        for indices in seen.values():
            self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))
            starts.append(indices[0] * self.step)
        expected = [(799 - 370) / 400, (799 - 120) / 400]
        for actual, desired in zip(sorted(starts), expected):
            self.assertAlmostEqual(actual, desired, delta=2 * self.step)
        self.assertFalse(board.any())

    def test_long_sustain_survives_identical_captures(self):
        # Con la cabeza fuera del recorte, una cola larga que entra por arriba da
        # capturas idénticas durante segundos. Si esas capturas no se procesan, la
        # cola se extrapola sola hasta salir del tablero y el sostenido se suelta.
        from guitar_flash import procesar_captura
        tracker = BoardTracker(initial_speed=400)
        sostenido, vio_cola = None, False
        for index in range(241):
            timestamp = index / 40
            y = 100 + 400 * timestamp
            frame = np.full((800, 500, 3), 15, np.uint8)
            cv2.line(frame, (350, int(y) - 4000), (350, int(y)), (240, 0, 0), 7)
            cv2.ellipse(frame, (350, int(y)), (30, 11), 0, 0, 360, (240, 0, 0), -1)
            procesar_captura(self.detector, tracker, frame, self.corners, timestamp)
            board, _ = tracker.observation(timestamp)
            vio_cola |= board[19, 3] == 2
            if vio_cola and board[19, 3] == 0 and sostenido is None:
                sostenido = timestamp
        self.assertIsNone(sostenido, "La cola de 10 s no debe soltarse a mitad de camino.")

    def test_repeated_notes_in_one_lane_keep_their_own_timing(self):
        # Cada nota pasa por el último punto visible poco después de que la
        # anterior entró en la banda ciega; no debe quedarse con su seguimiento.
        for spacing in (40, 140):
            with self.subTest(spacing=spacing):
                tracker = BoardTracker(initial_speed=400.0)
                starts = [200 - index * spacing for index in range(4)]
                first_seen = {}
                for index in range(int(2.6 / self.step)):
                    timestamp = index * self.step
                    notes = [(3, start + 400 * timestamp, 0) for start in starts]
                    result = self.detector.detect(self.frame(notes), self.corners)
                    tracker.update(result.heads, result.tail_mask, timestamp)
                    board, tokens = tracker.observation(timestamp)
                    if tokens[3] is not None:
                        first_seen.setdefault(tokens[3], timestamp)
                self.assertEqual(len(first_seen), len(starts))
                expected = [(799 - start) / 400 for start in starts]
                for actual, desired in zip(sorted(first_seen.values()), expected):
                    self.assertAlmostEqual(actual, desired, delta=1.5 * self.step)

    def test_sustains_entering_from_above_continue_after_heads_and_end_separately(self):
        histories = {1: [], 3: []}
        head_ids = {1: set(), 3: set()}
        initial_tail_clipped = False
        for index in range(221):
            timestamp = index * self.step
            y = 250 + 400 * timestamp
            result, board, tokens = self.update([(1, y, 950), (3, y, 650)], timestamp)
            if index == 0:
                initial_tail_clipped = all(head.has_tail and head.tail_top == 0 for head in result.heads)
            for lane in (1, 3):
                histories[lane].append(int(board[19, lane]))
                if tokens[lane] is not None:
                    head_ids[lane].add(tokens[lane])
        self.assertTrue(initial_tail_clipped)
        for lane in (1, 3):
            history = np.asarray(histories[lane])
            heads = np.flatnonzero(history == 1)
            tails = np.flatnonzero(history == 2)
            self.assertEqual(len(head_ids[lane]), 1)
            self.assertGreater(heads.size, 0)
            self.assertGreater(tails.size, 0)
            self.assertGreater(tails[0], heads[-1])
            self.assertEqual(tails.tolist(), list(range(int(tails[0]), int(tails[-1]) + 1)))
            self.assertTrue(np.all(history[tails[-1] + 1 :] == 0))
        red_end = np.flatnonzero(np.asarray(histories[1]) == 2)[-1] * self.step
        blue_end = np.flatnonzero(np.asarray(histories[3]) == 2)[-1] * self.step
        self.assertAlmostEqual(red_end - blue_end, (950 - 650) / 400, delta=3 * self.step)
        self.assertFalse(board.any())


if __name__ == "__main__":
    unittest.main()
