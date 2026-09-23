"""Registro --debug: sin teclado real ni ventanas."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from guitarflash.debug import DebugLog, RecordingKeyboard
from guitarflash.tracking import BoardTracker
from guitarflash.vision import Detection


class FakeKeyboard:
    def __init__(self):
        self.events = []

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))


class DebugLogTests(unittest.TestCase):
    def test_precise_capture_and_tail_inputs_can_replay_a_sustain(self):
        # La captura y la observación ocurren en instantes distintos. El replay
        # debe conservar ambos, las posiciones fraccionarias y la cola visible,
        # incluso cuando la cabeza ya salió y hay una oclusión breve.
        with tempfile.TemporaryDirectory() as folder:
            log = DebugLog(Path(folder) / "sesion")
            tracker = BoardTracker(initial_speed=400, entry_ratio=1.0)
            mask = np.zeros((800, 500), np.uint8)
            mask[:704, 346:354] = 255
            mask[:3, 48:52] = 255
            mask[6:12, 48:52] = 255
            empty = np.zeros_like(mask)
            view = np.zeros((40, 60, 3), np.uint8)
            expected = []
            try:
                for index in range(140):
                    captured = index / 40 + 0.000037
                    now = captured + 0.006123
                    heads = ([Detection(3, 680.125 + 400 * index / 40,
                                        height=22.75, has_tail=True, tail_top=0.0)]
                             if index < 4 else [])
                    visible = empty if .5 <= captured < .85 else mask
                    tracker.update(heads, visible, captured)
                    board, tokens = tracker.observation(now)
                    expected.append((board.copy(), tokens, tracker.speed))
                    log.record(now, [0] * 5, board, heads, tracker, [], view,
                               captured=captured, tail_mask=visible)
            finally:
                log.close()
            rows = [json.loads(line) for line in
                    (log.folder / "eventos.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["version"], 2)
            self.assertEqual(rows[0]["captured"], 0.000037)
            self.assertNotEqual(rows[0]["observed"], rows[0]["t"])
            self.assertEqual(rows[0]["detalles_detecciones"][0]["y"], 680.125)
            self.assertEqual(rows[0]["colas_por_carril"][0], [[0, 3], [6, 12]])
            self.assertEqual(rows[0]["detalles_seguimiento"][0]["last_seen"], 0.000037)
            self.assertEqual(rows[0]["detalles_seguimiento"][0]["tail_top"], 0.0)
            self.assertTrue(rows[0]["detalles_seguimiento"][0]["tail_clipped"])
            self.assertEqual(rows[0]["detalles_seguimiento"][0]["last_tail_seen"], 0.000037)
            self.assertEqual(rows[-1]["detalles_detecciones"], [])

            replay = BoardTracker(initial_speed=400, entry_ratio=1.0)
            for row, (board, tokens, speed) in zip(rows, expected):
                heads = [Detection(h["carril"] - 1, h["y"], h["height"],
                                   h["has_tail"], h["tail_top"])
                         for h in row["detalles_detecciones"]]
                replay_mask = np.zeros_like(mask)
                for lane, runs in enumerate(row["colas_por_carril"]):
                    x = int((lane + 0.5) * replay.width / 5)
                    for start, end in runs:
                        replay_mask[start:end, x] = 255
                replay.update(heads, replay_mask, row["captured"])
                actual_board, actual_tokens = replay.observation(row["observed"])
                np.testing.assert_array_equal(actual_board, board)
                self.assertEqual(actual_tokens, tokens)
                self.assertAlmostEqual(replay.speed, speed)

    def test_recording_keyboard_forwards_and_remembers_keys(self):
        keyboard = FakeKeyboard()
        recorder = RecordingKeyboard(keyboard)
        recorder.key_down("a")
        recorder.key_up("a")
        self.assertEqual(keyboard.events, [("down", "a"), ("up", "a")])
        self.assertEqual([kind for _, _, kind in recorder.take_sent()], ["down", "up"])
        self.assertEqual(recorder.take_sent(), [])

    def test_quick_repress_is_logged_with_a_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            log = DebugLog(Path(folder) / "sesion")
            tracker = BoardTracker()
            board = np.zeros((20, 5), np.uint8)
            view = np.zeros((40, 60, 3), np.uint8)
            heads = [Detection(0, 700.0)]
            log.record(1.0, [1, 0, 0, 0, 0], board, heads, tracker, [(1.0, "a", "down")], view)
            log.record(1.2, [1, 0, 0, 0, 0], board, heads, tracker,
                       [(1.1, "a", "up"), (1.2, "a", "down")], view)
            log.record(2.0, [1, 0, 0, 0, 0], board, heads, tracker,
                       [(1.5, "a", "up"), (2.0, "a", "down")], view)
            log.close()
            lines = [json.loads(line) for line in (log.folder / "eventos.jsonl").read_text().splitlines()]
            self.assertEqual([line["repulsas"] for line in lines], [[], ["a"], []])
            self.assertEqual(lines[0]["detecciones"], [[1, 700]])
            self.assertIsNone(lines[0]["captured"])
            self.assertIsNone(lines[0]["colas_por_carril"])
            self.assertEqual(len(list(log.folder.glob("repulsa_*.jpg"))), 1)
            # La suelta de 1.2 a 1.5 dura 0.3 s; la de 2.0 en adelante, 1.4 s.
            self.assertEqual([line["fin_sostenido"] for line in lines], [[], [], []])

    def test_end_of_a_long_hold_is_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            log = DebugLog(Path(folder) / "sesion")
            tracker = BoardTracker()
            board = np.zeros((20, 5), np.uint8)
            view = np.zeros((40, 60, 3), np.uint8)
            log.record(1.0, [1, 0, 0, 0, 0], board, [], tracker, [(1.0, "a", "down")], view)
            log.record(3.0, [0, 0, 0, 0, 0], board, [], tracker, [(3.0, "a", "up")], view, view)
            log.close()
            lines = [json.loads(line) for line in (log.folder / "eventos.jsonl").read_text().splitlines()]
            self.assertEqual([line["fin_sostenido"] for line in lines], [[], ["a"]])
            self.assertEqual(len(list(log.folder.glob("soltada_*"))), 2)


if __name__ == "__main__":
    unittest.main()
