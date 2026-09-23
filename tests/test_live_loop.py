"""Traducción a teclas del bucle en vivo, con teclado falso y sin capturar pantalla."""

import time
import unittest

import numpy as np

from guitar_flash import send_action, wait_until
from guitarflash.controls import KeyController
from guitarflash.tracking import BoardTracker
from guitarflash.vision import Detection


class TimedKeyboard:
    def __init__(self, window=1):
        self.events = []
        self.window = window

    def key_down(self, key):
        self.events.append(("down", key, time.perf_counter()))

    def key_up(self, key):
        self.events.append(("up", key, time.perf_counter()))

    def foreground(self):
        return self.window


class LiveLoopTests(unittest.TestCase):
    def test_next_head_in_same_lane_releases_key_before_it_arrives(self):
        tracker = BoardTracker(initial_speed=400, entry_ratio=1.0)  # Notas ya cerca de la línea.
        mask = np.zeros((800, 500), np.uint8)
        # Dos cabezas del carril 1 separadas 40 px (100 ms a 400 px/s).
        for timestamp in (-0.1, -0.05, 0.0, 0.05):
            y = 400 * timestamp
            tracker.update([Detection(0, 700 + y), Detection(0, 660 + y)], mask, timestamp)
        keyboard = TimedKeyboard()
        controller = KeyController(keyboard, min_release=0.025)
        press = [1, 0, 0, 0, 0]
        # La primera cabeza llega a y=799 en t≈0.2475; la segunda en t≈0.3475.
        send_action(controller, tracker, press, 0.25)
        self.assertEqual([event[0] for event in keyboard.events], ["down"])
        send_action(controller, tracker, press, 0.30)
        self.assertEqual([event[0] for event in keyboard.events], ["down"])
        # A menos de min_release de la segunda cabeza suelta, aunque PPO mantenga 1.
        send_action(controller, tracker, press, 0.33)
        self.assertEqual([event[0] for event in keyboard.events], ["down", "up"])
        self.assertAlmostEqual(controller.next_press_time(), 0.355)
        send_action(controller, tracker, press, 0.35)
        self.assertEqual(len(keyboard.events), 2)
        controller.flush(0.356)
        self.assertEqual([event[0] for event in keyboard.events], ["down", "up", "down"])

    def test_wait_until_sends_pending_press_on_time(self):
        keyboard = TimedKeyboard()
        controller = KeyController(keyboard, min_release=0.02)
        start = time.perf_counter()
        controller.apply([1, 0, 0, 0, 0], [1, None, None, None, None], start)
        controller.apply([1, 0, 0, 0, 0], [2, None, None, None, None], start)
        wait_until(start + 0.08, controller, keyboard, target=1)
        finished = time.perf_counter()
        kinds = [event[0] for event in keyboard.events]
        self.assertEqual(kinds, ["down", "up", "down"])
        pressed = keyboard.events[-1][2] - start
        self.assertGreaterEqual(pressed, 0.02)
        self.assertLess(pressed, 0.05)
        self.assertGreaterEqual(finished - start, 0.08)

    def test_wait_until_does_not_press_after_focus_changes(self):
        keyboard = TimedKeyboard(window=2)
        controller = KeyController(keyboard, min_release=0.01)
        start = time.perf_counter()
        controller.apply([1, 0, 0, 0, 0], [1, None, None, None, None], start)
        controller.apply([1, 0, 0, 0, 0], [2, None, None, None, None], start)
        wait_until(start + 0.03, controller, keyboard, target=1)
        self.assertEqual([event[0] for event in keyboard.events], ["down", "up"])


if __name__ == "__main__":
    unittest.main()
