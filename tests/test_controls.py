import unittest

from guitarflash.controls import KeyController


class FakeKeyboard:
    def __init__(self):
        self.events = []
        self.fail_release = set()

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))
        if key in self.fail_release:
            raise OSError(f"No se pudo soltar {key}")


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeKeyboard()
        self.controller = KeyController(self.backend)

    def test_chord_and_sustains_use_transitions_only(self):
        self.controller.apply([1, 0, 0, 1, 0], [10, None, None, 20, None])
        self.controller.apply([1, 0, 0, 1, 0], [10, None, None, 20, None])
        self.controller.apply([1, 0, 0, 1, 0])
        self.assertEqual(self.backend.events, [("down", "a"), ("down", "f")])
        self.controller.apply([0, 0, 0, 1, 0])
        self.assertEqual(self.backend.events[-1], ("up", "a"))

    def test_new_head_retriggers_only_lane_requested_by_ppo(self):
        self.controller.apply([1, 0, 0, 0, 0], [1, None, None, None, None])
        self.controller.apply([1, 0, 0, 0, 0], [2, 3, None, None, None])
        self.assertEqual(self.backend.events,
                         [("down", "a"), ("up", "a"), ("down", "a")])

    def test_detection_alone_never_presses_a_key(self):
        self.controller.apply([0] * 5, [1, 2, 3, 4, 5])
        self.assertEqual(self.backend.events, [])

    def test_release_all_clears_keys_and_tokens(self):
        self.controller.apply([1, 1, 0, 0, 0], [1, 2, None, None, None])
        self.controller.release_all()
        self.assertEqual(self.backend.events[-2:], [("up", "a"), ("up", "s")])
        self.assertFalse(any(self.controller.pressed))
        self.assertEqual(self.controller.tokens, [None] * 5)
        count = len(self.backend.events)
        self.controller.release_all()
        self.assertEqual(len(self.backend.events), count)

    def test_release_failure_still_attempts_every_pressed_key(self):
        self.controller.apply([1, 1, 0, 0, 0])
        self.backend.fail_release.add("a")
        with self.assertRaises(OSError):
            self.controller.release_all()
        self.assertEqual(self.backend.events[-2:], [("up", "a"), ("up", "s")])
        self.assertEqual(self.controller.pressed, [True, False, False, False, False])

    def test_retrigger_keeps_key_released_for_min_release(self):
        controller = KeyController(self.backend, min_release=0.025)
        controller.apply([1, 0, 0, 0, 0], [1, None, None, None, None], now=0.0)
        controller.apply([1, 0, 0, 0, 0], [2, None, None, None, None], now=0.100)
        self.assertEqual(self.backend.events, [("down", "a"), ("up", "a")])
        self.assertAlmostEqual(controller.next_press_time(), 0.125)
        controller.flush(now=0.110)
        # La misma cabeza en la siguiente decisión no vuelve a soltar la tecla.
        controller.apply([1, 0, 0, 0, 0], [2, None, None, None, None], now=0.117)
        self.assertEqual(len(self.backend.events), 2)
        controller.flush(now=0.126)
        self.assertEqual(self.backend.events[-1], ("down", "a"))
        self.assertIsNone(controller.next_press_time())

    def test_min_release_also_applies_after_ppo_releases(self):
        controller = KeyController(self.backend, min_release=0.025)
        controller.apply([0, 1, 0, 0, 0], now=0.0)
        controller.apply([0, 0, 0, 0, 0], now=0.050)
        controller.apply([0, 1, 0, 0, 0], now=0.060)
        self.assertEqual(self.backend.events, [("down", "s"), ("up", "s")])
        controller.apply([0, 1, 0, 0, 0], now=0.076)
        self.assertEqual(self.backend.events[-1], ("down", "s"))

    def test_pending_press_is_cancelled_when_ppo_changes_its_mind(self):
        controller = KeyController(self.backend, min_release=0.025)
        controller.apply([1, 0, 0, 0, 0], now=0.0)
        controller.apply([0, 0, 0, 0, 0], now=0.050)
        controller.apply([1, 0, 0, 0, 0], now=0.060)
        controller.apply([0, 0, 0, 0, 0], now=0.070)
        controller.flush(now=0.200)
        self.assertEqual(self.backend.events, [("down", "a"), ("up", "a")])
        controller.apply([1, 0, 0, 0, 0], now=0.300)
        controller.release_all()
        controller.flush(now=0.400)
        self.assertEqual(self.backend.events[-1], ("up", "a"))

    def test_invalid_action_sends_no_events(self):
        for action in ([0, 0], [1, 0, 0, 0, 2]):
            with self.assertRaises(ValueError):
                self.controller.apply(action)
        self.assertEqual(self.backend.events, [])


if __name__ == "__main__":
    unittest.main()
