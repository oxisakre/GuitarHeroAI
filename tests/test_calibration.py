"""Flujo de calibración con ventanas simuladas: no abre nada en pantalla."""

import contextlib
import io
import unittest

import cv2
import numpy as np

from guitar_flash import countdown
from guitarflash.calibration import (BAND_HEIGHT, CORNER_SCREEN_FRACTION, REGION_SCREEN_FRACTION,
                                     calibrate, fit_scale, select_region, view_limit)

ENTER = 13
SCREEN = (1920, 1080)


class FakeGui:
    """Imita las ventanas de OpenCV; cada waitKey ejecuta el siguiente paso del guion."""

    def __init__(self, script):
        self.script = list(script)
        self.callbacks = {}
        self.open = set()
        self.current = None
        self.shown = []

    def namedWindow(self, name, flags):
        self.open.add(name)

    def setWindowProperty(self, name, prop, value):
        pass

    def setMouseCallback(self, name, callback):
        self.callbacks[name] = callback
        self.current = name

    def imshow(self, name, image):
        self.shown.append((name, image.shape))

    def getWindowProperty(self, name, prop):
        return 1.0 if name in self.open else 0.0

    def destroyWindow(self, name):
        if name not in self.open:
            raise cv2.error("NULL window")  # Como OpenCV real tras cerrar con la X.
        self.open.discard(name)

    def waitKey(self, delay):
        if not self.script:
            raise AssertionError("El guion terminó sin cerrar la calibración.")
        step = self.script.pop(0)
        if step[0] == "mouse":
            _, event, x, y = step
            self.callbacks[self.current](event, x, y, 0, None)
            return -1
        if step[0] == "close":
            self.open.discard(self.current)
            return -1
        return step[1]


def window_point(point, scale):
    """Píxel de la imagen -> coordenada de clic en la ventana (con la franja de texto)."""
    return round(point[0] * scale), round(point[1] * scale) + BAND_HEIGHT


def drag(start, end, scale):
    x0, y0 = window_point(start, scale)
    x1, y1 = window_point(end, scale)
    return [("mouse", cv2.EVENT_LBUTTONDOWN, x0, y0),
            ("mouse", cv2.EVENT_MOUSEMOVE, (x0 + x1) // 2, (y0 + y1) // 2),
            ("mouse", cv2.EVENT_LBUTTONUP, x1, y1)]


def clicks(points, scale):
    return [("mouse", cv2.EVENT_LBUTTONDOWN, *window_point(point, scale)) for point in points]


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.screen = np.zeros((1080, 1920, 3), np.uint8)
        self.region_scale = fit_scale(1920, 1080, view_limit(SCREEN, REGION_SCREEN_FRACTION))
        # Región de x 400..1000, y 200..900; la reducción la desplaza un par de píxeles.
        self.region_drag = drag((400, 200), (1000, 900), self.region_scale) + [("key", ENTER)]
        _, _, width, height = select_region(self.screen, FakeGui(self.region_drag), SCREEN)
        self.crop_scale = fit_scale(width, height, view_limit(SCREEN, CORNER_SCREEN_FRACTION), 2.0)
        self.corners = [(200, 60), (400, 60), (590, 690), (10, 690)]

    def run_calibration(self, script, screen=SCREEN):
        gui = FakeGui(script)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return calibrate(self.screen, offset=(1920, 0), keys="asdfg", gui=gui, screen=screen), gui
        finally:
            self.assertFalse(gui.open, "Las ventanas deben cerrarse siempre.")

    def test_full_screen_capture_is_shown_reduced_and_mapped_back(self):
        script = self.region_drag + clicks(self.corners, self.crop_scale) + [("key", ENTER)]
        config, gui = self.run_calibration(script)
        region = config["region"]
        self.assertAlmostEqual(region["left"], 1920 + 400, delta=2)
        self.assertAlmostEqual(region["top"], 200, delta=2)
        self.assertAlmostEqual(region["width"], 601, delta=3)
        self.assertAlmostEqual(region["height"], 701, delta=3)
        for actual, expected in zip(config["corners"], self.corners):
            self.assertAlmostEqual(actual[0], expected[0], delta=1)
            self.assertAlmostEqual(actual[1], expected[1], delta=1)
        self.assertEqual(config["keys"], "asdfg")

    def test_windows_fit_on_a_small_laptop_screen(self):
        laptop = (1366, 768)
        scale = fit_scale(1920, 1080, view_limit(laptop, REGION_SCREEN_FRACTION))
        region = drag((400, 200), (1000, 900), scale) + [("key", ENTER)]
        _, _, width, height = select_region(self.screen, FakeGui(region), laptop)
        crop_scale = fit_scale(width, height, view_limit(laptop, CORNER_SCREEN_FRACTION), 2.0)
        script = region + clicks(self.corners, crop_scale) + [("key", ENTER)]
        _, gui = self.run_calibration(script, screen=laptop)
        for name, (height, width, _) in gui.shown:
            fraction = REGION_SCREEN_FRACTION if "1/2" in name else CORNER_SCREEN_FRACTION
            self.assertLessEqual(height, laptop[1] * fraction, name)
            self.assertLessEqual(width, laptop[0] * fraction, name)

    def test_escape_cancels_first_step(self):
        with self.assertRaisesRegex(ValueError, "cancelada"):
            self.run_calibration([("mouse", cv2.EVENT_MOUSEMOVE, 10, 10), ("key", 27)])

    def test_closing_window_cancels_without_opencv_error(self):
        with self.assertRaisesRegex(ValueError, "cancelada"):
            self.run_calibration(self.region_drag + [("close",)])

    def test_tiny_drag_and_clicks_on_instructions_are_ignored(self):
        script = (drag((400, 200), (405, 205), self.region_scale) + [("key", ENTER)]
                  + [("mouse", cv2.EVENT_LBUTTONDOWN, 50, BAND_HEIGHT // 2), ("key", ENTER), ("key", 27)])
        with self.assertRaisesRegex(ValueError, "cancelada"):
            self.run_calibration(script)

    def test_wrong_corner_order_can_be_redone(self):
        wrong = [self.corners[1], self.corners[0], self.corners[2], self.corners[3]]
        script = (self.region_drag
                  + clicks(wrong, self.crop_scale) + [("key", ENTER), ("key", ord("r"))]
                  + clicks(self.corners, self.crop_scale) + [("key", ENTER)])
        config, _ = self.run_calibration(script)
        self.assertAlmostEqual(config["corners"][0][0], 200, delta=1)

    def test_small_track_is_enlarged_for_precise_clicks(self):
        self.assertEqual(fit_scale(300, 350, view_limit(SCREEN, CORNER_SCREEN_FRACTION), 2.0), 2.0)

    def test_countdown_waits_one_second_per_step(self):
        waits = []
        with contextlib.redirect_stdout(io.StringIO()):
            countdown(3, sleep=waits.append)
        self.assertEqual(waits, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
