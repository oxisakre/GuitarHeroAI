"""Pruebas sintéticas de percepción: no capturan pantalla ni envían teclas."""

import unittest

import cv2
import numpy as np

from guitarflash.vision import NoteDetector, tail_rows


class NoteDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = NoteDetector()
        self.frame = np.full((800, 500, 3), 15, dtype=np.uint8)
        self.corners = [(0, 0), (499, 0), (499, 799), (0, 799)]

    def note(self, lane, y, color=(0, 0, 255)):
        cv2.ellipse(self.frame, (lane * 100 + 50, y), (30, 11), 0, 0, 360, color, -1)

    def star(self, lane, y, color=(255, 255, 0)):
        angles = np.arange(10) * np.pi / 5 - np.pi / 2
        radii = np.where(np.arange(10) % 2 == 0, 33, 17)
        points = np.column_stack((lane * 100 + 50 + np.cos(angles) * radii, y + np.sin(angles) * radii))
        cv2.fillPoly(self.frame, [np.rint(points).astype(np.int32)], color)

    def test_colors_and_power_notes_use_lane_not_hue(self):
        colors = [(0, 220, 0), (0, 0, 230), (0, 230, 230), (240, 0, 0), (0, 150, 255)]
        for lane, color in enumerate(colors):
            self.note(lane, 300, color)
            self.note(lane, 550, (255, 255, 0))
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(len(result.heads), 10)
        for lane in range(5):
            notes = [note for note in result.heads if note.lane == lane]
            self.assertEqual(len(notes), 2)
            self.assertAlmostEqual(notes[0].y, 300, delta=2)
            self.assertAlmostEqual(notes[1].y, 550, delta=2)
        self.assertEqual(np.count_nonzero(result.tail_mask), 0)

    def test_round_and_star_heads_coexist(self):
        self.note(1, 260)
        self.star(2, 420)
        self.star(4, 520, (0, 200, 255))
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual([head.lane for head in result.heads], [1, 2, 4])
        self.assertAlmostEqual(result.heads[1].y, 420, delta=6)
        self.assertFalse(any(head.has_tail for head in result.heads))

    def test_chord_sustains_keep_independent_tails(self):
        cv2.line(self.frame, (150, 150), (150, 450), (0, 0, 255), 7)
        cv2.line(self.frame, (350, 280), (350, 450), (255, 0, 0), 7)
        self.note(1, 450)
        self.note(3, 450, (255, 0, 0))
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(len(result.heads), 2)
        self.assertTrue(all(head.has_tail for head in result.heads))
        self.assertAlmostEqual(result.heads[0].tail_top, 146, delta=3)
        self.assertAlmostEqual(result.heads[1].tail_top, 276, delta=3)
        self.assertGreater(np.count_nonzero(result.tail_mask[180:220, 145:156]), 0)
        self.assertEqual(np.count_nonzero(result.tail_mask[180:220, 345:356]), 0)
        self.assertGreater(np.count_nonzero(result.tail_mask[300:350, 345:356]), 0)

    def test_tails_of_outer_lanes_shifted_toward_the_center_are_found(self):
        # Registros reales: en la imagen rectificada la cabeza naranja queda 12 px
        # hacia el centro y su cola 19 px (la verde, igual hacia el otro lado).
        # Buscando la cola sólo en el centro exacto, el seguimiento no la veía.
        for lane, corrimiento, color in ((4, -19, (0, 140, 255)), (0, 19, (0, 200, 0))):
            x = lane * 100 + 50 + corrimiento
            cv2.line(self.frame, (x, 120), (x, 440), color, 7)
            cv2.ellipse(self.frame, (lane * 100 + 50 - (12 if lane == 4 else -12), 452),
                        (30, 11), 0, 0, 360, color, -1)
        result = self.detector.detect(self.frame, self.corners)
        for lane in (0, 4):
            head = next(head for head in result.heads if head.lane == lane)
            self.assertAlmostEqual(head.tail_top, 117, delta=4)
            filas = tail_rows(result.tail_mask, lane, 500)
            self.assertGreater(filas[150:420].mean(), 0.9)

    def test_star_outline_does_not_duplicate_colored_core(self):
        # Como en la captura: núcleo rojo, halo blanco y borde cian separado.
        outer = np.int32([(250, 310), (290, 348), (278, 380), (222, 380), (210, 348)])
        inner = np.rint((outer - (250, 347)) * 0.82 + (250, 347)).astype(np.int32)
        cv2.fillPoly(self.frame, [outer], (255, 255, 0))
        cv2.fillPoly(self.frame, [inner], (230, 230, 230))
        cv2.ellipse(self.frame, (250, 338), (23, 16), 0, 0, 360, (0, 0, 255), -1)
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(len(result.heads), 1)
        self.assertEqual(result.heads[0].lane, 2)
        self.assertAlmostEqual(result.heads[0].y, 338, delta=3)

    def test_thin_tail_is_not_a_head(self):
        cv2.line(self.frame, (250, 100), (250, 600), (0, 255, 255), 5)
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(result.heads, [])
        self.assertGreater(np.count_nonzero(result.tail_mask), 1000)

    def test_variable_tail_width_and_tiny_gaps_preserve_its_top(self):
        # Variación observada en las capturas: un píxel de más no debe partir
        # la cola. Dos filas sin color tampoco son el final del sostenido.
        for y in range(120, 501):
            width = (11, 14, 15, 18)[(y // 4) % 4]
            left = 250 - width // 2
            self.frame[y, left:left + width] = (255, 255, 0)
        self.frame[220:222] = 15
        self.note(2, 500, (255, 255, 0))
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(len(result.heads), 1)
        self.assertTrue(result.heads[0].has_tail)
        self.assertEqual(result.heads[0].tail_top, 120)
        rows = np.any(result.tail_mask[:, 240:261], axis=1)
        self.assertTrue(rows[120:220].all())
        self.assertTrue(rows[222:480].all())
        self.assertFalse(rows[220:222].any(), "No inventar píxeles en el hueco.")

    def test_wider_tail_does_not_create_heads(self):
        self.frame[100:500, 242:260] = (0, 0, 255)
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(result.heads, [])
        self.assertTrue(np.any(result.tail_mask[100:500], axis=1).all())

    def test_large_tail_gap_is_not_joined_to_another_segment(self):
        self.frame[100:250, 243:258] = (0, 0, 255)
        self.frame[300:501, 243:258] = (0, 0, 255)
        self.note(2, 500)
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(len(result.heads), 1)
        self.assertEqual(result.heads[0].tail_top, 300)
        self.assertFalse(result.tail_mask[250:300].any())

    def test_receptors_and_white_lane_guides_are_ignored(self):
        for lane in range(5):
            center = lane * 100 + 50
            cv2.ellipse(self.frame, (center, 783), (35, 17), 0, 0, 360, (0, 255, 0), 5)
            cv2.line(self.frame, (center, 0), (center, 730), (110, 110, 110), 3)
        result = self.detector.detect(self.frame, self.corners)
        self.assertEqual(result.heads, [])
        self.assertEqual(np.count_nonzero(result.tail_mask), 0)
        self.assertEqual(np.count_nonzero(result.mask[self.detector.detection_bottom :]), 0)

    def test_perspective_rectification_preserves_lanes(self):
        for lane in range(5):
            self.note(lane, 470, (255, 200, 0))
        trapezoid = np.float32([(160, 40), (340, 40), (499, 799), (0, 799)])
        transform = cv2.getPerspectiveTransform(np.float32(self.corners), trapezoid)
        frame = cv2.warpPerspective(self.frame, transform, (500, 800))
        result = self.detector.detect(frame, trapezoid)
        self.assertEqual(sorted(head.lane for head in result.heads), list(range(5)))
        self.assertTrue(all(abs(head.y - 470) <= 3 for head in result.heads))

    def test_bad_calibration_is_rejected(self):
        for points in (
            [(0, 0)] * 4,
            [(0, 0), (499, 799), (499, 0), (0, 799)],
            [(-1, 0), (499, 0), (499, 799), (0, 799)],
            [(0, 0), (float("nan"), 0), (499, 799), (0, 799)],
        ):
            with self.subTest(points=points), self.assertRaises(ValueError):
                self.detector.detect(self.frame, points)


if __name__ == "__main__":
    unittest.main()
