"""Capturas sintéticas de sostenidos y efectos; no se envían teclas reales."""

import unittest

import cv2
import numpy as np

from guitar_flash import procesar_captura
from guitarflash.tracking import BoardTracker
from guitarflash.vision import NoteDetector


class SustainEffectsTests(unittest.TestCase):
    def setUp(self):
        self.detector = NoteDetector()
        self.tracker = BoardTracker(initial_speed=400, latency_ms=0)
        self.corners = [(0, 0), (499, 0), (499, 799), (0, 799)]
        self.vivid = (255, 220, 0)
        self.pale = tuple(int(c) for c in cv2.cvtColor(
            np.uint8([[[100, 95, 255]]]), cv2.COLOR_HSV2BGR
        )[0, 0])

    def frame(self):
        return np.full((800, 500, 3), 15, dtype=np.uint8)

    def tail(self, frame, lane, top, bottom, color, width):
        center = lane * 100 + 50
        frame[max(0, top):min(800, bottom),
              center - width // 2:center - width // 2 + width] = color

    def capture(self, frame, timestamp):
        return procesar_captura(
            self.detector, self.tracker, frame, self.corners, timestamp
        )

    def confirm_sustain(self, lane=1):
        for timestamp, y in ((0.0, 120), (0.05, 140), (0.1, 160)):
            frame = self.frame()
            self.tail(frame, lane, 0, y, self.vivid, 12)
            cv2.ellipse(frame, (lane * 100 + 50, y), (30, 11),
                        0, 0, 360, self.vivid, -1)
            self.capture(frame, timestamp)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertTrue(self.tracker.tracks[0].moving)
        self.assertIsNotNone(self.tracker.tracks[0].tail_top)

    def test_ten_second_sustain_survives_brighter_wider_tail_and_finishes(self):
        speed = 400
        hit_time = (799 - 120) / speed
        end_time = hit_time + 10
        for index in range(245):
            timestamp = index * 0.05
            head_y = round(120 + speed * timestamp)
            bright = timestamp >= hit_time
            frame = self.frame()
            self.tail(frame, 1, head_y - 4000, head_y,
                      self.pale if bright else self.vivid, 20 if bright else 12)
            if head_y < 800:
                cv2.ellipse(frame, (150, head_y), (30, 11),
                            0, 0, 360, self.vivid, -1)
            self.capture(frame, timestamp)
            board, _ = self.tracker.observation(timestamp)
            if hit_time + 0.2 <= timestamp <= end_time - 0.1:
                self.assertEqual(board[19, 1], 2, f"Soltó a los {timestamp:.2f}s")
            if timestamp >= end_time + 0.2:
                self.assertFalse(board.any(), "La cola debe terminar, no quedar anclada.")
        self.assertFalse(self.tracker.tracks)

    def test_unconfirmed_pale_shapes_create_neither_tail_nor_heads(self):
        frame = self.frame()
        self.tail(frame, 1, 100, 550, self.pale, 20)
        cv2.ellipse(frame, (150, 550), (30, 11), 0, 0, 360, self.pale, -1)
        result = self.capture(frame, 0.0)
        self.assertEqual(result.heads, [])
        self.assertFalse(result.tail_mask.any())
        self.assertFalse(self.tracker.tracks)

    def test_confirmed_lane_does_not_relax_neighbor_or_head_filter(self):
        self.confirm_sustain(lane=1)
        frame = self.frame()
        for lane in (1, 2):
            self.tail(frame, lane, 0, 700, self.pale, 20)
        cv2.ellipse(frame, (150, 350), (30, 11), 0, 0, 360, self.pale, -1)
        result = self.capture(frame, 0.15)
        self.assertTrue(result.tail_mask[:, 100:200].any())
        self.assertFalse(result.tail_mask[:, 200:300].any())
        self.assertEqual(result.heads, [])
        self.assertTrue(all(track.lane == 1 for track in self.tracker.tracks))

    def test_white_and_gray_guides_are_ignored_in_confirmed_lanes(self):
        frame = self.frame()
        for lane, value in ((1, 220), (3, 90)):
            self.tail(frame, lane, 30, 680, (value, value, value), 16)
        result = self.detector.detect(frame, self.corners, sustain_lanes=(1, 3))
        self.assertEqual(result.heads, [])
        self.assertFalse(result.tail_mask.any())

    def test_confirmed_tail_tolerates_lower_brightness(self):
        self.confirm_sustain(lane=1)
        frame = self.frame()
        self.tail(frame, 1, 0, 700, (0, 0, 80), 12)
        self.assertFalse(self.detector.detect(frame, self.corners).tail_mask.any())
        result = self.capture(frame, 0.15)
        self.assertTrue(np.any(result.tail_mask[:700, 100:200], axis=1).all())
        self.assertEqual(result.heads, [])

    def test_short_pale_sign_strokes_do_not_extend_a_sustain(self):
        self.confirm_sustain(lane=1)
        frame = self.frame()
        # Trazos verticales de un cartel: claros, centrados y separados.
        for top in (40, 160, 280, 400):
            self.tail(frame, 1, top, top + 40, self.pale, 20)
        for timestamp in np.arange(0.15, 3.1, 0.15):
            result = self.capture(frame, float(timestamp))
            self.assertEqual(result.heads, [])
            self.assertFalse(result.tail_mask.any())
        self.assertFalse(self.tracker.tracks)


if __name__ == "__main__":
    unittest.main()
