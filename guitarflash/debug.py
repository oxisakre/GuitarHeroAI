"""Registro opcional (--debug) para diagnosticar el control en el juego real."""
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .vision import tail_rows

REPRESS_SECONDS = 0.4
LONG_HOLD_SECONDS = 0.4
MAX_SNAPSHOTS = 60


def tail_row_runs(tail_mask, width):
    """Filas visibles por carril, en intervalos [inicio, fin).

    Conserva exactamente la presencia de cola que consulta el tracker, sin
    escribir una imagen completa por cuadro. None indica que no fue registrada.
    """
    if tail_mask is None:
        return None
    lanes = []
    for lane in range(5):
        rows = tail_rows(tail_mask, lane, width)
        edges = np.flatnonzero(np.diff(np.r_[False, rows, False].astype(np.int8)))
        lanes.append(edges.reshape(-1, 2).tolist())
    return lanes


class RecordingKeyboard:
    """Envía las teclas al teclado real y anota cuándo se enviaron."""

    def __init__(self, keyboard):
        self.keyboard = keyboard
        self.sent = []

    def key_down(self, key):
        self.keyboard.key_down(key)
        self.sent.append((time.perf_counter(), key, "down"))

    def key_up(self, key):
        self.keyboard.key_up(key)
        self.sent.append((time.perf_counter(), key, "up"))

    def take_sent(self):
        sent, self.sent = self.sent, []
        return sent


class DebugLog:
    """Escribe eventos.jsonl y guarda la vista cuando una tecla se suelta y se
    vuelve a apretar en menos de REPRESS_SECONDS (el síntoma de sostenidos cortados)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = (self.folder / "eventos.jsonl").open("w", encoding="utf-8")
        self.last_up = {}
        self.last_down = {}
        self.snapshots = 0
        self.releases = 0
        # Guardar imágenes en otro hilo para no frenar el bucle de captura.
        self.pending = queue.Queue()
        self.worker = threading.Thread(target=self._save_images, daemon=True)
        self.worker.start()

    def record(self, now, action, board, heads, tracker, sent, view, track_image=None,
               *, captured=None, tail_mask=None):
        represses, sueltas = [], []
        for sent_at, key, kind in sent:
            if kind == "up":
                self.last_up[key] = sent_at
                if sent_at - self.last_down.get(key, sent_at) > LONG_HOLD_SECONDS:
                    sueltas.append(key)      # fin de un sostenido: ¿se soltó a tiempo?
            else:
                self.last_down[key] = sent_at
                if sent_at - self.last_up.get(key, -1e9) < REPRESS_SECONDS:
                    represses.append(key)
        self.stream.write(json.dumps({
            "version": 2,
            "t": round(now, 4),
            # Los campos viejos se conservan para los analizadores existentes.
            # El instante del tablero es posterior al de la imagen capturada.
            "captured": None if captured is None else float(captured),
            "observed": float(now),
            "accion": [int(value) for value in action],
            "fila19": board[19].tolist(),
            "detecciones": [[head.lane + 1, round(head.y)] for head in heads],
            "notas": [[track.identifier, track.lane + 1, round(track.y), track.moving]
                      for track in tracker.tracks],
            "velocidad": round(tracker.speed),
            "velocidad_exacta": float(tracker.speed),
            "detalles_detecciones": [
                {"carril": head.lane + 1, "y": float(head.y),
                 "height": float(head.height), "has_tail": bool(head.has_tail),
                 "tail_top": None if head.tail_top is None else float(head.tail_top)}
                for head in heads
            ],
            "detalles_seguimiento": [
                {"id": track.identifier, "carril": track.lane + 1,
                 "y": float(track.y), "last_seen": float(track.last_seen),
                 "velocity": float(track.velocity), "moving": bool(track.moving),
                 "tail_top": None if track.tail_top is None else float(track.tail_top),
                 "last_tail_seen": (None if track.last_tail_seen is None
                                    else float(track.last_tail_seen)),
                 "tail_clipped": bool(track.tail_clipped)}
                for track in tracker.tracks
            ],
            "colas_por_carril": tail_row_runs(tail_mask, tracker.width),
            "enviadas": [[round(sent_at, 4), key, kind] for sent_at, key, kind in sent],
            "repulsas": represses,
            "fin_sostenido": sueltas,
        }) + "\n")
        if sueltas and self.releases < MAX_SNAPSHOTS:
            self.releases += 1
            self.pending.put((self.folder / f"soltada_{self.releases:02d}_tecla_{'_'.join(sueltas)}.jpg", view.copy()))
            if track_image is not None:
                self.pending.put((self.folder / f"soltada_{self.releases:02d}_pista.png", track_image.copy()))
        if represses and self.snapshots < MAX_SNAPSHOTS:
            self.snapshots += 1
            name = f"repulsa_{self.snapshots:02d}_tecla_{'_'.join(represses)}.jpg"
            self.pending.put((self.folder / name, view.copy()))
            if track_image is not None:
                # La pista sin dibujos encima: sirve para ajustar colores después.
                self.pending.put((self.folder / f"pista_{self.snapshots:02d}.png", track_image.copy()))

    def _save_images(self):
        while (item := self.pending.get()) is not None:
            path, image = item
            cv2.imwrite(str(path), image)

    def close(self):
        self.pending.put(None)
        self.worker.join(timeout=5)
        self.stream.close()
