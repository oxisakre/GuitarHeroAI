"""Percepción inicial de Guitar Flash; no toma decisiones ni envía teclas.

Las coordenadas de salida pertenecen a la pista rectificada. El color ayuda a
encontrar notas, pero el carril determina su identidad: también se aceptan
estrellas y notas azules/cian. Es una heurística que requiere calibración y
verificación visual para cada tamaño, tema y calidad de captura.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# En la imagen rectificada los carriles de afuera quedan corridos hacia el centro
# (en los registros del usuario, ±12 px las cabezas y ±19 las colas): la pista real
# es algo más angosta que el rectángulo calibrado. Buscar la cola sólo en el centro
# exacto del carril perdía las naranjas y azules largas: se soltaban a los 1-2 s.
TAIL_CENTER_RATIO = 0.25


def tail_rows(tail_mask, lane, width):
    """Filas con cola del carril, dentro de su franja central."""
    lane_width = width / 5.0
    center = int(round(lane_width * (lane + 0.5)))
    radius = max(2, int(round(lane_width * TAIL_CENTER_RATIO)))
    return np.any(tail_mask[:, max(0, center - radius):center + radius + 1] > 0, axis=1)


@dataclass(frozen=True)
class Detection:
    lane: int
    y: float
    height: float = 1.0
    has_tail: bool = False
    tail_top: float | None = None


@dataclass
class VisionResult:
    warped: np.ndarray
    mask: np.ndarray
    heads: list[Detection]
    tail_mask: np.ndarray


def _runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Devuelve intervalos [inicio, fin) de filas activas consecutivas."""
    transitions = np.diff(np.r_[False, rows, False].astype(np.int8))
    return list(zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))


def _tail_runs(widths, centered, wide_rows, max_width, min_rows):
    """Colas uniformes; une hasta dos filas vacías sin atravesar una cabeza."""
    blocked = np.zeros_like(centered, dtype=bool)
    for start, end in _runs(wide_rows):
        blocked[max(0, start - 2):min(len(blocked), end + 2)] = True
    narrow = (widths > 0) & (widths <= max_width) & centered & ~blocked
    joined = []
    for start, end in _runs(narrow):
        if joined:
            previous_end = joined[-1][1]
            gap = slice(previous_end, start)
            if (start - previous_end <= 2 and not blocked[gap].any()
                    and np.all(widths[gap] == 0)):
                joined[-1] = (joined[-1][0], end)
                continue
        joined.append((start, end))
    tails = []
    for start, end in joined:
        if end - start < min_rows:
            continue
        samples = widths[start:end]
        samples = samples[samples > 0]
        low, high = np.quantile(samples, [0.2, 0.8])
        if low >= 0.6 * high:
            tails.append((start, end))
    return tails


class NoteDetector:
    """Rectifica la perspectiva y separa cabezas anchas de colas finas.

    ``corners`` debe ser [arriba izquierda, arriba derecha, línea de toque
    derecha, línea de toque izquierda], abarcando los cinco carriles completos.
    Los parámetros de tamaño son proporciones del ancho de un carril o de la
    altura rectificada; se mantienen al cambiar la resolución de captura.

    La banda inferior se excluye para no contar los receptores como notas. El
    seguimiento externo debe proyectar las notas desde esa banda hasta la línea
    de toque. No se infiere que una nota desaparecida haya sido acertada.
    """

    def __init__(
        self,
        width: int = 500,
        height: int = 800,
        # Las notas son de colores vivos; la foto del fondo que se transparenta
        # a través de la pista queda por debajo de este valor.
        saturation_min: int = 130,
        value_min: int = 90,
        min_head_width_ratio: float = 0.27,
        min_head_height_ratio: float = 0.10,
        lane_margin_ratio: float = 0.12,
        # Los aros de los botones y las llamas al acertar viven en esta banda.
        # Excluirla evita notas falsas aunque la línea quede un poco corrida.
        hit_exclusion_ratio: float = 0.12,
        min_tail_length_ratio: float = 0.015,
        max_tail_width_ratio: float = 0.18,
        max_head_height_ratio: float = 0.15,
        active_tail_saturation_min: int = 70,
        active_tail_value_min: int = 55,
    ):
        if width < 50 or height < 50:
            raise ValueError("La pista rectificada debe medir al menos 50 x 50.")
        if not 0 <= saturation_min <= 255 or not 0 <= value_min <= 255:
            raise ValueError("Los umbrales HSV deben estar entre 0 y 255.")
        if not 0 <= active_tail_saturation_min <= 255 or not 0 <= active_tail_value_min <= 255:
            raise ValueError("Los umbrales HSV de cola activa deben estar entre 0 y 255.")
        if not 0 <= lane_margin_ratio < 0.4:
            raise ValueError("lane_margin_ratio debe estar entre 0 y 0.4.")
        if not 0 < min_head_width_ratio < 1 - 2 * lane_margin_ratio:
            raise ValueError("La anchura mínima de cabeza debe caber en el carril.")
        if not 0 < min_head_height_ratio < 0.5:
            raise ValueError("min_head_height_ratio debe estar entre 0 y 0.5 del ancho de carril.")
        if not 0 <= hit_exclusion_ratio < 0.5:
            raise ValueError("hit_exclusion_ratio debe estar entre 0 y 0.5.")
        if not 0 < min_tail_length_ratio < 0.5:
            raise ValueError("min_tail_length_ratio debe estar entre 0 y 0.5.")
        if not 0 < max_tail_width_ratio < min_head_width_ratio:
            raise ValueError("La anchura máxima de cola debe ser menor que la de cabeza.")
        if not 0 < max_head_height_ratio < 0.5:
            raise ValueError("max_head_height_ratio debe estar entre 0 y 0.5.")
        self.width = int(width)
        self.height = int(height)
        self.saturation_min = saturation_min
        self.value_min = value_min
        self.min_head_width_ratio = min_head_width_ratio
        self.min_head_height_ratio = min_head_height_ratio
        self.lane_margin_ratio = lane_margin_ratio
        self.hit_exclusion_ratio = hit_exclusion_ratio
        self.min_tail_length_ratio = min_tail_length_ratio
        self.max_tail_width_ratio = max_tail_width_ratio
        self.max_head_height_ratio = max_head_height_ratio
        self.active_tail_saturation_min = active_tail_saturation_min
        self.active_tail_value_min = active_tail_value_min

    @property
    def detection_bottom(self) -> int:
        """Primera fila excluida por la banda de receptores."""
        return int(self.height * (1 - self.hit_exclusion_ratio))

    def detect(self, frame_bgr: np.ndarray, corners, *, sustain_lanes=()) -> VisionResult:
        """Detecta cabezas y colas en la pista rectificada.

        sustain_lanes contiene carriles (0 a 4) con un sostenido ya confirmado.
        Sólo su máscara de cola admite el color del efecto al mantener la nota;
        las cabezas y la asociación inicial de sostenidos usan el filtro estricto.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("Se necesita una imagen BGR con tres canales.")
        if frame_bgr.dtype != np.uint8 or min(frame_bgr.shape[:2]) < 2:
            raise ValueError("La imagen BGR debe ser uint8 y no estar vacía.")
        points = np.asarray(corners, dtype=np.float32)
        if points.shape != (4, 2) or not np.all(np.isfinite(points)):
            raise ValueError("Se necesitan cuatro esquinas finitas (x, y).")
        if not cv2.isContourConvex(points) or cv2.contourArea(points) < 4:
            raise ValueError("Las esquinas deben formar un cuadrilátero convexo ordenado.")
        if (
            np.any(points[:, 0] < 0)
            or np.any(points[:, 1] < 0)
            or np.any(points[:, 0] > frame_bgr.shape[1] - 1)
            or np.any(points[:, 1] > frame_bgr.shape[0] - 1)
        ):
            raise ValueError("Las esquinas deben estar dentro de la imagen recortada.")

        target = np.float32(
            [[0, 0], [self.width - 1, 0], [self.width - 1, self.height - 1], [0, self.height - 1]]
        )
        matrix = cv2.getPerspectiveTransform(points, target)
        warped = cv2.warpPerspective(frame_bgr, matrix, (self.width, self.height))
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # No se asignan teclas por hue: las notas del poder pueden compartirlo.
        color_mask = cv2.inRange(hsv, (0, self.saturation_min, self.value_min), (179, 255, 255))
        color_mask[self.detection_bottom :] = 0
        sustain_lanes = set(sustain_lanes)
        active_tail_mask = None
        if sustain_lanes:
            # Sólo prolonga colas confirmadas. Este filtro nunca crea cabezas
            # ni decide que una nueva nota tiene sostenido.
            active_tail_mask = cv2.inRange(
                hsv,
                (0, min(self.saturation_min, self.active_tail_saturation_min),
                 min(self.value_min, self.active_tail_value_min)),
                (179, 255, 255),
            )
            active_tail_mask[self.detection_bottom :] = 0
        mask = np.zeros_like(color_mask)
        tail_mask = np.zeros_like(color_mask)
        heads = []
        lane_width = self.width / 5.0
        min_head_pixels = max(3, int(round(lane_width * self.min_head_width_ratio)))
        max_tail_pixels = max(2, int(round(lane_width * self.max_tail_width_ratio)))
        min_tail_rows = max(4, int(round(self.height * self.min_tail_length_ratio)))
        # El borde cian de una estrella puede formar una banda horizontal
        # separada de su núcleo. Su poco grosor evita contar dos cabezas.
        min_head_rows = max(2, int(round(lane_width * self.min_head_height_ratio)))
        max_head_rows = max(min_head_rows, int(round(self.height * self.max_head_height_ratio)))

        for lane in range(5):
            left = int(round(lane_width * (lane + self.lane_margin_ratio)))
            right = int(round(lane_width * (lane + 1 - self.lane_margin_ratio)))
            center = int(round(lane_width * (lane + 0.5)))
            lane_mask = color_mask[:, left:right]
            mask[:, left:right] = lane_mask
            row_widths = np.count_nonzero(lane_mask, axis=1)
            center_radius = max(2, int(round(lane_width * 0.18)))
            central = color_mask[:, center - center_radius : center + center_radius + 1]
            center_present = np.any(central != 0, axis=1)
            tail_radius = max(2, int(round(lane_width * TAIL_CENTER_RATIO)))
            tail_zone = color_mask[:, center - tail_radius : center + tail_radius + 1]
            tail_present = np.any(tail_zone != 0, axis=1)
            wide_rows = (row_widths >= min_head_pixels) & center_present
            # Une pequeños huecos debidos al brillo o al borde de una estrella.
            wide_rows = cv2.morphologyEx(
                wide_rows.astype(np.uint8).reshape(-1, 1),
                cv2.MORPH_CLOSE,
                np.ones((3, 1), np.uint8),
            ).ravel().astype(bool)
            wide_rows[self.detection_bottom :] = False
            bands = [
                (start, end)
                for start, end in _runs(wide_rows)
                if min_head_rows <= end - start <= max_head_rows
                # Una cabeza cortada por el límite inferior no tiene centro fiable.
                and end < self.detection_bottom
            ]

            tail_runs = _tail_runs(row_widths, tail_present, wide_rows,
                                   max_tail_pixels, min_tail_rows)
            for start, end in tail_runs:
                tail_mask[start:end, center - tail_radius : center + tail_radius + 1] = tail_zone[start:end]

            if lane in sustain_lanes:
                # El brillo puede blanquear o ensanchar una cola al mantenerla.
                # Exigimos un tramo largo y centrado para descartar letras cortas
                # y formas del fondo. Las guías grises siguen fuera por saturación.
                relaxed_widths = np.count_nonzero(active_tail_mask[:, left:right], axis=1)
                centered = np.any(active_tail_mask[:, center - tail_radius:center + tail_radius + 1] != 0, axis=1)
                active_width = min(min_head_pixels - 1, max(max_tail_pixels, int(round(lane_width * 0.24))))
                active_runs = _tail_runs(relaxed_widths, centered, wide_rows,
                                         active_width, max(min_tail_rows, int(round(self.height * 0.08))))
                for start, end in active_runs:
                    section = np.s_[start:end, center - tail_radius:center + tail_radius + 1]
                    tail_mask[section] |= active_tail_mask[section]

            for start, end in bands:
                weights = row_widths[start:end].astype(float)
                y = float(np.average(np.arange(start, end), weights=weights))
                # La cola de una nota que baja queda por encima de su cabeza.
                attachment_gap = max(6, int((end - start) * 0.8))
                attached_tails = [
                    (tail_start, tail_end)
                    for tail_start, tail_end in tail_runs
                    if 0 <= start - tail_end <= attachment_gap
                ]
                tail_top = float(attached_tails[-1][0]) if attached_tails else None
                heads.append(Detection(lane, y, float(end - start), bool(attached_tails), tail_top))

        heads.sort(key=lambda note: (note.y, note.lane))
        return VisionResult(warped=warped, mask=mask, heads=heads, tail_mask=tail_mask)
