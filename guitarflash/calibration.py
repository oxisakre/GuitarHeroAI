"""Calibración local de región y perspectiva; se guarda en JSON."""
import ctypes
import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np

ENTER_KEYS = (10, 13)
ESCAPE_KEY = 27
# Una captura de todo el monitor mostrada a tamaño real tapa la pantalla y
# parece un escritorio congelado: se muestra reducida, con instrucciones.
REGION_SCREEN_FRACTION = 0.65
CORNER_SCREEN_FRACTION = 0.85
BAND_HEIGHT = 84
MIN_VIEW_WIDTH = 560
CORNER_LABELS = ("arriba izquierda", "arriba derecha",
                 "linea de toque derecha", "linea de toque izquierda")


def validate_config(config):
    region = config["region"]
    for key in ("left", "top", "width", "height"):
        if not isinstance(region.get(key), int):
            raise ValueError(f"region.{key} debe ser un entero.")
    if region["width"] < 20 or region["height"] < 20:
        raise ValueError("La región es demasiado pequeña.")
    corners = np.asarray(config["corners"], dtype=np.float32)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Se necesitan cuatro puntos (x,y) finitos.")
    if (corners < 0).any() or (corners[:, 0] >= region["width"]).any() or (corners[:, 1] >= region["height"]).any():
        raise ValueError("Los puntos deben estar dentro de la región capturada.")
    if not cv2.isContourConvex(corners) or abs(cv2.contourArea(corners)) < 100:
        raise ValueError("Los puntos deben formar un trapecio sin cruces.")
    if corners[0, 0] >= corners[1, 0] or corners[3, 0] >= corners[2, 0]:
        raise ValueError("Orden requerido: arriba izquierda, arriba derecha, abajo derecha, abajo izquierda.")
    if max(corners[:2, 1]) >= min(corners[2:, 1]):
        raise ValueError("La línea de toque debe quedar debajo de los puntos superiores.")
    return config


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))


def screen_size():
    """Resolución de la pantalla principal, en las mismas unidades que las ventanas."""
    try:
        return ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    except (AttributeError, OSError):  # Fuera de Windows.
        return 1366, 768


def view_limit(screen, fraction):
    """Tamaño máximo de la imagen para que la ventana, con su franja de texto, entre en pantalla."""
    return int(screen[0] * fraction), int(screen[1] * fraction) - BAND_HEIGHT


def fit_scale(width, height, max_size, max_scale=1.0):
    """Escala para que la imagen quepa en max_size sin deformarse."""
    return min(max_scale, max_size[0] / width, max_size[1] / height)


def _ascii(text):
    # Las fuentes de OpenCV no tienen tildes ni eñes.
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


class _CalibrationWindow:
    """Ventana con instrucciones arriba. ESC, la X de la ventana o Ctrl+C cancelan.

    A diferencia de cv2.selectROI, el bucle corre en Python: cerrar la ventana
    no la vuelve a abrir y Ctrl+C en la consola interrumpe el programa.
    """

    def __init__(self, gui, name, image, max_size, max_scale=1.0):
        self.gui, self.name = gui, name
        self.shape = image.shape[:2]
        self.scale = fit_scale(image.shape[1], image.shape[0], max_size, max_scale)
        interpolation = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
        self.view = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=interpolation)
        gui.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        try:
            # Encima de la consola; si queda detrás parece que el programa se colgó.
            gui.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def to_image(self, x, y):
        """Clic en la ventana -> (píxel de la imagen original, si cayó sobre la imagen)."""
        y -= BAND_HEIGHT
        inside = 0 <= x < self.view.shape[1] and 0 <= y < self.view.shape[0]
        height, width = self.shape
        point = (min(max(round(x / self.scale), 0), width - 1),
                 min(max(round(y / self.scale), 0), height - 1))
        return point, inside

    def to_view(self, point):
        return round(point[0] * self.scale), round(point[1] * self.scale)

    def show(self, canvas, lines):
        """Muestra el lienzo y devuelve la tecla; lanza ValueError si se cancela."""
        width = max(canvas.shape[1], MIN_VIEW_WIDTH)
        frame = np.full((BAND_HEIGHT + canvas.shape[0], width, 3), 40, np.uint8)
        frame[BAND_HEIGHT:, :canvas.shape[1]] = canvas
        for index, line in enumerate(lines[:3]):
            cv2.putText(frame, _ascii(line), (10, 24 + 26 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
        self.gui.imshow(self.name, frame)
        key = self.gui.waitKey(20) & 0xFF
        if key == ESCAPE_KEY or self.gui.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
            raise ValueError("Calibración cancelada.")
        return key

    def close(self):
        try:
            self.gui.destroyWindow(self.name)
        except cv2.error:
            pass  # El usuario ya la cerró con la X.


def select_region(image, gui=cv2, screen=None):
    """Paso 1: rectángulo alrededor de la pista. Devuelve (x, y, ancho, alto)."""
    limit = view_limit(screen or screen_size(), REGION_SCREEN_FRACTION)
    window = _CalibrationWindow(gui, "Calibracion 1/2 - pista", image, limit)
    drag = {"start": None, "end": None, "active": False}

    def on_mouse(event, x, y, flags, userdata):
        point, inside = window.to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            drag.update(start=point, end=point, active=True)
        elif event == cv2.EVENT_MOUSEMOVE and drag["active"]:
            drag["end"] = point
        elif event == cv2.EVENT_LBUTTONUP and drag["active"]:
            drag.update(end=point, active=False)

    gui.setMouseCallback(window.name, on_mouse)
    try:
        while True:
            canvas = window.view.copy()
            ready = False
            if drag["start"] is not None:
                (x0, y0), (x1, y1) = drag["start"], drag["end"]
                left, top, right, bottom = min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1
                ready = not drag["active"] and right - left >= 20 and bottom - top >= 20
                cv2.rectangle(canvas, window.to_view((left, top)), window.to_view((right, bottom)),
                              (0, 255, 255), 2)
            key = window.show(canvas, [
                "PASO 1/2: arrastra un rectangulo alrededor de la pista",
                "ENTER confirma" if ready else "Mantene el clic apretado y arrastra",
                "R reinicia | ESC o cerrar la ventana cancela",
            ])
            if key in (ord("r"), ord("R")):
                drag.update(start=None, end=None, active=False)
            if key in ENTER_KEYS and ready:
                return left, top, right - left, bottom - top
    finally:
        window.close()


def select_corners(crop, check=None, gui=cv2, screen=None):
    """Paso 2: cuatro esquinas en orden; check(points) puede rechazar con ValueError."""
    limit = view_limit(screen or screen_size(), CORNER_SCREEN_FRACTION)
    # Una pista chica se agranda hasta 2x para marcar las esquinas con precisión.
    window = _CalibrationWindow(gui, "Calibracion 2/2 - esquinas", crop, limit, max_scale=2.0)
    points = []
    message = ""

    def on_mouse(event, x, y, flags, userdata):
        point, inside = window.to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN and inside and len(points) < 4:
            points.append(list(point))

    gui.setMouseCallback(window.name, on_mouse)
    try:
        while True:
            canvas = window.view.copy()
            for index, point in enumerate(points):
                center = window.to_view(point)
                cv2.circle(canvas, center, 5, (0, 255, 255), -1)
                cv2.putText(canvas, str(index + 1), (center[0] + 7, center[1] - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if len(points) > 1:
                outline = np.int32([window.to_view(point) for point in points])
                cv2.polylines(canvas, [outline], len(points) == 4, (0, 255, 255), 1)
            step = (f"Clic {len(points) + 1}/4: {CORNER_LABELS[len(points)]}"
                    if len(points) < 4 else "ENTER guarda")
            key = window.show(canvas, [
                "PASO 2/2: marca las 4 esquinas de la pista",
                message or step,
                "R reinicia | ESC o cerrar la ventana cancela",
            ])
            if key in (ord("r"), ord("R")):
                points.clear()
                message = ""
            if key in ENTER_KEYS and len(points) == 4:
                try:
                    if check is not None:
                        check(points)
                except ValueError as exc:
                    message = f"{exc} (R reinicia)"
                    print(f"Esquinas rechazadas: {exc} Apretá R y marcalas de nuevo.")
                else:
                    return points
    finally:
        window.close()


def calibrate(image, offset=(0, 0), keys="asdfg", gui=cv2, screen=None):
    screen = screen or screen_size()
    print("Paso 1/2: en la ventana 'Calibracion 1/2' arrastrá un rectángulo alrededor de toda la pista")
    print("          y apretá ENTER. ESC, cerrar la ventana o Ctrl+C en esta consola cancelan.")
    x, y, width, height = select_region(image, gui, screen)
    region = {"left": x + offset[0], "top": y + offset[1], "width": width, "height": height}

    def config_for(corners):
        return {"version": 1, "region": region, "corners": [list(point) for point in corners],
                "keys": keys, "detector": {}}

    print("Paso 2/2: hacé clic en arriba izquierda, arriba derecha, línea de toque derecha y")
    print("          línea de toque izquierda (a la altura del centro de los botones). ENTER guarda.")
    crop = image[y:y + height, x:x + width].copy()
    corners = select_corners(crop, check=lambda points: validate_config(config_for(points)),
                             gui=gui, screen=screen)
    return validate_config(config_for(corners))


def save_config(path, config):
    validate_config(config)
    Path(path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
