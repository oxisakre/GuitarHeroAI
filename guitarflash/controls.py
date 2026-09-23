"""Teclado Windows con transiciones explícitas y backend sustituible en tests."""
import ctypes
import math
import sys
import time
from ctypes import wintypes


class KeyController:
    """Traduce las acciones de PPO a keyDown/keyUp.

    min_release (segundos) es el tiempo mínimo que una tecla queda suelta
    antes de volver a apretarla. Si el juego lee el teclado una vez por cuadro,
    un keyUp y un keyDown en el mismo instante pueden pasar desapercibidos.
    Los keyDown que esperan esa pausa se envían con flush().
    """

    def __init__(self, backend, keys="asdfg", min_release=0.0):
        if len(keys) != 5 or len(set(keys.lower())) != 5 or not keys.isascii() or not keys.isalnum():
            raise ValueError("Elegí cinco letras o números distintos, por ejemplo asdfg.")
        if min_release < 0:
            raise ValueError("min_release no puede ser negativo.")
        self.backend, self.keys = backend, keys.lower()
        self.min_release = float(min_release)
        self.pressed = [False] * 5
        self.wanted = [False] * 5
        self.tokens = [None] * 5
        self.released_at = [-math.inf] * 5

    def apply(self, action, tokens=None, now=None):
        if len(action) != 5 or any(value not in (0, 1) for value in action):
            raise ValueError("PPO debe devolver cinco acciones binarias.")
        now = time.perf_counter() if now is None else now
        tokens = tokens or [None] * 5
        for lane, value in enumerate(action):
            # Una nueva cabeza puede requerir otro keyDown incluso cuando el
            # modelo mantiene 1. Es traducción al teclado, no elección de nota.
            retrigger = bool(value and self.pressed[lane] and tokens[lane] is not None
                             and tokens[lane] != self.tokens[lane])
            if self.pressed[lane] and (not value or retrigger):
                self.backend.key_up(self.keys[lane])
                self.pressed[lane] = False
                self.released_at[lane] = now
            self.wanted[lane] = bool(value)
            if tokens[lane] is not None and value:
                self.tokens[lane] = tokens[lane]
        self.flush(now)

    def flush(self, now=None):
        """Aprieta las teclas pedidas cuya pausa mínima ya terminó."""
        now = time.perf_counter() if now is None else now
        for lane in range(5):
            if (self.wanted[lane] and not self.pressed[lane]
                    and now >= self.released_at[lane] + self.min_release):
                self.backend.key_down(self.keys[lane])
                self.pressed[lane] = True

    def next_press_time(self):
        """Momento del próximo keyDown que espera su pausa, o None."""
        pending = [self.released_at[lane] + self.min_release for lane in range(5)
                   if self.wanted[lane] and not self.pressed[lane]]
        return min(pending) if pending else None

    def release_all(self):
        errors = []
        self.wanted = [False] * 5
        for lane, key in enumerate(self.keys):
            if self.pressed[lane]:
                try:
                    self.backend.key_up(key)
                except OSError as exc:
                    errors.append(exc)
                else:
                    self.pressed[lane] = False
        self.tokens = [None] * 5
        if errors:
            raise errors[0]


class WindowsKeyboard:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("El control de teclado implementado requiere Windows.")
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self.user32.GetAsyncKeyState.restype = ctypes.c_short
        self.user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
        self.user32.MapVirtualKeyW.restype = wintypes.UINT
        ulong_ptr = ctypes.c_size_t

        class KeyboardInput(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ulong_ptr)]

        class MouseInput(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

        class Payload(ctypes.Union):
            _fields_ = [("ki", KeyboardInput), ("mi", MouseInput)]

        class Input(ctypes.Structure):
            _anonymous_ = ("payload",)
            _fields_ = [("type", wintypes.DWORD), ("payload", Payload)]

        self.Input = Input
        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT

    def _send(self, key, release):
        entry = self.Input()
        entry.type = 1
        entry.ki.wScan = self.user32.MapVirtualKeyW(ord(key.upper()), 0)
        entry.ki.dwFlags = 0x0008 | (0x0002 if release else 0)
        if self.user32.SendInput(1, ctypes.byref(entry), ctypes.sizeof(entry)) != 1:
            raise OSError("Windows no aceptó la tecla. Revisá el foco y los permisos del juego.")

    def key_down(self, key):
        self._send(key, False)

    def key_up(self, key):
        self._send(key, True)

    def hotkey(self, virtual_key):
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    def foreground(self):
        return self.user32.GetForegroundWindow()
