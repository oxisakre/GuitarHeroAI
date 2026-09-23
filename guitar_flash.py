"""Visor/calibración y conexión del modelo PPO a Guitar Flash (Sólo tocar)."""
import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np

from guitarflash.calibration import calibrate, load_config, save_config
from guitarflash.controls import KeyController, WindowsKeyboard
from guitarflash.debug import DebugLog, RecordingKeyboard
from guitarflash.tracking import BoardTracker, update_held_estimate
from guitarflash.vision import NoteDetector


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--calibrate", action="store_true", help="Seleccionar región y cuatro esquinas de la pista.")
    result.add_argument("--config", default="guitarflash_config.json")
    result.add_argument("--monitor", type=int, default=1, help="Monitor donde está el juego (1 = principal).")
    result.add_argument("--calibrate-delay", type=int, default=5, help="Segundos para mostrar el juego antes de capturar al calibrar.")
    result.add_argument("--image", type=Path, help="Analizar una imagen local; no captura pantalla ni envía teclas.")
    result.add_argument("--output", type=Path, default=Path("artifacts/deteccion.png"))
    result.add_argument("--model", type=Path, help="Modelo PPO .zip; sin esta opción solo muestra detecciones.")
    result.add_argument("--control", action="store_true", help="Habilitar control del teclado (se activa con F8).")
    result.add_argument("--keys", help="Cinco letras/números en orden verde,rojo,amarillo,azul,naranja; defecto asdfg.")
    result.add_argument("--fps", type=float, default=60, help="Frecuencia base de captura; aumenta para notas rápidas.")
    result.add_argument("--decision-hz", type=float, default=60, help="Consultas a PPO por segundo; con 60 consulta en cada captura.")
    result.add_argument("--latency-ms", type=float, default=30, help="Anticipación para compensar captura y teclado; subila si toca tarde, bajala si toca antes.")
    result.add_argument("--min-release-ms", type=float, default=25, help="Tiempo mínimo con una tecla suelta antes de volver a apretarla.")
    result.add_argument("--initial-speed", type=float, default=400, help="Velocidad inicial estimada en píxeles rectificados/segundo.")
    result.add_argument("--debug", action="store_true", help="Con --control: guarda en debug/ las teclas enviadas y una imagen cada vez que suelta y vuelve a apretar rápido.")
    return result


# Con decision-hz igual a fps, el jitter de captura no debe saltear consultas.
DECISION_TOLERANCE = 0.75
# En la partida de 99% la velocidad variaba x1.0 entre arriba y abajo; en la que
# perdía enseguida, x2.4. Por encima de esto conviene recalibrar.
CALIBRACION_TOLERADA = 1.3


def make_observation(model, board, held):
    if hasattr(model.observation_space, "spaces"):
        return {"board": board, "held": held.copy()}
    return board


def load_model(path):
    from gymnasium import spaces
    from stable_baselines3 import PPO
    model = PPO.load(str(path), device="cpu")
    space = model.observation_space
    if isinstance(space, spaces.Dict):
        if (set(space.spaces) != {"board", "held"}
                or not isinstance(space["held"], spaces.MultiBinary)
                or space["held"].shape != (5,)):
            raise ValueError("El modelo requiere una observación diferente de board/held.")
        space = space["board"]
    if (not isinstance(space, spaces.Box) or space.shape != (20, 5)
            or space.dtype != np.uint8 or not np.all(space.low == 0) or not np.all(space.high == 2)
            or not isinstance(model.action_space, spaces.MultiBinary)
            or model.action_space.shape != (5,)):
        raise ValueError("Se necesita un modelo con tablero 20x5 y cinco acciones binarias.")
    return model


def procesar_captura(detector, tracker, frame, corners, captured):
    """Analiza una captura y actualiza el seguimiento.

    También se procesan las capturas idénticas a la anterior: durante un sostenido
    largo la pantalla no cambia y la cola igual tiene que irse corriendo. De que una
    imagen repetida no invente notas ni velocidades se encarga el seguimiento,
    carril por carril.
    """
    sustain_lanes = {track.lane for track in tracker.tracks
                     if track.moving and track.tail_top is not None}
    result = detector.detect(frame, corners, sustain_lanes=sustain_lanes)
    tracker.update(result.heads, result.tail_mask, captured)
    return result


def countdown(seconds, sleep=time.sleep):
    """Da tiempo a poner el juego a la vista: si no, se captura la consola encima."""
    for remaining in range(seconds, 0, -1):
        print(f"  Captura en {remaining}...", flush=True)
        sleep(1)


def send_action(controller, tracker, action, now):
    """Envía la acción de PPO mirando min_release hacia adelante.

    Si otra cabeza llega a la fila de toque de un carril con la tecla apretada
    dentro de ese margen, la tecla se suelta ya: así puede volver a apretarse
    justo cuando la cabeza llega, sin quedar pegada al keyUp.
    """
    _, upcoming = tracker.observation(now + controller.min_release)
    controller.apply(action, upcoming, now)


def wait_until(deadline, controller=None, backend=None, target=None):
    """Duerme hasta deadline, enviando a tiempo los keyDown que esperan su pausa."""
    while True:
        now = time.perf_counter()
        pending = None
        if controller is not None and backend.foreground() == target:
            controller.flush(now)
            pending = controller.next_press_time()
        if now >= deadline:
            return
        wake = deadline if pending is None or pending <= now else min(deadline, pending)
        time.sleep(wake - now)


def preview(result, board, status, action=None, speed=0, calibracion=None):
    view = result.warped.copy()
    height, width = view.shape[:2]
    tail_pixels = result.tail_mask > 0
    view[tail_pixels] = (view[tail_pixels].astype(np.float32)*0.3 + np.array([255,80,255])*0.7).astype(np.uint8)
    for lane in range(1, 5):
        cv2.line(view, (lane*width//5, 0), (lane*width//5, height-1), (90,90,90), 1)
    for head in result.heads:
        x = int((head.lane+0.5)*width/5)
        cv2.circle(view, (x, int(head.y)), 15, (255,255,255), 2)
        cv2.putText(view, str(head.lane+1), (x-5, int(head.y)-18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cells = np.zeros((20, 5, 3), np.uint8)
    cells[board == 1] = (255,255,255)
    cells[board == 2] = (0,180,80)
    matrix = cv2.resize(cells, (200,height), interpolation=cv2.INTER_NEAREST)
    combined = np.hstack([view, matrix])
    combined = cv2.copyMakeBorder(combined, 80, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(combined, status, (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,255,255), 1, cv2.LINE_AA)
    cv2.putText(combined, "F8 activa/pausa | F9 sale | Q sale desde el visor", (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    teclas = "-----" if action is None else "".join("X" if value else "." for value in action)
    text = f"Velocidad: {speed:.0f} px/s | PPO: {teclas}"
    cv2.putText(combined, text, (10,71), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)
    # Las notas bajan a velocidad pareja: si arriba y abajo difieren, las esquinas
    # no siguen la perspectiva de la pista y el momento de apretar sale mal.
    if calibracion is not None:
        torcida = calibracion > CALIBRACION_TOLERADA
        aviso = f"CALIBRACION TORCIDA x{calibracion:.1f}: recalibra" if torcida else "Calibracion OK"
        cv2.putText(combined, aviso, (combined.shape[1] - 330, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 255) if torcida else (0, 200, 0), 1, cv2.LINE_AA)
    return combined


def analyze_image(args):
    if args.control:
        raise ValueError("--image no permite --control.")
    if args.model:
        raise ValueError("--image comprueba visión solamente; usá --model sin --image para el seguimiento temporal.")
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f"No se pudo leer {args.image}.")
    if args.calibrate:
        config = calibrate(frame, keys=args.keys or "asdfg")
        config["image_only"] = True
        save_config(args.config, config)
    else:
        config = load_config(args.config)
    region = config["region"]
    x,y,w,h = (region[k] for k in ("left","top","width","height"))
    if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
        raise ValueError("La región calibrada no cabe en esta imagen. Recalibrá con --image --calibrate.")
    detector = NoteDetector(**config.get("detector", {}))
    result = detector.detect(frame[y:y+h,x:x+w], config["corners"])
    # En una foto no podemos confirmar movimiento ni estimar tiempos.
    board = np.zeros((20,5), np.uint8)
    for row in range(20):
        top, bottom = row*result.warped.shape[0]//20, (row+1)*result.warped.shape[0]//20
        for lane in range(5):
            left, right = lane*result.warped.shape[1]//5, (lane+1)*result.warped.shape[1]//5
            if np.count_nonzero(result.tail_mask[top:bottom,left:right]) >= 3:
                board[row,lane] = 2
    for head in result.heads:
        row = min(19, int(head.y / result.warped.shape[0] * 20))
        board[row, head.lane] = 1
    output = preview(result, board, "IMAGEN ESTATICA: detecciones sin seguimiento")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise OSError("No se pudo guardar la vista de detecciones.")
    print(f"Vista: {args.output.resolve()}")
    print(f"Cabezas candidatas: {len(result.heads)} (requieren validación visual)")
    print(f"Píxeles de cola: {np.count_nonzero(result.tail_mask)} (marcados en magenta)")
    for head in result.heads:
        print(f"  carril={head.lane+1}, y={head.y:.1f}, sostenido={head.has_tail}")


def run_live(args):
    import mss
    if args.control and args.model is None:
        raise ValueError("--control requiere --model: PPO debe decidir las acciones.")
    model = load_model(args.model) if args.model else None
    # mss 10.2 marca mss.mss() como obsoleto; las 10.x anteriores sólo tienen ese nombre.
    screen_capture = mss.MSS if hasattr(mss, "MSS") else mss.mss
    with screen_capture() as screen:
        if args.calibrate:
            if not 1 <= args.monitor < len(screen.monitors):
                raise ValueError("Número de monitor inválido.")
            monitor = screen.monitors[args.monitor]
            print(f"Monitor {args.monitor} ({monitor['width']}x{monitor['height']}): dejá Guitar Flash visible, "
                  "sin la consola encima. Ctrl+C cancela.")
            countdown(args.calibrate_delay)
            frame = np.asarray(screen.grab(monitor))[:,:,:3].copy()
            config = calibrate(frame, offset=(monitor["left"],monitor["top"]), keys=args.keys or "asdfg")
            save_config(args.config, config)
            print(f"Calibración guardada en {args.config}. Volvé a ejecutar sin --calibrate para probarla.")
            return
        config = load_config(args.config)
        if config.get("image_only"):
            raise ValueError("Esta calibración es de una imagen. Ejecutá --calibrate en la pantalla real.")
        detector = NoteDetector(**config.get("detector", {}))
        tracker = BoardTracker(height=detector.height, width=detector.width,
                               initial_speed=args.initial_speed, latency_ms=args.latency_ms)
        backend = WindowsKeyboard() if args.control else None
        recorder = RecordingKeyboard(backend) if backend and args.debug else None
        debug = DebugLog(Path("debug") / datetime.now().strftime("%Y%m%d_%H%M%S")) if recorder else None
        controller = (KeyController(recorder or backend, args.keys or config.get("keys", "asdfg"),
                                    min_release=args.min_release_ms / 1000) if backend else None)
        held = np.zeros(5, np.int8)
        action = np.zeros(5, np.int8)
        active, target, old_f8 = False, None, False
        last_decision = -float("inf")
        last_frame = None
        name = "Guitar Flash - deteccion y PPO"
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        # A tamaño real: achicarla vuelve ilegibles los textos y las detecciones.
        cv2.resizeWindow(name, detector.width + 200, detector.height + 80)
        print("Colocá el visor fuera de la pista. El juego debe estar en modo Sólo tocar.")
        print("PPO decide; este programa no entrena ni lee el puntaje del juego.")
        if controller:
            print("Enfocá Guitar Flash y pulsá F8 para activar/pausar. F9 sale. Cambiar de ventana pausa.")
        try:
            while True:
                start = time.perf_counter()
                if backend:
                    if backend.hotkey(0x78):  # F9, incluso con navegador enfocado
                        break
                    f8 = backend.hotkey(0x77)
                    if f8 and not old_f8:
                        active = not active
                        controller.release_all()
                        tracker.reset()
                        held[:] = 0
                        target = backend.foreground() if active else None
                    old_f8 = f8
                    if active and backend.foreground() != target:
                        active = False
                        controller.release_all()
                        held[:] = 0
                        tracker.reset()
                    if active and last_frame is not None and start-last_frame > 0.3:
                        active = False
                        controller.release_all()
                        held[:] = 0
                        tracker.reset()
                        print("Pausa por demora de captura. F8 para reactivar.")
                # La imagen corresponde al inicio de la captura, no a su final.
                captured = time.perf_counter()
                frame = np.asarray(screen.grab(config["region"]))[:,:,:3].copy()
                result = procesar_captura(detector, tracker, frame, config["corners"], captured)
                now = time.perf_counter()
                board, _ = tracker.observation(now)
                cell_time = (detector.height-1)/19 / max(tracker.speed, 1)
                decision_interval = min(1/args.decision_hz, cell_time/2)
                if model is not None and now-last_decision >= DECISION_TOLERANCE*decision_interval:
                    action, _ = model.predict(make_observation(model, board, held), deterministic=True)
                    action = np.asarray(action, dtype=np.int8)
                    last_decision = now
                    if controller and active:
                        # Volver a comprobar foco y antigüedad después de inferir.
                        if backend.foreground() == target and time.perf_counter()-captured < 0.25:
                            send_action(controller, tracker, action, time.perf_counter())
                            held = update_held_estimate(held, board, action)
                        else:
                            active = False
                            controller.release_all()
                            held[:] = 0
                    elif not controller:
                        held = update_held_estimate(held, board, action)
                status = "TECLADO ACTIVO" if active else ("PAUSADO - F8 desde el juego" if controller else "VISTA PREVIA - sin enviar teclas")
                view = preview(result, board, status, action if model else None, tracker.speed,
                               tracker.calibracion())
                cv2.imshow(name, view)
                if debug and active:
                    debug.record(now, action, board, result.heads, tracker, recorder.take_sent(),
                                 view, result.warped, captured=captured,
                                 tail_mask=result.tail_mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (27,ord("q")) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                last_frame = start
                wait_until(start + min(1/args.fps, decision_interval),
                           controller if active else None, backend, target)
        finally:
            try:
                if controller:
                    controller.release_all()
            finally:
                cv2.destroyAllWindows()
                if debug:
                    debug.close()
                    print(f"Registro de depuración: {debug.folder.resolve()}")


def main():
    args = parser().parse_args()
    if args.fps <= 0 or args.decision_hz <= 0 or args.initial_speed <= 0:
        raise SystemExit("--fps, --decision-hz y --initial-speed deben ser positivos.")
    if args.min_release_ms < 0 or args.calibrate_delay < 0:
        raise SystemExit("--min-release-ms y --calibrate-delay no pueden ser negativos.")
    try:
        if args.image:
            analyze_image(args)
        else:
            run_live(args)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        print("Detenido.")


if __name__ == "__main__":
    main()
