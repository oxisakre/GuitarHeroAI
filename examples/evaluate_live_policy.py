"""Compara PPO sobre observaciones de un log v2 y canciones simuladas.

No captura pantalla ni envía teclas. El replay usa las observaciones/memoria
del jugador registrado: mide decisiones sobre esos estados, no puntaje real.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guitar_flash import load_model, make_observation
from guitarflash.tracking import update_held_estimate
from main import GuitarHeroEnv


def reconstruct(rows, height=800, latency_ms=30):
    boards, memories = [], []
    held = np.zeros(5, np.int8)
    for index, event in enumerate(rows):
        board = np.zeros((20, 5), np.uint8)
        cell = (height - 1) / 19
        for track in event["detalles_seguimiento"]:
            if not track["moving"]:
                continue
            elapsed = event["observed"] - track["last_seen"] + latency_ms / 1000
            speed = track["velocity"] or event["velocidad_exacta"]
            row = int(np.floor((track["y"] + speed * elapsed) / cell))
            lane = track["carril"] - 1
            if track["tail_top"] is not None:
                first = max(0, int(np.floor((track["tail_top"] + speed * elapsed) / cell)))
                last = min(20, row)
                if first < last:
                    board[first:last, lane] = 2
            if 0 <= row < 20:
                board[row, lane] = 1
        if board[-1].tolist() != event["fila19"]:
            raise ValueError(f"Fila 19 distinta en cuadro {index}; comprobar altura/latencia del log.")
        boards.append(board)
        memories.append(held.copy())
        # v2 no guarda held ni si hubo nueva inferencia: es una aproximación.
        held = update_held_estimate(held, board, event["accion"])
    return np.asarray(boards), np.asarray(memories)


def decisions(actions, boards, memories):
    last = boards[:, -1]
    empty = (actions == 1) & (last == 0)
    multi = np.count_nonzero(last == 2, axis=1) >= 2
    return {
        "empty_requests": int(empty.sum()),
        "empty_requests_during_multi_sustains": int(empty[multi].sum()),
        "head_requests_missed": int(((actions == 0) & (last == 1)).sum()),
        "held_tail_release_requests": int(((actions == 0) & (last == 2) & (memories == 1)).sum()),
    }


def simulate(model, style, episodes, steps, seed):
    env = GuitarHeroEnv(note_style=style, include_hold_state=True, max_steps=steps)
    totals = env._empty_metrics()
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            for _ in range(steps):
                action, _ = model.predict(make_observation(model, obs["board"], obs["held"]),
                                          deterministic=True)
                obs, _, _, _, info = env.step(action)
            for name in totals:
                totals[name] += info[name]
    finally:
        env.close()
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--latency-ms", type=float, default=30)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100042)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.height < 2 or args.episodes < 1 or args.steps < 1:
        parser.error("height >= 2, episodes y steps >= 1")
    import torch
    torch.set_num_threads(1)
    with args.log.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        parser.error("El registro está vacío")
    boards, memories = reconstruct(rows, args.height, args.latency_ms)
    recorded = np.asarray([r["accion"] for r in rows])
    report = {
        "log": str(args.log), "frames": len(rows), "height": args.height,
        "latency_ms": args.latency_ms, "seed": args.seed,
        "episodes_per_style": args.episodes, "steps_per_episode": args.steps,
        "limitations": "Memoria reconstruida desde acciones registradas; solicitudes por cuadro, no pulsaciones ni aciertos del juego. Simulación sin efectos visuales.",
        "recorded": decisions(recorded, boards, memories), "models": {},
    }
    for path in args.models:
        model = load_model(path)
        action, _ = model.predict(make_observation(model, boards, memories), deterministic=True)
        result = {"replay": decisions(action, boards, memories),
                  "matches_recorded_frames": float(np.mean(np.all(action == recorded, axis=1))),
                  "simulation": {style: simulate(model, style, args.episodes, args.steps, args.seed)
                                 for style in ("song", "sustain")}}
        report["models"][path] = result
        print(path, json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
