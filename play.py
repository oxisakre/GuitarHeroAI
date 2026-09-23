"""Visualizar un checkpoint en el simulador, sin controlar el teclado real."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="IA_guitarristaCanciones")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay-ms", type=int, default=50)
    parser.add_argument("--note-style", choices=("song", "dense", "sustain"), default="song",
                        help="song: canciones; dense: original; sustain: canciones y práctica de acordes sostenidos")
    args = parser.parse_args(argv)
    if args.delay_ms < 1:
        parser.error("delay-ms debe ser positivo")

    import cv2
    import numpy as np
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from main import GuitarHeroEnv

    model = PPO.load(args.model)
    env = GuitarHeroEnv(include_hold_state=isinstance(model.observation_space, spaces.Dict),
                        note_style=args.note_style)
    obs, _ = env.reset(seed=args.seed)
    episode = 1
    print("La IA está tocando en el simulador. Presiona Q para salir.")
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            board = obs["board"] if isinstance(obs, dict) else obs
            image = np.zeros((20, 5, 3), dtype=np.uint8)
            image[board == 1] = (255, 255, 255)
            image[board == 2] = (0, 255, 0)
            image = cv2.resize(image, (400, 800), interpolation=cv2.INTER_NEAREST)
            labels = [
                f"Episodio {episode} | Puntos: {info['episode_reward']:.0f}",
                f"Aciertos: {info['notes_hit']} | Perdidas: {info['notes_missed']}",
                f"Precision: {info['accuracy']:.1%} | Errores: {info['wrong_presses']}",
                f"Sostenidos completos: {info['sustains_completed']}",
            ]
            cv2.rectangle(image, (0, 0), (400, 105), (20, 20, 20), -1)
            for index, label in enumerate(labels):
                cv2.putText(image, label, (8, 22 + index * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.imshow("IA Jugando", image)
            if cv2.waitKey(args.delay_ms) & 0xFF == ord("q"):
                break
            if terminated or truncated:
                print(f"Episodio {episode}: {info}")
                obs, _ = env.reset()
                episode += 1
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
