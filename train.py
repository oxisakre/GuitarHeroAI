"""Entrenamiento explícito: importar este módulo nunca empieza a entrenar."""

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

# Cada pulsación se premia o castiga en el mismo paso, así que no hace falta
# planificar lejos. Con el 0.99 de PPO el retorno suma ~100 pasos de notas
# aleatorias y el −1 por tocar al aire queda tapado por ese ruido: en canciones,
# 0.5 llegó al 98% del jugador perfecto en 80.000 pasos y 0.99 al 32%.
NEW_MODEL_GAMMA = 0.5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Entrenar PPO en el simulador")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Checkpoint .zip para continuar")
    parser.add_argument("--output", type=Path,
                        help="Nombre del modelo, p. ej. IA_nueva (crea IA_nueva.zip, IA_nueva_mejor.zip "
                             "e IA_nueva_datos). Por defecto models/guitar_hero_<fecha y hora>")
    parser.add_argument("--legacy-observation", action="store_true",
                        help="Modelo nuevo con matriz sola, compatible con el original")
    parser.add_argument("--note-style", choices=("song", "dense", "sustain"), default="song",
                        help="song: canciones; dense: original; sustain: canciones y práctica de acordes sostenidos")
    parser.add_argument("--gamma", type=float,
                        help=f"Cuánto cuentan las recompensas futuras (modelos nuevos: {NEW_MODEL_GAMMA}; "
                             "al reanudar se conserva el del modelo si no se indica)")
    parser.add_argument("--learning-rate", type=float,
                        help="Tamaño del ajuste; al reanudar se conserva el original si no se indica")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("ppo_guitar_hero_logs"),
                        help="Carpeta de curvas para TensorBoard")
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.eval_freq < 1 or args.eval_episodes < 1:
        parser.error("steps, eval-freq y eval-episodes deben ser positivos")
    if args.gamma is not None and not 0 <= args.gamma <= 1:
        parser.error("gamma debe estar entre 0 y 1")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate debe ser finito y positivo")
    return args


def main(argv=None):
    args = parse_args(argv)
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from main import GuitarHeroEnv

    class SameSongsEvalCallback(EvalCallback):
        """Re-siembra el entorno antes de cada evaluación.

        Las notas no dependen de las acciones: así todas las evaluaciones tocan
        las mismas canciones y el mejor modelo no se elige por azar de densidad.
        """

        def __init__(self, *callback_args, eval_seed, **callback_kwargs):
            super().__init__(*callback_args, **callback_kwargs)
            self.eval_seed = eval_seed

        def _on_step(self):
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_env.seed(self.eval_seed)
            return super()._on_step()

    output = args.output or Path("models") / datetime.now().strftime("guitar_hero_%Y%m%d_%H%M%S.zip")
    if output.suffix.lower() != ".zip":
        output = Path(str(output) + ".zip")
    # Nombres fijos a partir del modelo: no hay que buscar carpetas con fecha.
    best_copy = output.with_name(f"{output.stem}_mejor.zip")
    run_dir = output.with_name(f"{output.stem}_datos")
    for path in (output, best_copy, run_dir):
        if path.exists():
            raise FileExistsError(f"Ya existe {path}; elegí otro nombre con --output.")
    run_dir.mkdir(parents=True)

    # Un checkpoint antiguo guarda su propia ruta de TensorBoard: se reemplaza.
    resume_overrides = {} if args.gamma is None else {"gamma": args.gamma}
    if args.learning_rate is not None:
        resume_overrides["learning_rate"] = args.learning_rate
    model = (PPO.load(str(args.resume), device="auto", tensorboard_log=str(args.tensorboard_log),
                      **resume_overrides)
             if args.resume else None)
    include_hold_state = (isinstance(model.observation_space, spaces.Dict)
                          if model is not None else not args.legacy_observation)
    if model is not None and args.legacy_observation and include_hold_state:
        raise ValueError("El checkpoint usa board/held; no admite --legacy-observation")
    raw_env = GuitarHeroEnv(include_hold_state=include_hold_state, note_style=args.note_style)
    check_env(raw_env, warn=True)
    env = Monitor(raw_env, filename=str(run_dir / "train"))
    eval_env = DummyVecEnv([lambda: Monitor(
        GuitarHeroEnv(include_hold_state=include_hold_state, note_style=args.note_style))])
    try:
        if model is None:
            policy = "MultiInputPolicy" if include_hold_state else "MlpPolicy"
            model = PPO(policy, env, seed=args.seed, verbose=1,
                        gamma=NEW_MODEL_GAMMA if args.gamma is None else args.gamma,
                        learning_rate=3e-4 if args.learning_rate is None else args.learning_rate,
                        tensorboard_log=str(args.tensorboard_log))
        else:
            model.set_env(env)
            model.set_random_seed(args.seed)

        callbacks = [
            CheckpointCallback(save_freq=args.eval_freq,
                               save_path=str(run_dir / "checkpoints"),
                               name_prefix="guitar_hero"),
            SameSongsEvalCallback(eval_env, eval_seed=args.seed + 10_000,
                                  best_model_save_path=str(run_dir / "best"),
                                  log_path=str(run_dir / "eval"), eval_freq=args.eval_freq,
                                  n_eval_episodes=args.eval_episodes,
                                  deterministic=True, render=False),
        ]
        print(f"Entrenando {args.steps:,} pasos con notas '{args.note_style}' y gamma {model.gamma}. "
              f"Ctrl+C lo corta y guarda lo aprendido hasta ese momento.")
        interrupted = False
        try:
            model.learn(total_timesteps=args.steps, callback=callbacks,
                        reset_num_timesteps=args.resume is None, tb_log_name=output.stem)
        except KeyboardInterrupt:
            interrupted = True
        model.save(str(output))
        best = run_dir / "best" / "best_model.zip"
        if best.exists():
            shutil.copyfile(best, best_copy)
        recommended = (best_copy if best_copy.exists() else output).as_posix()
        print(f"\n{'Entrenamiento cortado' if interrupted else 'Entrenamiento terminado'}.")
        print(f"  Último paso:     {output.as_posix()}")
        if best_copy.exists():
            print(f"  Mejor evaluado:  {best_copy.as_posix()}  <- el recomendado")
        print(f"  Checkpoints:     {(run_dir / 'checkpoints').as_posix()}")
        print(f"Verlo en el simulador:\n  ./.venv/Scripts/python.exe play.py --model {recommended}")
        print(f"Probarlo en Guitar Flash:\n  ./.venv/Scripts/python.exe guitar_flash.py --model {recommended} --control")
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
