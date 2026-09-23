"""Simulador de cinco carriles compatible con los modelos originales."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

NOTE_STYLES = ("song", "dense", "sustain")


class GuitarHeroEnv(gym.Env):
    """Cada acción indica qué teclas mantener durante un paso.

    Por defecto conserva la matriz (20, 5) de los checkpoints originales.
    include_hold_state=True añade la memoria de notas válidas sostenidas;
    requiere un modelo nuevo con MultiInputPolicy. No exige un flanco de
    pulsación para las cabezas: esa regla debe comprobarse en el juego real.

    note_style="song" imita una canción: cada episodio sortea silencios,
    tramos más o menos densos, acordes y sostenidos, para que el modelo
    también aprenda a no tocar cuando la pista está vacía.
    note_style="dense" es el generador original con el que se entrenó
    IA_guitarristaMultiple: note_probability de nota nueva por carril y
    sustain_probability de que la cola continúe. Casi nunca deja la pista vacía.
    note_style="sustain" mezcla canciones normales con episodios de práctica
    de acordes sostenidos largos. Conserva silencios, cabezas y recompensas.
    """

    def __init__(self, include_hold_state=False, max_steps=2000, note_style="song",
                 note_probability=None, sustain_probability=None):
        super().__init__()
        if max_steps < 1:
            raise ValueError("max_steps debe ser positivo")
        if note_style not in NOTE_STYLES:
            raise ValueError(f"note_style debe ser uno de {NOTE_STYLES}")
        if note_style != "dense":
            if note_probability is not None or sustain_probability is not None:
                raise ValueError("note_probability y sustain_probability solo se usan con note_style='dense'")
        else:
            note_probability = 0.1 if note_probability is None else note_probability
            sustain_probability = 0.9 if sustain_probability is None else sustain_probability
            if not 0 <= note_probability <= 1 or not 0 <= sustain_probability <= 1:
                raise ValueError("Las probabilidades deben estar entre 0 y 1")
        self.include_hold_state = include_hold_state
        self.max_steps = max_steps
        self.note_style = note_style
        self.note_probability = note_probability
        self.sustain_probability = sustain_probability
        self._reset_song()
        self.action_space = spaces.MultiBinary(5)
        board_space = spaces.Box(low=0, high=2, shape=(20, 5), dtype=np.uint8)
        self.observation_space = (
            spaces.Dict({"board": board_space, "held": spaces.MultiBinary(5)})
            if include_hold_state else board_space
        )
        self.state = np.zeros((20, 5), dtype=np.uint8)
        self.sosteniendo_nota = [False] * 5
        self.initial_steps = 0
        self.episode_reward = 0.0
        self.metrics = self._empty_metrics()

    @staticmethod
    def _empty_metrics():
        return dict(notes_hit=0, notes_missed=0, wrong_presses=0,
                    sustain_ticks_hit=0, sustain_ticks_missed=0,
                    sustains_completed=0, sustains_broken=0)

    def _observation(self):
        board = self.state.copy()
        if self.include_hold_state:
            return {"board": board,
                    "held": np.asarray(self.sosteniendo_nota, dtype=np.int8)}
        return board

    def _info(self):
        total = self.metrics["notes_hit"] + self.metrics["notes_missed"]
        return {**self.metrics,
                "accuracy": self.metrics["notes_hit"] / total if total else 0.0,
                "steps": self.initial_steps, "episode_reward": self.episode_reward}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros((20, 5), dtype=np.uint8)
        self.sosteniendo_nota = [False] * 5
        self.initial_steps = 0
        self.episode_reward = 0.0
        self.metrics = self._empty_metrics()
        self._reset_song(self.np_random)
        return self._observation(), self._info()

    def _reset_song(self, rng=None):
        """Sortea la "canción" del episodio; sin rng deja valores neutros."""
        self._tail_left = [0] * 5
        self._playing = False
        self._density = 0.0
        # Empieza con una intro en silencio de hasta 40 filas.
        self._section_left = int(rng.integers(0, 41)) if rng is not None else 0
        self._sustain_chance = rng.uniform(0.05, 0.5) if rng is not None else 0.0
        # La mitad de los episodios conserva el repertorio habitual. El resto
        # expone combinaciones simultáneas y colas que ocupan toda la pantalla,
        # poco frecuentes con colas independientes de sólo 2-12 filas.
        self._practice_episode = (self.note_style == "sustain" and rng is not None
                                  and rng.random() < 0.5)
        if self._practice_episode:
            self._sustain_chance = rng.uniform(0.7, 1.0)

    def _generate_dense_row(self):
        for col in range(5):
            if self.state[1, col] > 0:
                if self.np_random.random() < self.sustain_probability:
                    self.state[0, col] = 2
            elif self.np_random.random() < self.note_probability:
                self.state[0, col] = 1

    def _generate_song_row(self):
        rng = self.np_random
        row = self.state[0]
        # Las colas en curso siguen por encima de su cabeza, incluso en silencio.
        for col in range(5):
            if self._tail_left[col] > 0:
                row[col] = 2
                self._tail_left[col] -= 1
        if self._section_left <= 0:
            # Alterna tramos tocando (con su propia densidad) y silencios.
            self._playing = not self._playing
            if self._playing:
                self._section_left = int(rng.integers(30, 251))
                self._density = rng.uniform(0.05, 0.5)
            else:
                self._section_left = int(rng.integers(5, 61))
        self._section_left -= 1
        if not self._playing or rng.random() >= self._density:
            return
        free = [col for col in range(5) if row[col] == 0]
        size = min(len(free), int(rng.choice([1, 2, 3, 4], p=[0.1, 0.5, 0.3, 0.1])
                                 if self._practice_episode else
                                 rng.choice([1, 2, 3], p=[0.7, 0.22, 0.08])))
        if size == 0:
            return
        chord_length = int(rng.integers(4, 41)) if self._practice_episode else None
        for col in rng.choice(free, size=size, replace=False):
            row[col] = 1
            if rng.random() < self._sustain_chance:
                self._tail_left[col] = (chord_length if chord_length is not None
                                        else int(rng.integers(2, 13)))

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (5,) or not np.all((action == 0) | (action == 1)):
            raise ValueError("La acción debe contener cinco valores 0 o 1")
        reward = 0.0
        for col in range(5):
            note = self.state[-1, col]
            pressed = bool(action[col])
            if note == 1:
                if pressed:
                    reward += 6
                    self.metrics["notes_hit"] += 1
                else:
                    reward -= 5
                    self.metrics["notes_missed"] += 1
                self.sosteniendo_nota[col] = pressed
            elif note == 2:
                valid_hold = pressed and self.sosteniendo_nota[col]
                if valid_hold:
                    # Un sostenido acertado se recompensa una sola vez.
                    reward += 2
                    self.metrics["sustain_ticks_hit"] += 1
                else:
                    reward -= 5
                    self.metrics["sustain_ticks_missed"] += 1
                    if pressed:
                        reward -= 1
                        self.metrics["wrong_presses"] += 1
                self.sosteniendo_nota[col] = bool(valid_hold)
                if self.state[-2, col] != 2:
                    key = "sustains_completed" if valid_hold else "sustains_broken"
                    self.metrics[key] += 1
            else:
                if pressed:
                    reward -= 1
                    self.metrics["wrong_presses"] += 1
                self.sosteniendo_nota[col] = False

        # Las notas avanzan hacia la fila de toque, que es la última.
        self.state = np.roll(self.state, shift=1, axis=0)
        self.state[0] = 0
        if self.note_style != "dense":
            self._generate_song_row()
        else:
            self._generate_dense_row()

        self.initial_steps += 1
        self.episode_reward += reward
        return (self._observation(), reward, False,
                self.initial_steps >= self.max_steps, self._info())


def main():
    import cv2

    env = GuitarHeroEnv()
    env.reset()
    print("Presiona 'q' en la ventana para salir...")
    try:
        while True:
            board, _, terminated, truncated, _ = env.step(env.action_space.sample())
            image = np.zeros((20, 5, 3), dtype=np.uint8)
            image[board == 1] = (255, 255, 255)
            image[board == 2] = (0, 255, 0)
            image = cv2.resize(image, (200, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Vista de la IA", image)
            if cv2.waitKey(50) & 0xFF == ord("q"):
                break
            if terminated or truncated:
                env.reset()
    finally:
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
