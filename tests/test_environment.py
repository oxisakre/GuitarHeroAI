import unittest

import numpy as np

from main import GuitarHeroEnv


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        # Generador original sin notas aleatorias: cada test coloca las suyas.
        env = GuitarHeroEnv(note_style="dense", note_probability=0, **kwargs)
        env.reset(seed=123)
        self.addCleanup(env.close)
        return env

    def test_correct_sustain_is_rewarded_without_miss_penalty(self):
        env = self.make_env(include_hold_state=True)
        env.state[-1, 0] = 1
        env.state[-2, 0] = 2
        obs, reward, *_ = env.step([1, 0, 0, 0, 0])
        self.assertEqual(reward, 6)
        self.assertEqual(obs["held"].tolist(), [1, 0, 0, 0, 0])
        _, reward, _, _, info = env.step([1, 0, 0, 0, 0])
        self.assertEqual(reward, 2)
        self.assertEqual(info["sustain_ticks_hit"], 1)
        self.assertEqual(info["sustain_ticks_missed"], 0)
        self.assertEqual(info["sustains_completed"], 1)

    def test_releasing_a_sustain_cannot_be_recovered_without_a_head(self):
        env = self.make_env()
        env.state[-1, 0] = 1
        env.state[-3:-1, 0] = 2
        env.step([1, 0, 0, 0, 0])
        _, reward, *_ = env.step([0, 0, 0, 0, 0])
        self.assertEqual(reward, -5)
        _, reward, _, _, info = env.step([1, 0, 0, 0, 0])
        self.assertEqual(reward, -6)
        self.assertEqual(info["sustain_ticks_hit"], 0)
        self.assertEqual(info["sustain_ticks_missed"], 2)
        self.assertEqual(info["sustains_broken"], 1)

    def test_missed_head_does_not_activate_sustain(self):
        env = self.make_env()
        env.state[-1, 0] = 1
        env.state[-2, 0] = 2
        env.step([0, 0, 0, 0, 0])
        _, reward, _, _, info = env.step([1, 0, 0, 0, 0])
        self.assertEqual(reward, -6)
        self.assertEqual(info["notes_missed"], 1)
        self.assertEqual(info["sustain_ticks_hit"], 0)

    def test_chord_metrics_count_notes_not_rewarded_steps(self):
        env = self.make_env()
        env.state[-1, :3] = 1
        _, reward, _, _, info = env.step([1, 1, 0, 1, 0])
        self.assertEqual(reward, 6)
        self.assertEqual(info["notes_hit"], 2)
        self.assertEqual(info["notes_missed"], 1)
        self.assertEqual(info["wrong_presses"], 1)
        self.assertAlmostEqual(info["accuracy"], 2 / 3)

    def test_seed_reproduces_generated_notes(self):
        for style in ("song", "dense", "sustain"):
            with self.subTest(style=style):
                left, right = GuitarHeroEnv(note_style=style), GuitarHeroEnv(note_style=style)
                self.addCleanup(left.close)
                self.addCleanup(right.close)
                left.reset(seed=777)
                right.reset(seed=777)
                for _ in range(300):
                    board_left, *result_left = left.step([0] * 5)
                    np.random.random(13)  # El RNG global no altera al entorno.
                    board_right, *result_right = right.step([0] * 5)
                    np.testing.assert_array_equal(board_left, board_right)
                    self.assertEqual(result_left, result_right)

    @staticmethod
    def board_statistics(style, steps=20_000):
        env = GuitarHeroEnv(note_style=style)
        env.reset(seed=5)
        empty = occupied = heads = sustained_heads = 0
        for _ in range(steps):
            board, _, terminated, truncated, _ = env.step([0] * 5)
            if terminated or truncated:
                env.reset()  # Cada episodio sortea otra canción.
            empty += not board.any()
            occupied += np.count_nonzero(board)
            # Una cabeza en la fila 1 ya tiene decidido si lleva cola en la fila 0.
            heads += np.count_nonzero(board[1] == 1)
            sustained_heads += np.count_nonzero((board[1] == 1) & (board[0] == 2))
        env.close()
        return empty / steps, occupied / (steps * board.size), sustained_heads / heads

    def test_song_style_has_silences_and_fewer_sustains_than_dense(self):
        song_empty, song_occupied, song_sustains = self.board_statistics("song")
        dense_empty, dense_occupied, dense_sustains = self.board_statistics("dense")
        # El generador original solo deja la pista vacía al empezar cada episodio:
        # el modelo no aprende a quedarse quieto. Una canción sí tiene silencios.
        self.assertLess(dense_empty, 0.01)
        self.assertGreater(song_empty, 0.05)
        self.assertLess(song_occupied, dense_occupied / 2)
        self.assertGreater(dense_sustains, 0.85)
        self.assertLess(song_sustains, 0.4)

    def test_song_tails_stay_attached_above_their_heads(self):
        for style in ("song", "dense", "sustain"):
            with self.subTest(style=style):
                env = GuitarHeroEnv(note_style=style)
                self.addCleanup(env.close)
                env.reset(seed=9)
                for _ in range(5_000):
                    board, *_ = env.step([0] * 5)
                    tails = board[:-1] == 2
                    self.assertTrue(np.all(board[1:][tails] > 0),
                                    "Cada celda de cola debe tener debajo su cabeza u otra cola.")

    def test_dense_parameters_are_rejected_for_song_style(self):
        with self.assertRaises(ValueError):
            GuitarHeroEnv(note_probability=0.2)
        with self.assertRaises(ValueError):
            GuitarHeroEnv(note_style="otro")

    def test_sustain_practice_contains_long_chords_and_normal_songs(self):
        env = GuitarHeroEnv(note_style="sustain", include_hold_state=True, max_steps=400)
        self.addCleanup(env.close)
        episodes = set()
        long_chords = silences = 0
        for seed in range(12):
            obs, _ = env.reset(seed=seed)
            episodes.add(env._practice_episode)
            for _ in range(400):
                board = obs["board"]
                # Colas de más de una pantalla en dos carriles simultáneos.
                long_chords += np.count_nonzero(np.all(board == 2, axis=0)) >= 2
                silences += not board.any()
                action = ((board[-1] == 1) |
                          ((board[-1] == 2) & (obs["held"] == 1))).astype(np.int8)
                obs, _, _, _, info = env.step(action)
            self.assertEqual(info["notes_missed"], 0)
            self.assertEqual(info["wrong_presses"], 0)
            self.assertEqual(info["sustains_broken"], 0)
        self.assertEqual(episodes, {False, True})
        self.assertGreater(long_chords, 50)
        self.assertGreater(silences, 50)

    def test_observation_contract_and_independent_snapshots(self):
        for enriched in (False, True):
            with self.subTest(enriched=enriched):
                env = self.make_env(include_hold_state=enriched)
                obs, _ = env.reset(seed=10)
                self.assertTrue(env.observation_space.contains(obs))
                env.state[-1, 0] = 1
                obs, *_ = env.step([1, 0, 0, 0, 0])
                self.assertTrue(env.observation_space.contains(obs))
                board = obs["board"] if enriched else obs
                self.assertEqual(board.dtype, np.uint8)
                board[:] = 2
                self.assertFalse(np.all(env.state == 2))

    def test_reset_clears_episode_metrics_and_truncation(self):
        env = self.make_env(max_steps=1)
        env.state[-1, 0] = 1
        _, _, terminated, truncated, info = env.step([1, 0, 0, 0, 0])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["notes_hit"], 1)
        obs, info = env.reset()
        self.assertEqual(info["notes_hit"], 0)
        self.assertEqual(info["episode_reward"], 0)
        self.assertEqual(info["steps"], 0)
        self.assertFalse(any(env.sosteniendo_nota))
        self.assertFalse(obs.any())


if __name__ == "__main__":
    unittest.main()
