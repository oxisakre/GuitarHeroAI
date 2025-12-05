from stable_baselines3 import PPO
from main import GuitarHeroEnv
import os

# Creamos el entorno
env = GuitarHeroEnv()
env.reset()

# Creamos el modelo a entrenar
# Usamos "MlpPolicy" porque es una red neuronal simple para datos numéricos
# verbose=1: para que nos diga en la consola como le esta yendo
model = PPO("MlpPolicy", env, verbose=1)

print("Empezando el entrenamiento")

# total_timesteps: Cuántos "frames" va a jugar para practicar
# 10,000 es poco, pero sirve para probar que no explota.
model.learn(total_timesteps=10000)

print("Entrenamiento terminado.")

# Guardamos el cerebro entrenado
model.save("IA_guitarrista")