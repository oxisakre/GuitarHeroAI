from stable_baselines3 import PPO
from main import GuitarHeroEnv
import os
#para dejarlo toda la noche, importamos callbacks, que hacen checkpoints cada tantos pasos dados
from stable_baselines3.common.callbacks import CheckpointCallback

# Creamos el entorno
env = GuitarHeroEnv()
env.reset()

# Creamos el modelo a entrenar
# Usamos "MlpPolicy" porque es una red neuronal simple para datos numéricos
# verbose=1: para que nos diga en la consola como le esta yendo
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_guitar_hero_logs/")    (modelo para crear una IA de cero)
# modelo para seguir entrenando la IA creada con el otro modelo
model = PPO.load("IA_guitarrista", env, verbose=1, tensorboard_log="./ppo_guitar_hero_logs/")

# Guardamos una copia cada 500,000 pasos 
checkpoint_callback = CheckpointCallback(
    save_freq=2000000, 
    save_path='./logs_ia/', 
    name_prefix='ia_nocturna'
)

print("Empezando el entrenamiento")

# total_timesteps: Cuántos "frames" va a jugar para practicar
model.learn(total_timesteps=60000000, callback=checkpoint_callback)

print("Entrenamiento terminado.")

# Guardamos el cerebro entrenado
model.save("IA_guitarrista")