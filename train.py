from stable_baselines3 import PPO
import os

# 1. Crear el entorno
env = GuitarHeroEnv()
env.reset()

# 2. Crear el Modelo (El Cerebro)
# Usamos "MlpPolicy": Una red neuronal simple para datos numéricos.
# verbose=1: Para que nos vaya contando qué tal le va en la consola.
model = PPO("MlpPolicy", env, verbose=1)

print("🤖 Comenza el entrenamiento... (La IA no sabe nada aún)")

# 3. ¡A ESTUDIAR!
# total_timesteps: Cuántos "frames" va a jugar para practicar.
# 10,000 es poco, pero sirve para probar que no explota.
model.learn(total_timesteps=10000)

print("✅ Entrenamiento terminado.")

# 4. Guardar el cerebro entrenado
model.save("mi_ia_guitarrista")