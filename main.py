# gymnasium es una libreria para crear entornos de aprendizaje por refuerzo
import gymnasium as gym
# spaces es una libreria para definir el espacio de observacion y acciones
from gymnasium import spaces
# numpy es una libreria para manejar matrices
import numpy as np
# cv2 es una libreria para manejar imagenes
import cv2
# PPO es un algoritmo de aprendizaje por refuerzo
from stable_baselines3 import PPO

class GuitarHeroEnv(gym.Env):
    def __init__(self):
        # Definimos la cantidad de acciones que puede realizar el agente
        self.action_space = spaces.Discrete(6)
        # Definimos el espacio de observacion, 20x5(filas, columnas), cada celda puede ser 0 o 1
        self.observation_space = spaces.Box(low=0, high=2, shape=(20, 5), dtype=np.uint8)

    def reset(self, seed=None, options=None):
    # Esto es necesario para gestionar la aleatoriedad correctamente
        super().reset(seed=seed)
    
    # Creamos la matriz de 20x5 llena de ceros
        self.state = np.zeros((20, 5), dtype=int)
    
    # Devolvemos el estado inicial y un diccionario vacío (info)
        return self.state, {}
    
    def step(self, action):
        # Definimos la recompensa
        reward = 0
        terminated = False
        truncated = False

        # la IA intenta tocar
        if action > 0:
            columna = action - 1
            if self.state[19, columna] > 0:
                reward += 1 # Recompensa por tocar una nota 
                self.state[19, columna] = 0 # La nota que se toca, se elimina de la matriz
            else:
                reward -= 1 # Castigo por tocar el aire
        # Revisar si se nos escapó alguna nota en la última fila (fila 19)
        # Recorremos las 5 columnas
        for col in range(5):
            if self.state[19, col] == 1:
                # si hay una nota en la ultima fila, se resta 5 puntos
                reward -= 5 
        
        # usamos np.roll para mover la matriz una fila hacia arriba y borrar la ultima fila
        self.state = np.roll(self.state, shift=1, axis=0)

        # limpiamos la ultima fila
        self.state[0] = 0

        # Probabilidad de nota nueva (0.1 = 10% de probabilidad) , si queremos mas dificultad, aumentamos el valor
        dificultad = 0.1 
        
        for col in range(5):
            if np.random.random() < dificultad:
                self.state[0, col] = 1
        # devolvemos el diccionario vacio , para evitar errores
        info = {}
        return self.state, reward, terminated, truncated, info

env = GuitarHeroEnv()
reset = env.reset()
print(reset)