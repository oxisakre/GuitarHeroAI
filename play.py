import cv2
import numpy as np
from stable_baselines3 import PPO
from main import GuitarHeroEnv

#creamos el entorno
env = GuitarHeroEnv()
# cargamos a la IA entrenada
model = PPO.load("IA_guitarristaMultiple") 

obs, _ = env.reset()

print("la IA esta tocando , para salir presionar la Q)")
total_score = 0
keys_pressed = 0
while True:
    #  el lugar de hacer sample , que seria lo aleatorio como en el test de main, hacemos que haga un predict y que tome la decision
    # deterministic=True hace que la IA elija siempre la MEJOR opcion que conoce
    action, _ = model.predict(obs, deterministic=True)
    
    # Ejecutamos la acción en el juego
    obs, reward, terminated, truncated, info = env.step(action)
    # vamos viendo el puntaje de la IA
    total_score += reward
    if reward >= 1:
        keys_pressed +=1
    #vemos el juego en vivo
    img_color = np.zeros((20, 5, 3), dtype=np.uint8) # lienzo negro vacío (Alto, Ancho, 3 Canales de color)
    img_color[obs == 1] = [255, 255, 255] # Pintar las Cabezas (1) de Blanco
    img_color[obs == 2] = [0, 255, 0] # Pintar las Colas (2) de Verde
    img_grande = cv2.resize(img_color, (600, 1200), interpolation=cv2.INTER_NEAREST) # Agrandar esa imagen de color
    # usamos cv2.putText(image, text, org, fontFace, fontScale, color, thickness) para poder mostrar el puntaje en pantalla
    cv2.putText(img_grande, f"Puntos: {total_score}", (10, 20), # Y=20
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(img_grande, f"Teclas: {keys_pressed}", (10, 50), # Y=50
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow("IA Jugando", img_grande)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()