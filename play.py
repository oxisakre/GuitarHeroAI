import cv2
import numpy as np
from stable_baselines3 import PPO
from main import GuitarHeroEnv

#creamos el entorno
env = GuitarHeroEnv()
# cargamos a la IA entrenada
model = PPO.load("IA_guitarrista") 

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
    img = (obs * 255).astype(np.uint8)
    img_grande = cv2.resize(img, (200, 500), interpolation=cv2.INTER_NEAREST)
    # agrego BGR para leer mejor los numeros
    img_color = cv2.cvtColor(img_grande, cv2.COLOR_GRAY2BGR)
    # usamos cv2.putText(image, text, org, fontFace, fontScale, color, thickness) para poder mostrar el puntaje en pantalla
    cv2.putText(img_color, f"Puntos: {total_score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(img_color, f"Teclas: {keys_pressed}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("IA Jugando", img_color)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()