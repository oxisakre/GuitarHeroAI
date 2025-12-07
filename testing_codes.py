def lista(action):
    state = [1,1,0,1,0]
    reward = 0
    for col in range(5):
            if action[col] > 0:
                if state[col] > 0:
                    reward += 1
                else:
                    reward -= 1
            elif state[col] > 0:
                reward -= 5
    print(state)
    print(action)
    print(reward)
lista([0,1,0,1,0])