import random

states=["dirty","clean"]
actions=["suck","move"]

def get_reward(state,action):
    if state=="clean" and action=="move":
        return -1
    else :
        return 10

for i in range(15):
    state=random.choice(states)
    action=random.choice(actions)
    reward=get_reward(state,action)
    print(f"state :{state}, action : {action},reward: {reward}")