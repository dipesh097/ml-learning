import random

states=["start","middle","goal"]
actions=["left move","right move"]

def get_reward(state,action):
    if state=="middle" and action=="right move" :
        return +10
    else :
        return -1

for i in range(10):
    state=random.choice(states)
    action=random.choice(actions)
    reward=get_reward(state,action)
    print(f"state:{state},action:{action},reward:{reward}")
