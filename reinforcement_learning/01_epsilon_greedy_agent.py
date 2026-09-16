import random
import numpy as np

states=["start","middle","goal"]
actions=["move-left","move-right"]

alpha=0.1
gamma=0.9
epsilion=0.2
episodes=15

q=np.zeros((len(states),len(actions)))

def get_reward(state,action):
    if state=="middle" and action=="move-right":
        return +10
    else:
        return -5

for episode in range(episodes):
    state=random.choice([0,1])

    for i in range(10):
        if random.uniform(0,1)<epsilion:
            action=random.choice([0,1])
        else:
            action=np.argmax(q[state,:])
            print(f"np.argmax[{state},:] is {actions[action]}")
        reward=get_reward(states[state],actions[action])
        print(f"state:{states[state]},action:{actions[action]},reward:{reward}")
        print(f"q matrix is {q}")
        next_state = random.choice([0, 1, 2])
#        upgration 1 and state
        q[state,action]=q[state,action]+alpha*(reward+gamma*np.max(q[next_state,:])-q[state,action])

        print(f"updated q is {q} and next state : {states[next_state]}")
        state=next_state
    print(f"here compelte {episode}th ")




