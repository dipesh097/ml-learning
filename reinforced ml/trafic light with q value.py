import numpy as np
import random


states = ["red", "green"]
actions = ["go", "stop"]

alpha=0.1
gamma=0.9
episodes=15

q=np.zeros((len(states),len(actions)))

def get_reward(state,action):
    if state=="red" and action=="stop":
        return +10
    elif state=="green" and action=="go":
        return +10
    else :
        return -10


for episode in range(episodes):
    print(f"from start {episode}th episode")
    state=random.choice([0,1])
    for i in range(10):
        print(f"from here start {i}th loop of {episode}th for q upgration\n")
        action=random.choice([0,1])
        reward=get_reward(states[state],actions[action])
        next_state=random.choice([0,1])
        q[state,action]=q[state,action]+alpha*(reward+gamma*np.max(q[next_state,:])-q[state,action])
        print(f"state:{states[state]},action:{actions[action]}, reward :{reward},q:{q},next state is :{states[next_state]}")
        state=next_state
        print(f" here complete {i}th loop of {episode}th for q upgration\n")
    print(f"from start {i}th episode\n")
print("final q value is :",q)


