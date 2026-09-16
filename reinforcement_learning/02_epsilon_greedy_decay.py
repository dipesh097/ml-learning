import random
import numpy as np

states=["dirty","clean"]
actions=["suck","move"]

alpha=0.1
gamma=0.9
epsilion=0.2
episodes=10

q=np.zeros((len(states),len(actions)))

def get_reward(state,action):
    if state=="dirty" and action=="suck":
        return +10
    elif state=="clean" and action=="move":
        return +10
    else:
        return -10

for episode in range(episodes):
    print(f"{episode}th episode  👍:")
    state=random.choice([0,1])
    for i in range(10):
        if random.uniform(0,1)<epsilion:
            action=random.choice([0,1])
        else :
            action=np.argmax(q[state,:])

        reward=get_reward(states[state],actions[action])
        print(f"for {i}th loop q matrix is {q} \n ,state:{states[state]} , action:{actions[action]} and reward:{reward}")
        next_state= 1 if state==0 else 0

        q[state,action]=q[state,action]+alpha*(reward+np.max(q[next_state,:])-q[state,action])
        print(f"\n next state:{states[next_state]}, index of of max q for {i}th loop :{np.max(q[next_state,:])}")
        print(f"\n (q[next_state,:]) is {q[next_state, :]} ")
        print(f"updated q matrix is {q}")
        state=next_state
