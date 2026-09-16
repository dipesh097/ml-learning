import numpy as np
import random

states=5
actions=2

q=np.zeros((states,actions))

alpha=0.1
gamma=0.9
epilsion=0.2

episodes=10

for episode in range(episodes):

    state=0

    while state != 4 :

        if random.uniform(0,1)<epilsion:

            action=random.randint(0,1)
        else :
            action=np.argmax(q[state])
        print(f"action:{action},state:{state}")

        if action==1:
            next_state=min(state+1,4)
        else:
            next_state=max(state-1,0)

        if next_state==4:
            r=+16
        else:
            r=-5
        print(f"next state :{next_state}")
        q[state,action]=q[state,action]+alpha*(r+gamma*np.max(q[next_state])-q[state,action])

        state=next_state
        print(f"updated state:{next_state}")


print("state=",state)
print("learned  q table",q)


