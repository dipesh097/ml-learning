import random
import numpy as np

states=["dirty","clean"]
actions=["suck","move"]

alpha=0.1
gamma=0.9
episodes=15

q=np.zeros((len(states),len(actions)))

def get_reward(state,action):
    if state=="dirty" and action=="suck":
        return +10
    elif state=="clean" and action=="move":
        return +5
    else:
        return -5
for episode in range(episodes):
    state=random.choice([0,1])
    for i in range(5):
        print(f"from here start  {i}th loop of {episode}th episode for q value upgration \n")
        action=random.choice([0,1])
        reward=get_reward(states[state],actions[action])
        next_state=random.choice([0,1])

#q learning , main line of rl
        q[state,action]=q[state,action]+alpha*(reward+gamma*np.max(q[next_state,:])-q[state,action])

        print("value of q is :",q)

        print(f"state:{states[state]},action:{actions[action]},reward:{reward},next state:{states[next_state]}")
        print(f"here complete {i}th loop for q value upgration \n")
        state = next_state
    print(f"here complete {episode}th episode")

print("the q value , matrix for this rl system is \n",q)