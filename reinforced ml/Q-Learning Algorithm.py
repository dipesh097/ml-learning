# Basic Setup
import numpy as np
import random
# Step 1️⃣ – Create the Environment
s
actions=[0,1]
tates=[0,1,2,3,4]
# Step 2️⃣ – Initialize Q-Table

Q=np.zeros((len(states),len(actions)))

# Step 3️⃣ – Define Hyperparameters
alpha = 0.1     # learning rate (kitna seekhna hai)
gamma = 0.9     # discount factor (future reward importance)
episodes = 10   # number of training iterations

# Step 4️⃣ – Define Reward Logic (simple environment)

def get_rewards(states,actions):
    if actions==1: #right move
        return 1
    else :  #left move
        return -1

# Step 5️⃣ – Training Loop

for episode in range(episodes):
    state=random.choice(states)

    for i in range(5):
        action=random.choice(actions)
        reward=get_rewards(state,action)
        next_state=random.choice(states)

        Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )
print("final q-table")
print(Q)