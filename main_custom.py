import gym
import numpy as np
import torch

env = gym.make("Riverraid-v0")
#print(env.action_space.n)

#print(env.observation_space.high)
#print(env.observation_space.low)

# state is simply a pixel pool in RGB
state = env.reset()
print(state)

state_b = np.array([state], copy=False)
#print(state_b)

state_v = torch.tensor(state_b)
print(state_v)


# vertical pixels
print(len(state))
# horiz pixels
print(len(state[0]))

# just prints all pixels of single frame
#for i in state:
    #print(i)

a = torch.randn(4,4)
print(a)

print(torch.max(a,0))








done = False
while not done:
    action = env.action_space.sample()
    # new_state is simply next frame pixels(after action applied)
    new_state, reward, done, _ = env.step(action)
    #print(reward)
    env.render()