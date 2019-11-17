import gym
import random
import time
import numpy as np
from utils import isInAir, getJumpOutcome
from Action_Replay_Buffer import ActionReplayBuffer
import matplotlib.pyplot as plt


env = gym.make('MontezumaRevengeNoFrameskip-v4')

#seed everything
random.seed(42)
np.random.seed(42)
env.seed(42)

ARP = ActionReplayBuffer()
Goals = []

# from PIL import Image
#
# obs= env.reset()
# obs[115:117,:,:] = np.ones((2, 1, 3))
# img = Image.fromarray(obs, 'RGB')
# img.show()

num_random_actions = 6000
for episode in range(10):
    observation = env.reset()
    total_reward = 0
    LastInAir = False
    last_action = 0
    for t in range(num_random_actions):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        original_lives = info["ale.lives"]
        total_reward += reward
        if total_reward == 400:
            print("Congratulations!! You've reached the end of the first room :)")
            break
        inAir = isInAir(env, obs, action, last_action)
        jumping = LastInAir == False and inAir == True
        if jumping:
            jump_outcome = getJumpOutcome(env, original_lives)
            jumping = False
            ARP.store(obs, action, jump_outcome, env)
        if not jumping and not inAir:
            ARP.store(obs, action, reward, env)
        LastInAir = inAir
        last_action = action

    print("End of Random Action Sequence {0}".format(episode))
    subgoals = ARP.find_subgoals()
    print(len(subgoals))

#plot a heatmap
obs = env.reset()
plt.imshow(obs)
for subgoal_location in subgoals:
    plt.scatter(subgoal_location[0], subgoal_location[1], c = "yellow", alpha = 0.5, s = 10)
plt.show()


#timestep 905 doesn't realize jumping off teadmill and dying is -1 reward
