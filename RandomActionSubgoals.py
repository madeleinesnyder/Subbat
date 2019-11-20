import gym
import random
import time
import numpy as np
from utils import isInAir, getJumpOutcome
from Action_Replay_Buffer import ActionReplayBuffer
import matplotlib.pyplot as plt
import pdb


env = gym.make('MontezumaRevengeNoFrameskip-v4')

#seed everything
random.seed(42)
np.random.seed(42)
env.seed(42)

ARP = ActionReplayBuffer()
Goals = []

# def map_subgoals_so_far(subgoals):
#     obs = env.reset()
#     plt.imshow(obs)
#     for subgoal_location in oc:
#         plt.scatter(subgoal_location[0], subgoal_location[1], c = "yellow", alpha = 0.5, s = 10)
#     plt.show()
#     print("Done")

def map_actions_so_far(subgoals):
    obs = env.reset()
    plt.imshow(obs)
    for subgoal_location in subgoals:
        if subgoal_location == -1:
            continue
        else:
            plt.scatter(subgoal_location[1], subgoal_location[0], c = "yellow", alpha = 0.5, s = 10)
    plt.show()
    pdb.set_trace()
    print("Done")

# all_actions = [[np.random.randint(0,18) for _ in range(6000)] for _ in range(10)]

# all_actions = [[4,4,4,1] + [0 for _ in range(5595)] for _ in range(10)]

# env.render()
num_random_actions = 6000
for episode in range(100):
    oc = []
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
        # env.render()
        inAir = isInAir(env, obs, action, last_action)
        jumping = LastInAir == False and inAir == True
        if jumping:
            jump_outcome = getJumpOutcome(env, original_lives)
            # print(t, action, "jumping", jump_outcome)
            jumping = False
            observed_coords,rew = ARP.store(obs, action, jump_outcome, env)
            #oc.append(observed_coords)
            #env.render()
            #time.sleep(.5)
        if (not jumping) and (not inAir):
            # print(t, "storing from the ground")
            #env.render()
            #time.sleep(.5)
            #oc.append(observed_coords)
            ARP.store(obs, action, reward, env)
        LastInAir = inAir
        last_action = action

    #map_actions_so_far(oc)
    print("End of Random Action Sequence {0}".format(episode))
    subgoals = ARP.find_subgoals()
    #map_subgoals_so_far(subgoals)
    print(len(subgoals))

#the first issue is at timestep 22

#plot a heatmap
obs = env.reset()
pdb.set_trace()
plt.imshow(obs)
for subgoal_location in subgoals:
    plt.scatter(subgoal_location[1], subgoal_location[0], c = "yellow", alpha = 0.5, s = 10)
plt.show()
pdb.set_trace()
print("Done")

#timestep 905 doesn't realize jumping off teadmill and dying is -1 reward
