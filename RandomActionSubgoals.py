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

num_episodes = 2500
num_random_actions = 2000

all_actions = [[np.random.randint(0,18) for _ in range(num_random_actions)] for _ in range(num_episodes)]

for episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    LastInAir = False
    last_action = 0
    for t in range(num_random_actions):
        action = all_actions[episode][t]
        obs, reward, done, info = env.step(action)
        lives = info["ale.lives"]
        total_reward += reward
        if total_reward == 400:
            print("Congratulations!! You've reached the end of the first room :)")
            break


        # if episode == 17 and (t > 1485 or t < 1510):
        #     env.render()
        #     time.sleep(0.5)
        #
        # if episode == 43 and (t > 1515 or t < 1545):
        #     env.render()
        #     time.sleep(0.5)
        #
        # if episode == 74 and (t > 1095 or t < 1120):
        #     env.render()
        #     time.sleep(0.5)



        inAir = isInAir(env, obs, action, last_action)
        jumping = LastInAir == False and inAir == True
        if jumping:
            jump_outcome = getJumpOutcome(env, lives)
            jumping = False
            ARP.store(obs, action, jump_outcome, env,t)
        if (not jumping) and (not inAir):
            #need logic to override reward if die by walking, aka skull
            # reward = reward if lives == 6 else -1
            ARP.store(obs, action, reward, env,t)
        LastInAir = inAir
        last_action = action

        if lives == 5:
            #episode ends anytime ALE dies
            break

    print("End of Random Action Sequence {0}".format(episode))
    subgoals = ARP.find_subgoals()
    print(len(subgoals))

#One outstanding issue I can envision -- subgoals used to be states, but now they are not.
#This means ALE will not know when to jump because the skull is ahead versus just
#jumping because ALE reached a coordinate that had the skull ahead in a past episode,
#but doesn't currently. A brute force way to resolve this is by checking the location of ALE,
#and if ALE is in the skull zone, storing the state as well. Then we'd have to do
#an additional check in at_subgoal to see first if at coordinate of skull zone, then
#if the state also matches up (and therefore the skull is ahead). I think I'm ok with
#this from a hack perspective, as ALE already has access to the state in general, so
#I don't think we're cheating our algorithm or anything.

#plot a heatmap
obs = env.reset()
plt.imshow(obs)
for subgoal_location in subgoals:
    plt.scatter(subgoal_location[0], subgoal_location[1], c = "yellow", alpha = 0.5, s = 10)
plt.show()
print("Done")
