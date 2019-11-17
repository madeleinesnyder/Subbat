import gym
# from gym.utils.play import play
import random
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
import time
import numpy as np

#Useful imports for debugging
from PIL import Image

random.seed(42)

d1 = ReplayMemory()
d2 = ReplayMemory()

#need to code in logic to end game at total reward of 400


#
# Goals = [] #TODO
# meta_controller = MetaController(0.0025)
# controller = Controller(0.00025)

# num_episodes = 10000
# env = gym.make('MontezumaRevenge-v0')
# for i in range(num_episodes):
#     #observation space is 210x160x3
#     observation = env.reset()
#     goal = meta_controller.epsGreedy(observation, Goals)
#     done = False
#     while not done:
#         F = 0
#         initial_observation = observation
#         while not (done or observation == goal):
#             #action space is discrete on set {0,1,...,17}
#             Actions = env.action_space
#             action = controller.epsGreedy([observation, goal], Actions)
#             next_observation, f, done, info = env.step(action)
#             r = 1 if next_observation == goal else 0
#             d1.store([observation, goal], action, r, [next_observation, goal])
#             controller.update(d1)
#             meta_controller.update(d2)
#             F += f
#             observation = next_observation
#         d2.store(initial_observation, goal, F, next_observation)
#         if not done:
#             goal = meta_controller.epsGreedy(observation, Goals)
#     meta_controller.anneal()
#     controller.anneal()
# env.close()


def isInAir(env, original_observation, action, last_action):

    test_action = 0
    clone_state = env.clone_full_state()
    for _ in range(2):
        observation, reward, done, info = env.step(test_action)
        test_observation, reward, done, info = env.step(test_action)
    env.restore_full_state(clone_state)

    #replace treadmill in picture
    observation[135:142,59:101,:] = np.zeros((7,42,3))
    test_observation[135:142,59:101,:] = np.zeros((7,42,3))

    #replace header in picture
    observation[:20,:,:] = np.zeros((20,160,3))
    test_observation[:20,:,:] = np.zeros((20,160,3))

    #replace skull in picture
    observation[165:180,52:114,:] = np.zeros((15, 62, 3))
    test_observation[165:180,52:114,:] = np.zeros((15, 62, 3))

    #replace key in picture
    observation[98:116,10:23,:] = np.zeros((18, 13, 3))
    test_observation[98:116,10:23,:] = np.zeros((18, 13, 3))

    #add in key logic
    key_test_observation = test_observation[23:45,53:67,:]
    key_original_observation = original_observation[23:45,53:67,:]

    if np.any(key_original_observation - key_test_observation):
        return True

    if not np.any(test_observation - observation):
        return False

    treadmill_observation = original_observation[135:136,60:100,:]
    valid_jumps = [1,10,11,12,14,15,16,17]
    if np.any(treadmill_observation) and action not in valid_jumps and last_action not in valid_jumps:
        return False

    return True

def getJumpOutcome(env, original_lives):
    #outcomes: death (-1), no death (0), reward (1)

    action = 0
    clone_state = env.clone_full_state()
    while True:
        obs, reward, done, info = env.step(action)
        lives = info['ale.lives']
        if reward != 0:
            env.restore_full_state(clone_state)
            return 1
        if lives < original_lives:
            env.restore_full_state(clone_state)
            return -1
        if not isInAir(env, obs, action, action):
            obs, reward, done, info = env.step(action)
            lives = info['ale.lives']
            if lives < original_lives:
                env.restore_full_state(clone_state)
                return -1
            env.restore_full_state(clone_state)
            return 0
    #Key reward is gained while isInAir is True. For skull case, self.lives
    #decremented at same time isInAir is false. For falling to death case,
    #isInAir if False one frame before self.lives is decremented

    return

# def canJump(original_observation):
#
#     test_action = 1
#     clone_state = env.clone_full_state()
#     for _ in range(2):
#         # time.sleep(0.1)
#         observation, reward, done, info = env.step(test_action)
#         # env.render()
#         # time.sleep(0.1)
#         test_observation, reward, done, info = env.step(test_action)
#         # env.render()
#     env.restore_full_state(clone_state)
#     # env.render()
#
#     #replace treadmill in picture
#     observation[135:142,59:101,:] = np.zeros((7,42,3))
#     test_observation[135:142,59:101,:] = np.zeros((7,42,3))
#
#     #replace header in picture
#     observation[:20,:,:] = np.zeros((20,160,3))
#     test_observation[:20,:,:] = np.zeros((20,160,3))
#
#     #replace skull in picture
#     observation[165:180,52:114,:] = np.zeros((15, 62, 3))
#     test_observation[165:180,52:114,:] = np.zeros((15, 62, 3))
#
#     #replace key in picture
#     observation[98:116,10:23,:] = np.zeros((18, 13, 3))
#     test_observation[98:116,10:23,:] = np.zeros((18, 13, 3))
#
#     if np.any(test_observation - observation):
#         return True
#     return False


# 0: do nothing
# 1: jump vertically up
# 2: nothing??? (may be an action not able to be used in current position, up maybe)
# 3: move one step right
# 4: move one step left
# 5: move one step down
# 6: looks like another move one step right
# 7: looks like another move one step left
# 8: looks like another move one step right
# https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py

env = gym.make('MontezumaRevengeNoFrameskip-v4')
# play(env)
env.render()
for i_episode in range(1):
    observation = env.reset()
    #jump right to the rope
    # actions = [11] + [0 for _ in range(60)]
    #jump to the upper right platform
    # actions = [3,3,3,3,3,3,3,3,3,3,3,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #go down the treadmill and walk to the rightmost edge
    #actions = [5 for _ in range(34)] + [3 for _ in range(45)]
    #go down the treadmill and walk left and die
#
#     #jump left and die
    # actions = [0,0,0,0,0,0,12] + [0 for _ in range(115)]
#     #testing the vertical jump issue
#     # actions = [5,5,5] + [1,5] * 50
#     #get to and jump over the skull
#     # actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
#     # [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)]
#     #get to the key
#     # actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
#     # [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)] + \
#     # [4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)]
#     #finish the room
    actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
    [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)] + \
    [4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)] + \
    [3 for _ in range(4)] + [5 for _ in range(40)] + [3 for _ in range(45)] + [11] + [0 for _ in range(30)] + \
    [3 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [12] + [0 for _ in range(20)] + \
    [12] + [0 for _ in range(30)] + [4 for _ in range(7)] + [2 for _ in range(40)] + [4 for _ in range(10)] + \
    [12] + [0 for _ in range(30)] + [4 for _ in range(35)]
#
#     #wouldn't mind making a more robust one
#
    total_reward = 0
    LastInAir = False
    last_action = 0
    for t in range(len(actions)):
        action = actions[t]
        obs, reward, done, info = env.step(action)
        original_lives = info["ale.lives"]
        total_reward += reward
        if total_reward == 400:
            print("Congratulations!! You've reached the end of the first room :)")
            break
        env.render()
        inAir = isInAir(env, obs, action, last_action)
        jumping = LastInAir == False and inAir == True
        if jumping:
            jump_outcome = getJumpOutcome(env, original_lives)
            print("jumping", jump_outcome)
            jumping = False
            time.sleep(3)
        print(t, action, inAir, info["ale.lives"], reward)
        LastInAir = inAir
        last_action = action
        #
        # if t > 70:
        #     time.sleep(2)

        # if t > 630 and t <= 645:
            # time.sleep(3)
        # if t == 635:
        #     img = Image.fromarray(obs, 'RGB')
        #     img.show()
        #     obs = obs[135:136,60:100,:]
        #     print(np.any(obs))
        #     img = Image.fromarray(obs, 'RGB')
        #     img.show()

        #one is because of foot still in air after jumping

        time.sleep(0.1)



#I think every action should now have a point of termination. Vertical jump terminates
#at the top, but that should actually be ok. ARP should be a set, and every time we
#perform an action

#peak of some jumps is an issue like I thought it would be. Hard code solution...
#jump takes 18 frames. I think death in general is fine. Anytime you lose a life,
#we know the action has terminated and we can trace back from there. Need to run this
#by Madeleine and through my head a bit more. Umm apparently you can't jump vertically
#two times in a row?

#code that creates the treadmill file for IsInAir. Only here for the record
# treadmill_spaces = set()
# ladder_actions = [5 for _ in range(32)]
# #get to the treadmill
# env.reset()
# for t in range(len(ladder_actions)):
#     action = ladder_actions[t]
#     observation, reward, done, info = env.step(action)
#     env.render()
#
# # end_actions = [5 for _ in range(34)] + [3 for _ in range(45)] + [11] + [0 for _ in range(20)] + \
# # [11] + [0 for _ in range(40)] + [5 for _ in range(40)] + [4 for _ in range(45)] + [12] + [0 for _ in range(25)] + \
# # [4 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [1] + [0 for _ in range(25)] + \
# # [3 for _ in range(4)] + [5 for _ in range(40)] + [3 for _ in range(45)] + [11] + [0 for _ in range(30)] + \
# # [3 for _ in range(50)] + [2 for _ in range(40)] + [4 for _ in range(5)] + [12] + [0 for _ in range(20)] + \
# # [12] + [0 for _ in range(30)] + [4 for _ in range(7)] + [2 for _ in range(40)] + [4 for _ in range(10)] + \
# # [12] + [0 for _ in range(30)] + [4 for _ in range(35)]
# actions = [5] + [3 for _ in range(49)] + [4 for _ in range(29)] + [3 for _ in range(87)] + \
# [4 for _ in range(29)] + [3 for _ in range(87)] + [4 for _ in range(29)] + [3 for _ in range(87)] + \
# [0 for _ in range(75)] + [3 for _ in range(60)]
# for t in range(len(actions)):
#     env.render()
#     action = actions[t]
#
#     observation, reward, done, info = env.step(action)
#     observation = observation[117:136,50:110,:]
#
#     # if t == 10:
#     #     img = Image.fromarray(observation, 'RGB')
#     #     img.show()
#
#     treadmill_spaces.add(str(observation))
#     print(len(treadmill_spaces))
#     # time.sleep(0.05)

# env.reset()
# for t in range(len(end_actions)):
#     env.render()
#     action = end_actions[t]
#
#     observation, reward, done, info = env.step(action)
#     observation = observation[100:136,50:110,:]
#
#     treadmill_spaces.add(str(observation))
#     print(len(treadmill_spaces))
#     time.sleep(0.05)

# filename = "treadmill_spaces"
# outfile = open(filename, 'wb')
# pickle.dump(treadmill_spaces, outfile)


#best path forward may be just digging into the data and seeing what it looks like.
#it'd be better if it was only one of four directions, but issue is that looks like jumping
#horizontally as well. But if travel different distances, that could be resolved.

#Turning upside down while dying?

#could test if all black is around Ale. Hard.

#NOOP!!! Treadmill and Skull will be issue. I think it would be possible to
#hardcode treadmill... watch out for skull moving, just window it. Makes it
#easier to store actually. Can even try cutting above skull for the regular
#NOOP check but that may create weird edge cases that worry me. Cut out treadmill too!!
#Also may need to cut banner at top. Also need to box out key still


#Window will also be difficult. The only time the window size won't be 1 though is
#when an action results in us being IsInAir. So I may be able to play the same
#subtraction game, I'll have to try visualizing it. Looks like taking difference
#between current and last observation gives near perfect trace of where ALE is at.
#so can just take a rough outline of the rectangle from where ALE starts the action
#to where ALE ends the action.
