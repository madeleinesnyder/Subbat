# Testing environment

import gym
#from gym.utils.play import play
import random
import tensorflow as tf
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
import time
import numpy as np
import pickle
import pdb
from Action_Replay_Buffer import ActionReplayBuffer
from isInAir import *
from utils import *

'''
Initialize the environment and h-params (adjust later)
'''
random.seed(41)

'''
Make the Gym environment and open a tensorflow session
'''
env = gym.make('MontezumaRevengeNoFrameskip-v4')
sess = tf.Session()

'''
Initialize the replay buffers
'''
ARP = ActionReplayBuffer()
Goals = ARP.find_Goals()

'''
Pre-training step. Get random goals, store things in ARP.
Must check if dead, done, or jumping before storing
'''
for i in range(10):
    pdb.set_trace()
    observation = env.reset()
    dead = False
    done = False
    jumping = False
    while not (done or dead):
        F = 0
        initial_observation = observation
        lives = 6
        while not (done or dead):
            #action space is discrete on set {0,1,...,17}
            action = env.action_space.sample()
            if action == 12:
                action = 3
            # Get if ALE is done, jumping, or dead in this next step.
            next_observation, reward, done, next_lives = env.step(action)
            jumping = isInAir(env,next_observation)
            dead = next_lives['ale.lives'] < lives
            if not jumping:
                ARP.store(tuple(tuple(row[0]) for row in observation),action,reward)
            F += 1
            observation = next_observation
            lives = next_lives['ale.lives']

ARP.find_Goals()
Goals = ARP.Goals
print(len(Goals))
print(str(len(Goals))+" subgoals identified")
env.close()
