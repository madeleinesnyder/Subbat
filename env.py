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
import pdb

'''
Initialize the environment and h-params (adjust later)
'''
random.seed(42)
learning_rate = 2.5e-4
num_episodes = 10000
num_pre_training_episodes = 5000
discount = 0.99
batch_size = 256

'''
Make the Gym environment and open a tensorflow session
'''
env = gym.make('MontezumaRevengeNoFrameskip-v4')
sess = tf.Session()

'''
Initialize the replay buffers
'''
ARP = ActionReplayBuffer()
d1 = ReplayMemory()
d2 = ReplayMemory()
Goals = ARP.find_Goals()

'''
Build the controller and meta-Controller
'''
meta_controller_input_shape = env.observation_space.shape
controller_input_shape = env.observation_space.shape
controller_input_shape = (controller_input_shape[0] * 2, controller_input_shape[1], controller_input_shape[2])

meta_controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "goal_dim": 10, "input_shape": meta_controller_input_shape}
controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "action_dim": env.action_space.n, "input_shape": controller_input_shape}

meta_controller = MetaController(sess, meta_controller_hparams)
controller = Controller(sess, controller_hparams)

'''
Pre-training step. Get random goals, store things in d1, d2, ARP.
Must check if jumping before storing
'''
for i in range(num_pre_training_episodes):
    observation = env.reset()
    goal = ARP.random_Goal()
    done = False
    dead = False
    lives = 6
    while not (done or dead):
        F = 0
        initial_observation = observation
        while not (done or observation == goal or dead):
            #action space is discrete on set {0,1,...,17}
            action = controller.epsGreedy([observation, goal], env.action_space)
            next_observation, f, done, next_lives = env.step(action)
            r = intrinsic_reward(next_observation, goal)

            d1.store([initial_observation, goal], action, r, [next_observation, goal])
            controller_batch = d1.sample(batch_size)
            c_targets = controller_targets(controller_batch[:, 2], controller_batch[:, 3], controller, discount)
            controller.update(controller_batch[:, 0], controller_batch[:, 1], c_targets)

            jumping = isInAir(env,next_observation)
            dead = next_lives['ale.lives'] < lives
            if not jumping:
                ARP.store(tuple(tuple(row[0]) for row in next_observation),action,r)
            F += f
            observation = next_observation

        d2.store(initial_observation, goal, F, next_observation)
        if not (done or dead):
            goal = random_goal(Goals)
    controller.anneal()

'''
Main h-DQN algorithm
'''
for i in range(num_episodes):
    observation = env.reset()
    Goals = ARP.find_Goals()
    goal = meta_controller.epsGreedy(observation, Goals)
    done = False
    while not done:
        F = 0
        initial_observation = observation
        while not (done or observation == goal):
            #action space is discrete on set {0,1,...,17}
            action = controller.epsGreedy([observation, goal], env.action_space)
            next_observation, f, done, info = env.step(action)
            r = intrinsic_reward(next_observation, goal)

            d1.store([initial_observation, goal], action, r, [next_observation, goal])
            controller_batch = d1.sample(batch_size)
            c_targets = controller_targets(controller_batch[:, 2], controller_batch[:, 3], controller, discount)
            controller.update(controller_batch[:, 0], controller_batch[:, 1], c_targets)

            jumping = isInAir(env,next_observation)
            if jumping == False:
                ARP.store(next_observation,action,r,done)

            meta_controller_batch = d2.sample(batch_size)
            m_targets = meta_controller_targets(meta_controller_batch[:, 2], meta_controller_batch[:, 3], meta_controller, discount)
            meta_controller.update(controller_batch[:, 0], controller_batch[:, 1], m_targets)
            F += f
            observation = next_observation
        d2.store(initial_observation, goal, F, next_observation)
        if not done:
            goal = meta_controller.epsGreedy(Goals, obervation)
    meta_controller.anneal()
    controller.anneal()
env.close()
