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
from Action_Replay_Buffer import ActionReplayBuffer
from isInAir import *
from utils import *
from getJumpOutcome import *
import pdb
import sys

'''
Initialize the environment and h-params (adjust later)
'''
random.seed(42)
learning_rate = 2.5e-4
num_episodes = 20
num_pre_training_episodes = 5
discount = 0.99
batch_size = 256

'''
Make the Gym environment and open a tensorflow session
'''
env = gym.make('MontezumaRevengeNoFrameskip-v4')
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

'''
Initialize the replay buffers
'''
d1 = ReplayMemory(name = "controller", write_capacity = 20)
d2 = ReplayMemory(name = "metacontroller", write_capacity = 20)

'''
Initialize subgoals 
'''
goals_xy = np.load("subgoals.npy")
#goals = {}
#for i in range(len(goals_xy)):
#    goals[i] = goals_xy[i]
goals = {0: goals_xy[0]}
goal_dim = len(goals)

'''
Build the controller and meta-Controller
'''
meta_controller_input_shape = env.observation_space.shape
controller_input_shape = env.observation_space.shape
controller_input_shape = (controller_input_shape[0] * 2, controller_input_shape[1], controller_input_shape[2])

meta_controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "goal_dim": goal_dim, "input_shape": meta_controller_input_shape}
controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "action_dim": env.action_space.n, "input_shape": controller_input_shape}

controller = Controller(sess, controller_hparams)
meta_controller = MetaController(sess, meta_controller_hparams)

'''
Pre-training step. Iterate over subgoals randomly and train controller to achieve subgoals
'''
for i in range(num_pre_training_episodes):
    print("episode {0}".format(i))
    observation = env.reset()
    done = False
    dead = False
    at_subgoal = False
    lives = 6
    next_lives = 6
    goal_idx = random_goal_idx(goal_dim)
    goal_xy = goals[goal_idx]
    goal_mask = convertToBinaryMask([(goal_xy[0] - 5, goal_xy[1] - 5),(goal_xy[0] + 5, goal_xy[1] + 5)])

    while not (done or dead):
        F = 0
        initial_observation = observation
        iteration = 0
        while not (done or at_subgoal or dead):
            if iteration % 10 == 0:
                print("iteration {0} of episode {1}; ALE lives {2}; controller epsilon {3}".format(iteration, i, lives, controller.epsilon))

            # Get an action from the controller.
            observation_goal = np.concatenate([observation, goal_mask], axis = 0)
            action = controller.epsGreedy(observation_goal[np.newaxis, :, :, :], env.action_space)

            # STEP THE ENV
            next_observation, f, done, next_lives = env.step(action)

            # Check if ALE died during this env step.
            dead = next_lives['ale.lives'] < lives

            # Record the reward if reached the subgoal.
            #if iteration % 10 == 0:
            #    print("ALE coord: {0}".format(get_ALE_coord(env, observation)))
            #    print("goal coord: {0}".format(goal_xy))

            #at_subgoal = achieved_subgoal(env, next_observation, goal_xy)
            if at_subgoal:
                print("subgoal achieved at iteration {0} of episode {1}".format(iteration, i))
                r = 1
            else:
                r = 0

            # Store the obs, goal, action, reward, etc. in the controller buffer
            d1.store(np.concatenate([observation, goal_mask], axis = 0), action, r, np.concatenate([next_observation, goal_mask], axis = 0))

            # Sample a batch from the buffer if there's enough in the buffer
            controller_batch = d1.sample(batch_size)

            # Get the controller targets
            c_targets = controller_targets(controller_batch[2], controller_batch[3], controller, discount)

            # Update the controller.
            controller.update(controller_batch[0], controller_batch[1], c_targets)

            # update lives according to whether or not he died
            lives = next_lives['ale.lives']

            # Update the observation
            observation = next_observation
            iteration += 1
        #d2.store(initial_observation, goal_idx, F, next_observation)
        if not (done or dead):
            goal_idx = random_goal_idx(goal_dim)
            goal_xy = goals[goal_idx]
            at_subgoal = False 
    controller.anneal()
    
env.close()

