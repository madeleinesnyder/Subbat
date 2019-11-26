import gym
#from gym.utils.play import play
import random
import tensorflow as tf
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
import time
import numpy as np
import pandas as pd
import pickle
#from Action_Replay_Buffer import ActionReplayBuffer
from utils import *
#from getJumpOutcome import *#
import pdb
import sys
import os
import time

start_t = time.time()

'''
Initialize the environment and h-params (adjust later)
'''
random.seed(42)
learning_rate = 2.5e-4
num_episodes = 100
num_pre_training_episodes = 100
discount = 0.99
batch_size = 64

'''
Make the Gym environment and open a tensorflow session
'''
env = gym.make('MontezumaRevengeNoFrameskip-v4')
sess = create_tf_session(use_gpu = True, which_gpu = 1)
tf.global_variables_initializer().run(session=sess)

'''
Initialize subgoals 
'''
goals_xy = np.load("subgoals.npy")
goals = {}
#goals = {0: goals_xy[1], 1: goals_xy[3], 2: goals_xy[9], 3: goals_xy[9]}
for i in range(len(goals_xy)):
    goals[i] = goals_xy[i][np.newaxis, :]
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
Initialize the replay buffers
'''
d1 = ReplayMemory(name = "controller", buffer_capacity = 1024, storage_capacity = 4096, obs_shape = controller_input_shape)
d2 = ReplayMemory(name = "metacontroller", buffer_capacity = 1024, storage_capacity = 4096, obs_shape = meta_controller_input_shape)

'''
Pre-training step. Iterate over subgoals randomly and train controller to achieve subgoals
'''

stopwatch = 0
total_iterations = 0
goal_idx = random_goal_idx(goal_dim)
for i in range(num_pre_training_episodes):
    print("episode {0}".format(i))
    observation = env.reset()
    done = False
    dead = False
    at_subgoal = False
    iteration = 0
    lives = 6
    next_lives = 6
    goal_xy = goals[goal_idx]
    goal_mask = convertToBinaryMask([(goal_xy[0][0] - 5, goal_xy[0][1] - 5),(goal_xy[0][0] + 5, goal_xy[0][1] + 5)])
    
    while not (done or dead):
        F = 0
        initial_observation = observation
        while not (done or at_subgoal or dead):
            start = time.perf_counter()
            if iteration % 10 == 0:
                print("iteration {0} of episode {1}; controller epsilon {2}".format(iteration, i, controller.epsilon))

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

            at_subgoal = achieved_subgoal(env, next_observation, goal_xy)
            if at_subgoal:
                print("subgoal achieved at iteration {0} of episode {1}".format(iteration, i))
                r = 1
            else:
                r = 0

            next_observation_goal = np.concatenate([observation, goal_mask], axis = 0)
            # Store the obs, goal, action, reward, etc. in the controller buffer
            d1.store([observation_goal, action, r, next_observation_goal])

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
            total_iterations += 1

            # stuck
            if iteration % 500 == 0:
                dead = True
                
            end = time.perf_counter()
            stopwatch += end - start
            print(stopwatch, " ", total_iterations)
            print("average iterations per second: ", total_iterations/stopwatch)

        d2.store([initial_observation, goal_idx, F, next_observation])
        if not (done or dead) and at_subgoal:
            print("subgoal was: ", goals[goal_idx])
            goal_idx = random_goal_idx(goal_dim)
            goal_xy = goals[goal_idx]
            print("new subgoal is: ", goal_xy)
            at_subgoal = False 
    controller.anneal()
    print("Time of episode: ", end - start)

# Initialize array for storing performance
if not os.path.exists("results"):
    os.makedirs("results")

performanceDf = pd.DataFrame(columns = ["episode", "total_intrinsic_reward", "extrinsic_reward"])

'''
Main h-DQN algorithm
'''
for i in range(num_episodes):
    print("episode {0}".format(i))
    observation = env.reset()
    done = False
    dead = False
    at_subgoal = False
    lives = 6
    next_lives = 6
    goal_idx = meta_controller.epsGreedy(goals, observation)
    goal_xy = goals[goal_idx]
    goal_mask = convertToBinaryMask([(goal_xy[0][0] - 5, goal_xy[0][1] - 5),(goal_xy[0][0] + 5, goal_xy[0][1] + 5)])

    total_r_per_episode = 0
    total_F_per_episode = 0

    while not (done or dead):
        F = 0
        initial_observation = observation
        iteration = 0
        while not (done or at_subgoal or dead):
            if iteration % 10 == 0:
                print("iteration {0} of episode {1}; controller epsilon {2}".format(iteration, i, controller.epsilon))
            if iteration % 50 == 0:
                performanceDf.to_csv("results/performance.csv", index = False)

            # Get an action from the controller.
            observation_goal = np.concatenate([observation, goal_mask], axis = 0)
            action = controller.epsGreedy(observation_goal[np.newaxis, :, :, :], env.action_space)

            # STEP THE ENV
            next_observation, f, done, next_lives = env.step(action)

            # Check if ALE died during this env step.
            dead = next_lives['ale.lives'] < lives

            at_subgoal = achieved_subgoal(env, next_observation, goal_xy)
            if at_subgoal:
                print("subgoal achieved at iteration {0} of episode {1}".format(iteration, i))
                r = 1
            else:
                r = 0
            total_r_per_episode += r

            next_observation_goal = np.concatenate([observation, goal_mask], axis = 0)
            # Store the obs, goal, action, reward, etc. in the controller buffer
            d1.store([observation_goal, action, r, next_observation_goal])
            # Sample a batch from the buffer if there's enough in the buffer
            controller_batch = d1.sample(batch_size)
            # Get the controller targets
            c_targets = controller_targets(controller_batch[2], controller_batch[3], controller, discount)
            # Update the controller.
            controller.update(controller_batch[0], controller_batch[1], c_targets)

            meta_controller_batch = d2.sample(batch_size)
            m_targets = meta_controller_targets(meta_controller_batch[2], meta_controller_batch[3], meta_controller, discount)
            meta_controller.update(meta_controller_batch[0], meta_controller_batch[1], m_targets)
            F += f
            total_F_per_episode += f
            observation = next_observation

            # update lives according to whether or not he died
            lives = next_lives['ale.lives']

            # Update the observation
            observation = next_observation
            iteration += 1

            # stuck
            if iteration % 500 == 0:
                dead = True

        d2.store([initial_observation, goal_idx, F, next_observation])
        if not (done or dead):
            goal_idx = meta_controller.epsGreedy(goals, observation)
            goal_xy = goals[goal_idx]
            at_subgoal = False 

    newDf = pd.DataFrame([[i, total_r_per_episode, total_F_per_episode]], 
        columns = ["episode", "total_intrinsic_reward", "extrinsic_reward"])
    performanceDf.append(newDf)

    meta_controller.anneal()
    controller.anneal()

performanceDf.to_csv("results/performance.csv", index = False)
env.close()

end_t = time.time()

print("time elaspsed: {0} min".format((end_t - start_t) / 60))




