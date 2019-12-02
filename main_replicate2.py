import gym
#from gym.utils.play import play
import random
import tensorflow as tf
from replay_memory2 import ReplayMemory
from meta_controller2 import MetaController
from controller2 import Controller
import time
import numpy as np
import pandas as pd
#from Action_Replay_Buffer import ActionReplayBuffer
from utils import *
#from getJumpOutcome import *#
import pdb
import sys
import os

start_t = time.time()

'''
Initialize the environment and h-params (adjust later)
'''
random.seed(42)
learning_rate = 2.5e-4
num_episodes = 5
num_pre_training_episodes = 5
discount = 0.99
batch_size = 64

'''
Make the Gym environment and open a tensorflow session
'''
env = gym.make('MontezumaRevengeNoFrameskip-v4')
sess = tf.Session()
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
meta_controller_input_shape = (2, )
controller_input_shape = (4, )

meta_controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "goal_dim": goal_dim, "input_shape": meta_controller_input_shape}
controller_hparams = {"learning_rate": learning_rate, "epsilon": 1, "action_dim": env.action_space.n, "input_shape": controller_input_shape}

controller = Controller(sess, controller_hparams)
meta_controller = MetaController(sess, meta_controller_hparams)

'''
Initialize the replay buffers
'''
d1 = ReplayMemory(name = "controller", buffer_capacity = 128, storage_capacity = 4096, obs_shape = controller_input_shape)
d2 = ReplayMemory(name = "metacontroller", buffer_capacity = 128, storage_capacity = 4096, obs_shape = meta_controller_input_shape)

'''
Pre-training step. Iterate over subgoals randomly and train controller to achieve subgoals
'''

for i in range(num_pre_training_episodes):
    start = time.perf_counter()
    print("episode {0}".format(i))
    observation = env.reset()
    done = False
    dead = False
    at_subgoal = False
    lives = 6
    next_lives = 6
    goal_idx = random_goal_idx(goal_dim)
    goal_xy = goals[goal_idx]

    while not (done or dead):
        F = 0
        initial_observation = observation
        initial_ale_coord = np.array(get_ALE_coord(env, initial_observation))
        iteration = 0
        while not (done or at_subgoal or dead):
            if iteration % 10 == 0:
                print("iteration {0} of episode {1}; controller epsilon {2}".format(iteration, i, controller.epsilon))

            # Get an action from the controller.
            ale_coord = np.array(get_ALE_coord(env, observation))
            observation_goal = np.append(goal_xy, ale_coord)
            action = controller.epsGreedy(observation_goal[np.newaxis, :], env.action_space)

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

            next_ale_coord = np.array(get_ALE_coord(env, next_observation))
            next_observation_goal = np.append(goal_xy, next_ale_coord)
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
            F += f
            iteration += 1

            # stuck
            if iteration % 500 == 0:
                dead = True

        d2.store([initial_ale_coord, goal_idx, F, next_ale_coord])
        if not (done or dead):
            goal_idx = random_goal_idx(goal_dim)
            goal_xy = goals[goal_idx]
            at_subgoal = False 
    controller.anneal()
    end = time.perf_counter()
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
    
    total_r_per_episode = 0
    total_F_per_episode = 0

    while not (done or dead):
        F = 0
        initial_observation = observation
        initial_ale_coord = np.array(get_ALE_coord(env, initial_observation))
        iteration = 0
        while not (done or at_subgoal or dead):
            if iteration % 10 == 0:
                print("iteration {0} of episode {1}; controller epsilon {2}".format(iteration, i, controller.epsilon))
            if iteration % 50 == 0:
                performanceDf.to_csv("results/performance.csv", index = False)

            # Get an action from the controller.
            observation_goal = np.append(goal_xy, ale_coord)
            action = controller.epsGreedy(observation_goal[np.newaxis, :], env.action_space)
            
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

            next_ale_coord = np.array(get_ALE_coord(env, next_observation))
            next_observation_goal = np.append(goal_xy, next_ale_coord)
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

        d2.store([initial_ale_coord, goal_idx, F, next_ale_coord])
        if not (done or dead):
            goal_idx = meta_controller.epsGreedy(goals, next_ale_coord)
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



