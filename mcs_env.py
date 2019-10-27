import gym
from gym.utils.play import play
import random
from replay_memory import ReplayMemory
from meta_controller import MetaController
from controller import Controller
import time
import numpy as np
import pickle
import pdb
from Action_Replay_Buffer import *
from isInAir import *

random.seed(42)
# Initialize the gym environment
env = gym.make('MontezumaRevengeNoFrameSkip-v4')
# Store the first observation as the last_obs
last_obs = env.reset()
# Render the MontezumaRevengeNoFrameSkip
env.render()
# Initialize Action Replay Buffer, d1 replaymemory, d2 replaymemory
ARP = ActionReplayBuffer
d1 = ReplayMemory()
d2 = ReplayMemory()
# Define the contrnoller and metacontroller
controller = Controller(0.00025)
meta_controller = MetaController(0.0025)
# For every episode, get an observation and take many actions
ep_num = 1
for i_episode in range(ep_num):
    # Get observation by resetting the envrionment
    observation = env.reset()
    # Find the subgoals from the ARP
    Goals = ARP.find_subgoals()
    # Use the meta contrnoller to select one of the Goals
    goal = meta_controller.epsGreedy(observation,Goals)
    # set done to false
    done = False
    # Before the end of a run...
    while not done:
        F = 0
        # define the initial observation
        initial_observation = observation
        # Before the end f the run or achievement of a goal
        while not (done or observation == goal):
            Actions = env.action_space
            # Get an action from the controller by sampling according to epsilon greedy
            action = controller.epsGreedy([observation,goal],Actions)
            # Step the environment and get obs,reward,done,info. info=lives
            obs,reward,done,info = env.step(action)
            # Render the MontezumaRevengeNoFrameSkip
            env.render()
            # Ask if ALE is jumping. If he is, we can't store the reward of that action yet.
            jumping = isInAir(env,obs)
            if jumping == False or done == False:
                r = 1 if obs == goal else 0
                d1.store([last_obs,goal],action,r,[obs,goal])
            if jumping == False or done == False or last_info==info or obs != goal:
                ARP.store(obs,action,reward,done) # This is the furry line
            controller.update(d1)
            meta_controller.update(d2)
            # Set current observation to last observation
            last_obs = obs
            # Set current info to last info
            last_info = info
            # Conditions for storing: 1. ALE isInAir == False; 2. last_info > info (he's dead); 3. done == True; if observation == goal
            time.sleep(0.4)
        d2.store(initial_observation,goal,F,obs)
        if not done:
            goal = meta_controller.epsGreedy(observation,Goals)
    meta_controller.anneal()
    controller.anneal()
env.close()
