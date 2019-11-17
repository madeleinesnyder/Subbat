import numpy as np
from collections import defaultdict
import pdb
from utils import *

class ActionReplayBuffer:

    '''
    TODO: Find location of subgoal generator that observation starting from where the subgoal is and take 1 action, subtract the 2 and that's the location of the subgoals
    This gives 3d matrix of subgoal, go into 2d.
    Radius calculation
    '''

    def __init__(self,capacity = np.float("inf")):
        self.capacity = capacity
        self.memories = set()
        self.ARP_dict = defaultdict(list)
        self.subgoal_locations = []

    def store(self, obs , action , reward , env):
        '''
        Stores each state-action-reward sample from the environment
        If the capacity of the buffer is exceeded, delete the first sample
        '''
        # arg1 - state, arg2 - action, arg3 - reward.
        # Convert to a tuple to store in memories
        obs = self.get_observation_coordinates(env,obs)
        # If ALE is likely dead, return -1 -1 from get_observation_coordinates and don't store this.
        if obs == (-1,-1):
            return
        memory = [obs, action, reward]
        memory = tuple(memory)
        # If this new memory is already in the set of the memories, don't put it in.
        self.memories.add(memory)
        # Eliminated because we can't index into a set
        if len(self.memories) > self.capacity:
            print("Change data structure to ordered dict")
            self.memories = self.memories[1:]

    def find_subgoals(self):
        '''
        Find the states that corrospond to goals.
        '''
        if len(self.memories) < 1:
            return self.subgoal_locations
        for memory in self.memories:
            self.ARP_dict[memory[0]].append(memory[1:])

        for key, value in self.ARP_dict.items():
            unique_arps = set(value)
            if (len(unique_arps) > 1):
                self.subgoal_locations.append(key)
        return self.subgoal_locations

    def attempt_action(self, env, action):
        '''
        This function attempts different actions to get an xy location for ALE (4 and 5)
        '''
        env = env.unwrapped
        clone_state = env.clone_full_state()
        test_action = action
        for _ in range(2):
            observation, reward, done, info = env.step(test_action)
            test_observation, reward, done, info = env.step(test_action)

        # Black box things that move
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

        rgb_coords = test_observation-observation
        if np.sum(rgb_coords) != 0:
            env.restore_full_state(clone_state)
            return rgb_coords
        else:
            env.restore_full_state(clone_state)
            return np.zeros(1)

    def get_observation_coordinates(self,env,observation):
        '''
        Get the xy location of the goal from the state
        Fix this unwrapping shit. first check 4 (left); then check 5 (down)
        '''
        obs = observation
        for action in [4,5,11]:
            rgb_coords = self.attempt_action(env,action)
            if np.sum(rgb_coords) > 0:
                rgb_coords = rgb_coords
                break
        if (action == 11 and np.sum(rgb_coords) == 0):
            return (-1,-1)
        nonzero_coords = np.where(rgb_coords[:,:,0] != 0)
        [mean_x,mean_y] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
        coords = (float(np.ceil(mean_x)),float(np.ceil(mean_y)))
        return coords

    def at_subgoal(self,env,observation,goal):
        '''
        Is ALE near enough to a subgoal?
        '''
        #goal_xy = convertToSubgoalCoordinates(goal)
        goal_xy = goal
        coords = self.get_observation_coordinates(env,observation)
        k = 10 # Because Calvin says so.
        if np.linalg.norm(coords - goal_xy) < k:
            return True
        else:
            return False

    def load_temp_subgoals(self,sg_file):
        '''
        For testing purposes only. Loading in Calvin's list of subgoals npy array
        '''
        sgs = np.load(sg_file)
        for i in range(len(sgs)):
            subgoal_xy = convertToSubgoalCoordinates(sgs[i])
            self.subgoal_locations.append(subgoal_xy)

    def get_random_goal(self):
        '''
        Get a random subgoal to initialize the exploration run in env.
        '''
        idx = np.random.randint(len(self.subgoal_locations))
        return self.subgoal_locations[idx]
