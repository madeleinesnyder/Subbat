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
        self.Goals = []
        self.ARP_dict = defaultdict(list)
        self.subgoal_locations = [] # TODO Write this function

    def store(self,arg1,arg2,arg3):
        '''
        Stores each state-action-reward sample from the environment
        If the capacity of the buffer is exceeded, delete the first sample
        '''
        # arg1 - state, arg2 - action, arg3 - reward.
        # Convert to a tuple to store in memories
        memory = [arg1, arg2, arg3]
        memory = tuple(memory)
        # If this new memory is already in the set of the memories, don't put it in.
        self.memories.add(memory)
        if len(self.memories) > self.capacity:
            self.memories = self.memories[1:]

    def find_Goals(self):
        '''
        Find the states that corrospond to goals.
        '''
        if len(self.memories) < 1:
            return self.Goals
        for memory in self.memories:
            self.ARP_dict[memory[0]].append(memory[1:])

        for key, value in self.ARP_dict.items():
            unique_arps = set(value)
            if len(unique_arps) > 1:
                self.Goals.append(key)
        return self.Goals

    def attempt_action(self, env, action, obs):
        '''
        This function attempts different actions to get an xy location for ALE (4 and 5)
        '''
        env = env.unwrapped
        clone_state = env.clone_full_state()
        action = action
        next_obs,reward,done,info = env.step(action)
        rgb_coords = next_obs-obs
        if np.sum(rgb_coords) != 0:
            env.restore_full_state(clone_state)
            return rgb_coords
        else:
            env.restore_full_state(clone_state)
            return np.zeros(1)


    def get_Goal_xy(self,env,observation):
        '''
        How to find key?? TODO
        Get the xy location of the goal from the state
        Fix this unwrapping shit. first check 4 (left); then check 5 (down)
        '''
        obs = observation
        # Change this such that you have at least one action that will always work
        # He might be on the ladder in which case jumping will yeild rgb_coords = 0
        for action in [4,5,11]:
            rgb_coords = self.attempt_action(env,action,obs)
            if np.sum(rgb_coords) > 0:
                return rgb_coords
        nonzero_coords = np.where(rgb_coords[:,:,0] != 0)
        [mean_x,mean_y] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
        coords = (int(np.ceil(mean_x)),int(np.ceil(mean_y)))
        self.subgoal_locations.append(coords)

    def at_subgoal(self,env,observation,goal):
        '''
        Is ALE near enough to a subgoal?
        '''
        #goal_xy = convertToSubgoalCoordinates(goal)
        goal_xy = goal
        pdb.set_trace()
        for action in [4,5,11]:
            location_ale = self.attempt_action(env,action,observation)
            if int(np.sum(location_ale)) > 0:
                return location_ale
            pdb.set_trace()
        nonzero_coords = np.where(location_ale[:,:,0] != 0)
        [mean_x,mean_y] = [np.mean(nonzero_coords[0]),np.mean(nonzero_coords[1])]
        coords = (int(np.ceil(mean_x)),int(np.ceil(mean_y)))
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

    def random_Goal(self):
        '''
        Get a random subgoal to initialize the exploration run in env.
        '''
        idx = np.random.randint(len(self.subgoal_locations))
        return self.subgoal_locations[idx]
